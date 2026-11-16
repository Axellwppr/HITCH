"""The session host: one session, one local HTTP endpoint.

  python -m wbc.session.host --connect TARGET [--vision URL] [--port P] [--root DIR]   start a new session
  python -m wbc.session.host --resume DIR                                               bring a session back

Both run a supervisor that keeps one worker process serving the session on a
fixed port. If the worker dies other than by ``close``, it is restarted on the
same port from the session directory (unfinished runs are settled from the
robot), so clients keep the same URL. The first line on stdout is
``{"session": URL, "session_id": ...}`` or ``{"error": {...}}``; everything
after that goes to ``host.log`` in the session directory.

Worker endpoints: ``POST /call {"op", "args"}`` runs one Session method,
``GET /health``, and ``GET /files/<path>`` serves artifacts (paths relative to
the session, as results name them).
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

from wbc.core import logs
from wbc.core.errors import ErrorCode, WbcError, invalid
from wbc.core.jsonhttp import JsonServer, wait_for_signal
from wbc.core.process import die_with_parent
from wbc.link.local import stop_process_group
from wbc.paths import data_root
from wbc.session.session import OPS, Session, new_directory
from wbc.session.store import read_json, write_json

log = logging.getLogger("wbc.session.host")

POLL_S = 0.1
CLOSED, SIGNALLED = 0, 3  # worker exit codes; anything else is a crash
RESTARTS, RESTART_WINDOW_S = 3, 60.0  # this many worker exits within the window: give up


class Host:
    """A Session served over HTTP, with a monitor thread polling the robot."""

    def __init__(self, session: Session, port: int = 0):
        self.session = session
        session.monitored = True
        self.closed = threading.Event()
        routes = {("POST", "/call"): self.call, ("GET", "/health"): self.health}
        self.server = JsonServer(routes, port=port, quiet=("/call", "/health"), files=session.dir)
        self.monitor = threading.Thread(target=self._monitor, name="monitor", daemon=True)

    @property
    def url(self) -> str:
        return self.server.url

    def start(self) -> Host:
        self.session.poll()
        self.server.start()
        self.monitor.start()
        return self

    def call(self, payload) -> object:
        if not isinstance(payload, dict):
            raise invalid('body must be {"op": ..., "args": {...}}')
        op, args = payload.get("op"), payload.get("args") or {}
        if op not in OPS:
            raise invalid(f"unknown operation {op!r}; one of {', '.join(OPS)}")
        if not isinstance(args, dict):
            raise invalid("args must be an object")
        method = getattr(self.session, op)
        try:
            inspect.signature(method).bind(**args)
        except TypeError as error:
            raise invalid(f"{op}: {error}") from None
        result = method(**args)
        if op == "close":
            self.closed.set()
        return result

    def health(self) -> dict:
        return {"session_id": self.session.id, "pid": os.getpid(), "supervisor": os.getppid()}

    def _monitor(self) -> None:
        while not self.closed.is_set():
            try:
                self.session.poll()
            except Exception:  # keep monitoring; the next poll may succeed
                log.exception("poll failed")
            self.closed.wait(POLL_S)

    def stop(self) -> None:
        self.closed.set()
        time.sleep(0.2)  # let an in-flight reply (e.g. to close) finish
        self.server.close()


# ------------------------------------------------------------------ processes
def _say(message: dict) -> None:
    print(json.dumps(message), flush=True)


def _log_to(path: Path, *fds: int) -> None:
    """Point the given standard streams at the session's log file."""
    target = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    for fd in fds:
        os.dup2(target, fd)
    os.close(target)


def _error(error: BaseException) -> dict:
    if isinstance(error, WbcError):
        return {"error": error.to_dict()}
    return {"error": {"code": ErrorCode.INTERNAL.value, "message": repr(error)}}


def worker(directory: Path, port: int, connect: str | None, vision: str | None, timeout: float) -> int:
    """Serve one session until it is closed (exit 0) or a signal arrives (exit 3)."""
    session = None
    for sig in (signal.SIGTERM, signal.SIGINT):  # stopped while starting: unwind, so a new simulator is stopped too
        signal.signal(sig, lambda *_: sys.exit(SIGNALLED))
    try:
        if connect:
            session = Session.connect(connect, vision=vision, directory=directory, timeout=timeout)
        else:
            session = Session.load(directory)
        host = Host(session, port).start()
        session.served_at(host.url)
    except BaseException as error:
        log.exception("session host failed to start")
        if connect and session is not None:
            session.close()  # a new session that cannot be served is not left half-open
        _say(_error(error))
        return 1
    _say({"session": host.url, "session_id": session.id})
    _log_to(directory / "host.log", 1)
    log.info("session %s at %s", session.id, host.url)
    wait_for_signal(lambda: not host.closed.is_set())
    closed = host.closed.is_set()
    host.stop()
    return CLOSED if closed else SIGNALLED


def _spawn_worker(directory: Path, port: int, *extra: str) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "wbc.session.host",
            "--worker",
            "--directory",
            str(directory),
            "--port",
            str(port),
            *extra,
        ],
        stdout=subprocess.PIPE,
        preexec_fn=die_with_parent,
    )


def _terminate(process: subprocess.Popen) -> None:
    """SIGTERM to a worker, once: a second one would interrupt the worker's own shutdown."""
    if not getattr(process, "terminated", False) and process.poll() is None:
        process.terminated = True
        process.terminate()


def _reply(process: subprocess.Popen) -> dict:
    """The worker's first line: where it serves, or why it could not start."""
    line = process.stdout.readline()
    process.stdout.close()
    try:
        return json.loads(line)
    except ValueError:
        process.wait()
        return _error(WbcError(ErrorCode.INTERNAL, f"worker exited ({process.returncode}) without a reply"))


def supervise(directory: Path, port: int, first: tuple[str, ...]) -> int:
    """Start the worker, report its first line, then restart it on the same port until it is closed. However the
    supervisor ends after its worker served, a simulator the session started does not outlive it (``_release``). A
    worker that could not start has already stopped what it started, and a session it failed to resume is left as
    it was."""
    served = threading.Event()
    try:
        return _supervise(directory, port, first, served)
    finally:
        if served.is_set():
            _release(directory)


def _release(directory: Path) -> None:
    """Stop the simulator of a session that was not closed, and mark the session closed."""
    try:
        record = read_json(directory / "session.json")
    except (OSError, ValueError):  # the session never started: its worker stopped what it had started
        return
    if record.get("closed") or not record.get("owned"):
        return
    log.warning("session host ending without close; stopping its simulator")
    stop_process_group(record["owned"]["pid"])
    write_json(directory / "session.json", {**record, "closed": True})


def _supervise(directory: Path, port: int, first: tuple[str, ...], served: threading.Event) -> int:
    stopping = threading.Event()
    process = _spawn_worker(directory, port, *first)

    def stop(*_):  # the worker learns too, even while we are still waiting for its first line
        stopping.set()
        _terminate(process)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    reply = _reply(process)
    try:
        _say(reply)
    except BrokenPipeError:  # the command that started us is gone: nobody will use this session
        stop()
    _log_to(directory / "host.log", 1)
    if "error" in reply:
        return 1
    served.set()
    port = urlparse(reply["session"]).port
    exits: deque[float] = deque(maxlen=RESTARTS)
    while True:
        while process.poll() is None:
            if stopping.wait(0.2):
                process.wait()
                return 0
        if stopping.is_set() or process.returncode == CLOSED:  # told to stop before the worker's exit was seen
            return 0
        exits.append(time.monotonic())
        if len(exits) == RESTARTS and exits[-1] - exits[0] < RESTART_WINDOW_S:
            log.error("worker keeps exiting; giving up")
            return 1
        log.warning("worker exited (%s); restarting on port %s", process.returncode, port)
        process = _spawn_worker(directory, port)
        if stopping.is_set():  # told to stop while it was being started
            _terminate(process)
        reply = _reply(process)
        if "error" in reply:
            log.error("worker restart failed: %s", reply["error"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connect", help="new session on http://HOST:PORT or sim://SCENE[?...]")
    parser.add_argument("--vision", help="vision service URL when attaching to a robot")
    parser.add_argument("--port", type=int, default=0, help="port to serve on (default: any free port)")
    parser.add_argument("--root", type=Path, help="data root (default: $WBC_DATA or data/)")
    parser.add_argument("--resume", type=Path, help="bring back the session in this directory")
    parser.add_argument("--timeout", type=float, default=120.0, help="simulator startup timeout")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        logs.setup("host-worker", logs.session_of(args.directory))
        return worker(args.directory.resolve(), args.port, args.connect, args.vision, args.timeout)
    if bool(args.connect) == bool(args.resume):
        _say(_error(invalid("give --connect TARGET or --resume DIR")))
        return 2
    try:
        if args.connect:
            directory, port = new_directory(args.root or data_root()), args.port
            first = ["--connect", args.connect, "--timeout", str(args.timeout)]
            first += ["--vision", args.vision] if args.vision else []
        else:
            directory = args.resume.resolve()
            record = read_json(directory / "session.json")  # a closed one is refused by the worker
            port, first = urlparse(record.get("url", "")).port or args.port, []
    except BaseException as error:
        _say(_error(error))
        return 1
    logs.setup("host", logs.session_of(directory))
    _log_to(directory / "host.log", 2)
    return supervise(directory, port, tuple(first))


if __name__ == "__main__":
    raise SystemExit(main())
