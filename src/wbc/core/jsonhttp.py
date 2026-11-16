"""Minimal JSON over HTTP for the session host and the robot link: a server (routes, error codes, journal)
and the matching client ``request``."""

from __future__ import annotations

import contextlib
import http.client
import json
import logging
import mimetypes
import os
import signal
import threading
import time
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote
from urllib.request import ProxyHandler, Request, build_opener

from wbc.core.errors import ErrorCode, WbcError

log = logging.getLogger(__name__)


def plain_json(value):
    """JSON fallback (``default=``) for numpy arrays and scalars and enums, in results and session records."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"cannot serialise {type(value).__name__}")


STATUS = {
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.INVALID_SEGMENT: 400,
    ErrorCode.STALE_TAIL: 409,
    ErrorCode.EPOCH_CHANGED: 409,
    ErrorCode.NOT_TRACKING: 409,
    ErrorCode.BUSY: 503,
    ErrorCode.PENDING: 202,
    ErrorCode.NOT_FOUND: 404,
}
MAX_BODY = 64 * 1024 * 1024
_OPENER = build_opener(ProxyHandler({}))  # reached through loopback/SSH forwards, never a proxy


def request(url: str, payload: dict | None = None, timeout: float = 5.0):
    """GET (no payload) or POST JSON; returns the result or raises the server's WbcError."""
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    headers = {"Content-Type": "application/json", "X-Request-Id": uuid.uuid4().hex}
    try:
        response = _OPENER.open(Request(url, data=body, headers=headers), timeout=timeout)
    except HTTPError as error:
        response = error
    except (TimeoutError, URLError, ConnectionError, OSError) as error:
        raise WbcError(ErrorCode.UNREACHABLE, f"{url}: {getattr(error, 'reason', error)}") from error
    with response:
        try:
            result = json.load(response)
        except ValueError as error:
            raise WbcError(ErrorCode.INTERNAL, f"{url}: malformed response") from error
        except (TimeoutError, OSError, http.client.HTTPException) as error:  # the request may have been applied
            raise WbcError(ErrorCode.UNREACHABLE, f"{url}: reply lost ({error})") from error
    if not result.get("ok"):
        raise WbcError.from_dict(result.get("error", {}))
    return result["result"]


def fetch(url: str, timeout: float = 30.0) -> bytes:
    """GET a file as bytes (e.g. a session's ``/files/<path>``)."""
    try:
        with _OPENER.open(Request(url), timeout=timeout) as response:
            return response.read()
    except HTTPError as error:
        raise WbcError(ErrorCode.NOT_FOUND, f"{url}: HTTP {error.code}") from error
    except (TimeoutError, URLError, ConnectionError, OSError, http.client.HTTPException) as error:
        raise WbcError(ErrorCode.UNREACHABLE, f"{url}: {getattr(error, 'reason', error)}") from error


class JsonServer:
    """Serve ``routes[(method, path)] -> callable(payload) -> result`` on a thread.

    With ``files``, ``GET /files/<path>`` returns that file under the directory, as is.
    """

    def __init__(
        self,
        routes: dict[tuple[str, str], Callable],
        host: str = "127.0.0.1",
        port: int = 0,
        journal: Path | None = None,
        quiet: tuple[str, ...] = ("/state",),
        files: Path | None = None,
    ):
        lock = threading.Lock()
        root = None if files is None else Path(files).resolve()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                log.debug(fmt, *args)

            def _handle(self, method):
                started = time.monotonic()
                route = routes.get((method, self.path))
                status, payload = 200, None
                try:
                    if route is None:
                        raise WbcError(ErrorCode.NOT_FOUND, f"no route {method} {self.path}")
                    if method == "POST":
                        size = int(self.headers.get("Content-Length", 0))
                        if not 0 < size <= MAX_BODY:
                            raise WbcError(ErrorCode.INVALID_REQUEST, "missing or oversized body")
                        try:
                            payload = json.loads(self.rfile.read(size))
                        except ValueError:
                            raise WbcError(ErrorCode.INVALID_REQUEST, "body is not JSON") from None
                    body = {"ok": True, "result": route(payload) if method == "POST" else route()}
                except WbcError as error:
                    status = STATUS.get(error.code, 500)
                    body = {"ok": False, "error": error.to_dict()}
                except Exception as error:  # report, never hide, unexpected failures
                    log.exception("%s %s failed", method, self.path)
                    status = 500
                    body = {"ok": False, "error": {"code": ErrorCode.INTERNAL.value, "message": repr(error)}}
                try:
                    encoded = json.dumps(body, allow_nan=False, default=plain_json).encode()
                except (TypeError, ValueError) as error:
                    log.exception("%s %s: reply is not JSON", method, self.path)
                    status = 500
                    body = {"ok": False, "error": {"code": ErrorCode.INTERNAL.value, "message": f"bad reply: {error}"}}
                    encoded = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(encoded)
                if journal is not None and (status != 200 or self.path not in quiet):
                    entry = {
                        "time": time.time(),
                        "method": method,
                        "path": self.path,
                        "status": status,
                        "request_id": self.headers.get("X-Request-Id"),
                        "seconds": time.monotonic() - started,
                        "error": body.get("error"),
                    }
                    with lock, journal.open("a") as stream:
                        stream.write(json.dumps(entry) + "\n")

            def _file(self) -> None:
                path = (root / unquote(self.path[len("/files/") :])).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    encoded = json.dumps({"ok": False, "error": {"code": "NOT_FOUND", "message": "no such file"}})
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded.encode())
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(data)

            def do_GET(self):
                if root is not None and self.path.startswith("/files/"):
                    self._file()
                else:
                    self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.url = f"http://{host}:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> JsonServer:
        self.thread.start()
        return self

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def write_json_atomic(path: Path, value, *, end: str = "", **dumps) -> None:
    """Write ``json.dumps(value, **dumps) + end`` to ``path`` atomically (a temporary file beside it, then a rename):
    a reader polling for the file sees nothing or all of it, never a partial file. Every handshake file between
    processes, every endpoint file a launcher polls for, and every session record (``wbc.session.store.write_json``)
    is written with it."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(json.dumps(value, **dumps) + end)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def wait_for_signal(alive: Callable[[], bool] = lambda: True) -> None:
    """Block until SIGTERM/SIGINT/SIGHUP or until ``alive()`` turns false."""
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.wait(0.5) and alive():
        pass
