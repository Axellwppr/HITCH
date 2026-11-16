"""Robot link server over the controller's motion channel (memory-mapped files).

Translation only: wire <-> canonical joint order, and the segment bookkeeping
that lets a client tell where its append went. Holding the tail for the
policy's look-ahead is the controller's job; no motion semantics here.

Run after the controller is up:  python -m wbc_link_server.launch.control --attach RUNTIME_DIR

With ``--pause-key`` (``scripts/robot link`` on the real robot; never in simulation) the space bar in this terminal pauses
the robot for good: appends after it are answered as accepted but never reach the controller, so to a client they
stay queued, as if the motion before them were still running; finger commands likewise. The controller keeps
running (state stays fresh) and plays out what it already holds. Restart this server to move on.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import logging
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np

from wbc.core import logs
from wbc.core.errors import ErrorCode, WbcError
from wbc.core.jsonhttp import JsonServer, wait_for_signal, write_json_atomic
from wbc.core.layout import CANONICAL_TO_WIRE, FPS, JOINTS, SIDES, WIRE_TO_CANONICAL
from wbc.core.wire import unpack
from wbc.link import fingers as finger_commands
from wbc.link.protocol import LinkInfo, Phase, RobotState, Segment

log = logging.getLogger("wbc_link_server.launch.control")

ACK_TIMEOUT_S = 2.0
KEEP_SEGMENTS = 32  # reported in state; all of an epoch's segments are kept for idempotent appends


def to_canonical(wire) -> np.ndarray:
    return np.asarray(wire, float)[list(WIRE_TO_CANONICAL)]


def pose_to_canonical(pose) -> np.ndarray:
    pose = np.asarray(pose, float)
    return np.r_[pose[:7], to_canonical(pose[7:])]


def poses_to_wire(frames) -> np.ndarray:
    frames = np.asarray(frames, float)
    out = frames.copy()
    out[:, 7:] = frames[:, 7:][:, list(CANONICAL_TO_WIRE)]
    return out


def process_alive(pid: int) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return b"deploy.py" in command


def _read(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _channel(runtime: Path):
    from motion_channel import MotionChannel  # controller package

    return MotionChannel(runtime / "channel")


class ChannelLink:
    """The robot link of one controller runtime. Only one may be attached at a time (``link.lock``)."""

    def __init__(self, runtime: Path):
        self.runtime = Path(runtime)
        self._owner = (self.runtime / "link.lock").open("a")
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._owner.close()
            raise RuntimeError(f"another link server is attached to {self.runtime}") from None
        self.config = json.loads((self.runtime / "runtime.json").read_text())
        self.channel = _channel(self.runtime)
        self.lock = threading.Lock()
        self.segments_file = self.runtime / "segments.json"
        self.pending_file = self.runtime / "pending.json"
        self.segments = [Segment(**s) for s in _read(self.segments_file, [])]
        # An append is recorded before it is handed to the controller (write-ahead), so a restart of this
        # server can always tell what happened to it: never published, still waiting, or acknowledged.
        pending = _read(self.pending_file, None)
        self.pending: tuple[int, str, int] | None = tuple(pending) if pending else None  # (serial, id, frames)
        if self.pending is not None and self._published_serial() < self.pending[0]:
            self._clear_pending()  # we stopped before publishing it
        # After pause(): appends answered but held here, never sent (to clients they follow the real tail)
        self.paused = threading.Event()
        self.held: list[Segment] = []
        self.held_tail = None  # [36] canonical pose ending the held appends
        self.held_hands: dict[str, str] = {}

    def pause(self) -> None:
        """Hold every later append and finger command for good; what the controller already has plays out."""
        self.paused.set()

    def close(self) -> None:
        self.channel.close()
        self._owner.close()

    def _published_serial(self) -> int:
        return int(self.channel.header["serial"][0])

    def _clear_pending(self) -> None:
        self.pending = None
        self.pending_file.unlink(missing_ok=True)

    @property
    def epoch(self) -> str:
        return self.config["runtime_id"]

    def info(self) -> LinkInfo:
        horizon = int(self.config["metadata"]["horizon"])
        return LinkInfo(self.epoch, self.config["backend"], FPS, JOINTS, horizon, bool(self.config.get("paused")))

    def state(self) -> RobotState:
        with self.lock:
            return self._state()

    def _state(self) -> RobotState:
        snapshot = self.channel.read()
        if not process_alive(self.config["policy_pid"]):
            return RobotState(self.epoch, Phase.FAULT)
        if snapshot is None:
            return RobotState(self.epoch, Phase.NOT_TRACKING)
        self._settle(snapshot)
        # after pause(), the held appends as a client was told: after the real tail, never reached
        held = bool(self.held)
        return RobotState(
            self.epoch,
            Phase.TRACKING,
            tick=int(snapshot["tick"]),
            age_s=max(0.0, time.monotonic() - float(snapshot["received"])),
            q=to_canonical(snapshot["q"]),
            dq=to_canonical(snapshot["dq"]),
            quat_wxyz=np.asarray(snapshot["quat"], float),
            gyro=np.asarray(snapshot["gyro"], float),
            q_target=to_canonical(snapshot["target"]),
            cursor=int(snapshot["index"]),
            tail=self.held[-1].end if held else int(snapshot["length"]) - 1,
            revision=int(snapshot["revision"]) + len(self.held),
            cursor_pose=pose_to_canonical(snapshot["reference"]),
            tail_pose=self.held_tail if held else pose_to_canonical(snapshot["tail"]),
            segments=tuple([*self.segments, *self.held][-KEEP_SEGMENTS:]),
            pending=None if self.pending is None else self.pending[1],
        )

    def _settle(self, snapshot) -> bool | None:
        """Resolve the pending append once the controller acknowledged it: its frames end the reference."""
        if self.pending is None or int(snapshot["ack"]) < self.pending[0]:
            return None
        _, segment_id, count = self.pending
        accepted = int(snapshot["result"]) == 1
        if accepted and all(s.segment_id != segment_id for s in self.segments):  # recorded once, even after a crash
            length = int(snapshot["length"])
            self.segments = [*self.segments, Segment(segment_id, length - count, length - 1)]
            write_json_atomic(self.segments_file, [s.to_dict() for s in self.segments])
        self._clear_pending()
        return accepted

    def hand(self, epoch: str, fingers: dict[str, str]) -> dict[str, str]:
        """Command each named hand ``open`` or ``closed`` (``hands.json``, read by the simulator bridge); returns
        every hand's command. Applied at once, between body frames."""
        with self.lock:
            if epoch != self.epoch:
                raise WbcError(ErrorCode.EPOCH_CHANGED, "controller restarted; reconnect")
            if not set(fingers) <= set(SIDES) or not set(fingers.values()) <= set(finger_commands.STATES):
                raise WbcError(ErrorCode.INVALID_REQUEST, 'fingers maps "left"/"right" to "open" or "closed"')
            current = _read(self.runtime / finger_commands.FILE, {"serial": 0, "hands": finger_commands.all_open()})
            if self.paused.is_set():  # answered, never applied
                self.held_hands.update(fingers)
                log.info("paused: finger command %s held", fingers)
                return {**current["hands"], **self.held_hands}
            value = {"serial": int(current["serial"]) + 1, "hands": {**current["hands"], **fingers}}
            write_json_atomic(self.runtime / finger_commands.FILE, value)
            return value["hands"]

    def append(self, segment_id: str, epoch: str, revision: int, frames) -> Segment:
        """Append ``frames`` after the reference tail; idempotent per ``segment_id``."""
        with self.lock:
            if epoch != self.epoch:
                raise WbcError(ErrorCode.EPOCH_CHANGED, "controller restarted; reconnect")
            known = next((s for s in [*self.segments, *self.held] if s.segment_id == segment_id), None)
            if known is not None:
                return known
            state = self._state()
            if not state.tracking:
                raise WbcError(ErrorCode.NOT_TRACKING, f"controller is {state.phase.value}")
            if self.pending is not None:
                raise WbcError(ErrorCode.BUSY, "previous append not yet acknowledged")
            if revision != state.revision:
                raise WbcError(ErrorCode.STALE_TAIL, f"reference revision is {state.revision}, not {revision}")
            frames = np.asarray(frames, float)
            if frames.ndim != 2 or frames.shape[1] != 36 or len(frames) == 0 or not np.isfinite(frames).all():
                raise WbcError(ErrorCode.INVALID_SEGMENT, "expected finite [N >= 1, 36] frames")
            if self.paused.is_set():  # accepted as far as the client can tell; the controller never sees it
                segment = Segment(segment_id, state.tail + 1, state.tail + len(frames))
                self.held.append(segment)
                self.held_tail = frames[-1].copy()
                log.info("paused: segment %s (%d frames) held", segment_id, len(frames))
                return segment
            record = (self._published_serial() + 1, segment_id, len(frames))
            write_json_atomic(self.pending_file, list(record))  # a failed write leaves nothing pending
            self.pending = record
            try:
                serial = self.channel.submit(poses_to_wire(frames), state.revision)
            except ValueError as error:  # too many frames for the channel
                self._clear_pending()
                raise WbcError(ErrorCode.INVALID_SEGMENT, str(error)) from error
            except RuntimeError as error:  # the channel still holds an unacknowledged request
                self._clear_pending()
                raise WbcError(ErrorCode.BUSY, str(error)) from error
            except Exception:
                self._clear_pending()
                raise
            assert serial == record[0], "only this server writes the channel"
            deadline = time.monotonic() + ACK_TIMEOUT_S
            while time.monotonic() < deadline:
                snapshot = self.channel.read()
                if snapshot is not None:
                    outcome = self._settle(snapshot)
                    if outcome is True:
                        return self.segments[-1]
                    if outcome is False:
                        raise WbcError(ErrorCode.STALE_TAIL, "controller rejected the segment: the reference moved")
                time.sleep(0.002)
            raise WbcError(ErrorCode.PENDING, "not acknowledged yet; the segment id shows up in state")


def routes(link: ChannelLink) -> dict:
    def hand(payload):
        return link.hand(payload["epoch"], payload["fingers"])

    def append(payload):
        frames = unpack(payload["frames"])
        return link.append(payload["segment_id"], payload["epoch"], int(payload["revision"]), frames).to_dict()

    return {
        ("GET", "/info"): lambda: link.info().to_dict(),
        ("GET", "/state"): lambda: link.state().to_dict(),
        ("POST", "/append"): append,
        ("POST", "/hand"): hand,
    }


def pause_key(link: ChannelLink) -> None:
    """Pause ``link`` when the space bar is pressed in this terminal (cbreak mode; restored on exit)."""
    stdin = sys.stdin.fileno()
    saved = termios.tcgetattr(stdin)
    tty.setcbreak(stdin)
    atexit.register(termios.tcsetattr, stdin, termios.TCSADRAIN, saved)

    def listen():
        while (key := sys.stdin.read(1)) != " ":
            if not key:  # end of input: nothing can press space any more
                log.warning("the pause key's terminal closed: space no longer pauses")
                return
        link.pause()
        log.warning("PAUSED: new motions and finger commands are held and never sent; restart this server to go on")

    threading.Thread(target=listen, name="pause-key", daemon=True).start()
    log.info("press space to pause: what the controller holds plays out, nothing new is sent")


def serve(runtime: Path, port: int, endpoint_file: Path | None, host: str = "127.0.0.1", pause: bool = False) -> None:
    if pause and not sys.stdin.isatty():  # before the link, the server and the endpoint file: nothing to undo
        raise SystemExit("--pause-key needs this server in a terminal")
    link = ChannelLink(runtime)
    server = None
    try:
        server = JsonServer(routes(link), host, port, journal=Path(runtime) / "link.jsonl").start()
        log.info("robot link at %s (epoch %s)", server.url, link.epoch)
        if endpoint_file:
            write_json_atomic(endpoint_file, {"control": server.url, "epoch": link.epoch})
        if pause:
            pause_key(link)
        wait_for_signal(lambda: process_alive(link.config["policy_pid"]))
        if not process_alive(link.config["policy_pid"]):
            log.warning("the controller (pid %s) has exited; the link server stops", link.config["policy_pid"])
    finally:
        if server is not None:
            server.close()
        link.close()  # releases link.lock: the next server may attach


def wait_for_tracking(runtime: Path, timeout: float) -> None:
    """Wait until the controller publishes tracking state (operator pressed start and A)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (runtime / "runtime.json").exists() and (runtime / "channel" / "ready").exists():
            channel = _channel(runtime)
            tracking = channel.read() is not None
            channel.close()
            if tracking:
                return
        time.sleep(0.05)
    raise TimeoutError("controller is not tracking; start it and press A first")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attach", type=Path, required=True, help="runtime directory of a running controller")
    parser.add_argument("--port", type=int, default=18767)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--endpoint-file", type=Path)
    parser.add_argument("--wait", type=float, default=0.0, help="seconds to wait for tracking before serving")
    parser.add_argument(
        "--pause-key",
        action="store_true",
        help="space pauses for good: later appends are held, never sent (the real robot only)",
    )
    args = parser.parse_args()
    runtime = args.attach.resolve()  # a run directory, not the link to the latest one, which a new run moves
    logs.setup("control", logs.session_of(runtime))
    if args.wait:
        wait_for_tracking(runtime, args.wait)
    check_controller(runtime)
    serve(runtime, args.port, args.endpoint_file, args.host, args.pause_key)


def check_controller(runtime: Path) -> None:
    """Refuse to start unless the controller of ``runtime`` is running."""
    try:
        pid = json.loads((runtime / "runtime.json").read_text())["policy_pid"]
    except (OSError, ValueError, KeyError):
        raise SystemExit(f"{runtime} holds no controller runtime; start the controller first") from None
    if not process_alive(pid):
        raise SystemExit(f"the controller of {runtime} (pid {pid}) is not running; start it first")


if __name__ == "__main__":
    main()
