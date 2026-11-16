"""The public interface: one Session per robot connection, living in one process.

The loop is observe -> (perceive) -> plan -> execute -> wait. The
session keeps its state in memory and is the only writer of its directory.
``poll`` reads the robot once: it adopts a controller restart, settles and
advances runs, and classifies the session; the host calls it continuously
(``monitored``), otherwise ``status`` and ``wait`` do. Commands that change
anything are serialised by one command lock; ``status`` and ``wait`` never
wait for them. File names in results are relative to the session directory
(the host serves them at ``/files/``).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from wbc.core import limits as L
from wbc.core import target as targets
from wbc.core.errors import UNCERTAIN, Cause, ErrorCode, WbcError, invalid
from wbc.core.eye import Eye
from wbc.core.hands import GRIP_S
from wbc.core.layout import FPS, JOINTS, SIDES
from wbc.core.params import parse
from wbc.core.request import Request, parse_request
from wbc.core.types import RUN_ENDED, Posture, RunState, StartState, posture_for_height
from wbc.link.http import HttpLink
from wbc.link.local import start_simulator, stop_process_group
from wbc.link.logged import LoggedLink
from wbc.link.protocol import LinkInfo, Phase, RobotLink, RobotState
from wbc.paths import data_root, timestamped
from wbc.planner import plan as plan_motion
from wbc.robot.hands import current_hand
from wbc.session import observation as observations
from wbc.session.perception import draw_points
from wbc.session.perception import perceive as perceive_points
from wbc.session.runs import RunBook, public, segment_id
from wbc.session.store import Journal, SessionFiles, read_json, write_json

__all__ = ["OPS", "Session", "SessionState", "new_directory"]

# The operations a session host serves (its /call) and the CLI's commands: each the Session method of that name
OPS = ("status", "observe", "perceive", "plan", "execute", "grip", "look", "aim", "wait", "close", "show", "log")

BUSY_RETRY_S = 1.0  # an append the link is not ready for is retried this long
RESEND_S = 1.0  # an unknown run's append is sent again at most this often
WAIT_POLL_S = 0.1
KIND_NAMES = {"obs": "observation", "plan": "plan", "run": "run"}  # artifact kind -> name in results


class SessionState(str, Enum):
    READY = "ready"
    NOT_TRACKING = "not_tracking"
    STALE = "stale"
    FAULT = "fault"
    CLOSED = "closed"


# A session state that makes a command fail as the robot side being down (the agent loop stops on it).
DOWN_CAUSES = {SessionState.FAULT: Cause.CONTROLLER_DOWN, SessionState.STALE: Cause.ROBOT_UNREACHABLE}


@dataclass(frozen=True)
class Snapshot:
    state: SessionState
    robot: RobotState | None
    detail: str = ""


def new_directory(root: Path) -> Path:
    """A fresh ``ROOT/sessions/ses-YYYYMMDD-HHMMSS[-n]`` directory."""
    try:
        return timestamped(Path(root) / "sessions", "ses")
    except FileExistsError:
        raise WbcError(ErrorCode.INTERNAL, "could not allocate a session directory") from None


class Session:
    def __init__(self, directory: Path, link: RobotLink):
        self.dir = Path(directory)
        self.record = read_json(self.dir / "session.json")
        self.files = SessionFiles(self.dir)
        self.journal = Journal(self.dir / "journal.jsonl")
        self.link = LoggedLink(link, self.journal)
        self.runs = RunBook(self.files, self.journal)
        self.monitored = False  # set by the host, whose monitor thread polls
        self._command = threading.RLock()  # grip holds it across its stand-still
        self._state = threading.RLock()
        self._snapshot = Snapshot(SessionState.STALE, None, "not polled yet")
        self._tick = -1  # newest tracking tick applied in the current epoch
        self._resent: dict[str, float] = {}

    # ----------------------------------------------------------------- lifecycle
    @property
    def id(self) -> str:
        return self.record["id"]

    @property
    def info(self) -> LinkInfo:
        return LinkInfo.from_dict(self.record["info"])

    @classmethod
    def connect(
        cls,
        target: str,
        *,
        vision: str | None = None,
        directory: Path | None = None,
        root: Path | None = None,
        timeout: float = 120.0,
    ) -> Session:
        """Attach to ``http://host:port`` or start a local simulator with ``sim://SCENE?paused=1``."""
        directory = Path(directory) if directory else new_directory(root or data_root())
        directory.mkdir(parents=True, exist_ok=True)
        record = {"id": directory.name, "target": target, "created": time.time(), "closed": False}
        process = None
        try:
            if targets.is_sim(target):
                process, endpoints = start_simulator(directory / "runtime", target, timeout)
                record["owned"] = {"pid": process.pid}
                control, vision = endpoints["control"], endpoints.get("vision")
            elif urlparse(target).scheme in ("http", "https"):
                control = target
            else:
                raise invalid("target must be http://HOST:PORT or sim://SCENE")
            link = HttpLink(control, vision)
            info = link.info()
            if info.fps != FPS or tuple(info.joints) != JOINTS:
                raise WbcError(ErrorCode.INVALID_REQUEST, "link uses a different frame rate or joint order")
            record.update(control=control, vision=vision, info=info.to_dict())
            write_json(directory / "session.json", record)
        except BaseException:
            if process is not None:
                stop_process_group(process.pid)
            raise
        session = cls(directory, link)
        session._event("connect", target=target, epoch=info.epoch)
        return session

    @classmethod
    def load(cls, directory: Path, link: RobotLink | None = None) -> Session:
        """Reopen a session after its host restarted; unfinished runs are settled from the robot."""
        record = read_json(Path(directory) / "session.json")
        if record["closed"]:
            raise WbcError(ErrorCode.SESSION_STATE, f"session {record['id']} is closed")
        session = cls(directory, link or HttpLink(record["control"], record.get("vision")))
        session._event("reopen")
        return session

    @classmethod
    def with_link(cls, link: RobotLink, *, root: Path | None = None) -> Session:
        """A session over an in-process link (FakeLink)."""
        directory = new_directory(root or data_root())
        info = link.info()
        record = {"id": directory.name, "target": "inprocess", "created": time.time()}
        write_json(directory / "session.json", {**record, "closed": False, "info": info.to_dict()})
        return cls(directory, link)

    def close(self) -> dict:
        with self._command:
            if not self.record["closed"]:
                owned = self.record.get("owned")
                if owned:
                    stop_process_group(owned["pid"])
                self._update(closed=True)
                self._event("close")
            self._set(Snapshot(SessionState.CLOSED, None))
        return {"state": SessionState.CLOSED.value}

    def served_at(self, url: str) -> None:
        """Remember the session's endpoint, so a restarted host serves it at the same URL."""
        self._update(url=url)

    # ------------------------------------------------------------------- state
    def poll(self) -> Snapshot:
        """Read the robot once: adopt a controller restart, settle and advance runs, classify."""
        if self.record["closed"]:
            return self._set(Snapshot(SessionState.CLOSED, None))
        try:
            robot = self.link.state()
            if robot.epoch != self.info.epoch and not self._adopt(robot.epoch):
                return self._snapshot  # a read from before the restart arrived late
        except Exception as error:  # whatever went wrong, the robot is not known now
            return self._set(Snapshot(SessionState.STALE, None, str(error)))
        with self._state:
            if robot.epoch != self.info.epoch:
                return self._snapshot
            if not robot.tracking or robot.tick >= self._tick:  # concurrent polls may finish out of order
                self._tick = max(self._tick, robot.tick)
                self.runs.update(robot)
            snapshot = self._set(self._classify(robot))
        self._resend(robot)
        return snapshot

    def _adopt(self, epoch: str) -> bool:
        """Take a controller restart once the link itself reports the new epoch."""
        info = self.link.info()
        if info.epoch != epoch:
            return False
        with self._state:
            self._update(info=info.to_dict())
            self._tick = -1
            self._event("epoch_changed", epoch=epoch)
        return True

    def _classify(self, robot: RobotState) -> Snapshot:
        if robot.phase == Phase.FAULT:
            return Snapshot(SessionState.FAULT, robot, "controller reports a fault")
        if robot.phase == Phase.NOT_TRACKING:
            return Snapshot(SessionState.NOT_TRACKING, robot, "start the controller and enter tracking")
        if robot.age_s is not None and robot.age_s > L.STATE_AGE and not self.info.paused:
            return Snapshot(SessionState.STALE, robot, f"robot state is {robot.age_s:.2f} s old")
        return Snapshot(SessionState.READY, robot)

    def _set(self, snapshot: Snapshot) -> Snapshot:
        with self._state:
            self._snapshot = snapshot
        return snapshot

    def status(self) -> dict:
        """Session state, the kind of robot, how much motion is still queued, and the latest ids."""
        snapshot = self._snapshot if self.monitored else self.poll()
        out = {"state": snapshot.state.value, "robot": self.info.backend}
        if snapshot.detail:
            out["detail"] = snapshot.detail
        robot = snapshot.robot
        if robot is not None and robot.tracking:
            out["queued_s"] = robot.queued_s
        latest = {f"{name}_id": self.files.latest(kind) for kind, name in KIND_NAMES.items()}
        out["latest"] = {key: ident for key, ident in latest.items() if ident}
        return out

    def wait(self, run: str | None = None, timeout: float = 30.0) -> dict:
        """A run (default: the latest) once it finishes, or as it is after ``timeout`` seconds (0: now)."""
        run = run or self._latest("run")
        deadline = time.monotonic() + timeout
        while True:
            if not self.monitored:
                self.poll()
            with self._state:
                result = public(self.runs.get(run))
            if RunState(result["state"]) in RUN_ENDED or time.monotonic() >= deadline:
                return result
            time.sleep(WAIT_POLL_S)

    # -------------------------------------------------------------- operations
    def observe(self, since: str | None = None) -> dict:
        """Robot state and RGB. Allowed while moving. ``since`` (an earlier
        observation of this session) adds where the new frame O lies in that one's (``since`` in the result)."""
        with self._command:
            self._open()
            earlier = None if since is None else observations.load(self.files, since)
            obs = observations.capture(self.link, self.files, self._tail_intent, earlier, self.eye().to_json())
            self._event("observe", observation=obs.id)
            out = {"observation_id": obs.id, "queued_s": obs.state.queued_s, "body": obs.record["body"]}
            if obs.record.get("since"):
                out["since"] = obs.record["since"]
            fingers = self.fingers()
            if fingers is not None:
                out["body"] = {
                    **out["body"],
                    "hands": {s: {**h, "fingers": fingers[s]} for s, h in out["body"]["hands"].items()},
                }
            if obs.record.get("eye"):
                out["eye"] = obs.record["eye"]
            if obs.record.get("vision_error"):
                out["vision_error"] = obs.record["vision_error"]
            if obs.has_vision:
                out["image_size"] = [obs.camera.width, obs.camera.height]
            out["images"] = [self._relative(path) for path in obs.images()]
            return out

    def perceive(self, points, show: bool = False) -> dict:
        """3D points in the latest observation's frame O for normalized image points ``[x, y]`` (0–1000); ``show``
        adds the camera image with the points marked and numbered."""
        with self._command:
            self._open()
            obs = observations.load(self.files, self._latest("obs"))
            if not obs.has_vision:
                raise WbcError(ErrorCode.PRECONDITION, f"{obs.id} has no RGB-D")
            located = perceive_points(obs.depth(), obs.camera, points)
            out = {"observation_id": obs.id, "points": [{k: v for k, v in p.items() if k != "pixel"} for p in located]}
            if show:
                image = obs.file(f"perceive-{len(list(obs.path.glob('perceive-*.png'))) + 1}.png")
                draw_points(obs.rgb(), located, image, f"perceive on {obs.id}: points in request order")
                out["images"] = [self._relative(image)]
            with obs.file("percepts.jsonl").open("a") as stream:
                stream.write(json.dumps(out) + "\n")
            self._event("perceive", observation=obs.id)
            return out

    def plan(self, request: dict, *, slower: bool = False) -> dict:
        """Plan a request (docs/REFERENCE.md) from the reference tail, in the frame O of the latest observation;
        ``slower`` stretches its body change and walk timing.

        Every coordinate (waypoints, walk targets, footsteps) is in that O. The robot must be idle, and the
        observation taken while idle with nothing appended since: then O sits at the tail, with the IMU heading, and
        maps to the reference world W exactly.
        """
        parsed = parse_request(request)  # argument errors before the robot is asked
        if parsed.eye is not None:
            raise invalid("plan: eye is not planned; turn the camera with look (the agent's execute does both)")
        return self._plan(parsed, request, slower)

    def _plan(self, parsed: Request, request: dict, slower: bool = False) -> dict:
        with self._command:
            self._open()
            robot = self._ready("plan")
            if robot.queued_s > 0:
                raise WbcError(
                    ErrorCode.PRECONDITION,
                    f"still moving ({robot.queued_s:.1f} s queued); plan when idle",
                    cause=Cause.MOVING,
                )
            obs = self._anchor(robot)
            to_world = obs.to_world  # O -> W
            posture, torso, intents = self._tail_intent(robot)
            start = StartState(
                to_world.inverse().apply_poses(robot.tail_pose),
                posture,
                torso=torso,
                hands=observations.intents_in(intents, to_world.inverse()),
            )
            planned = plan_motion(parsed, start=start, frame=to_world, horizon=self.info.horizon, slower=slower)
            motion, report = planned.motion, planned.report
            meta = {
                "state": "ready" if report.passed else "rejected",
                "anchor": {"epoch": robot.epoch, "revision": robot.revision},
                "observation": obs.id,
                "end": {
                    "posture": motion.end_posture.value,
                    "torso": list(motion.end_torso or start.torso),
                    "hands": observations.intents_in(planned.hands, to_world),
                },
            }
            with self.files.create("plan") as (path, ident):
                write_json(path / "plan.json", meta)
                write_json(path / "request.json", {"request": request, "slower": slower})
                write_json(path / "start.json", {**start.to_dict(), "to_world": to_world.to_list()})
                write_json(path / "checks.json", report.to_dict())
                np.savez_compressed(
                    path / "motion.npz",
                    frames=motion.frames,
                    labels=motion.labels(),
                    support=motion.support if motion.support is not None else np.zeros((0, 2), bool),
                    keyframes=np.asarray(motion.keyframes, int),
                    world=planned.world,
                )
                write_json(
                    path / "diagnostics.json",
                    {"limits": planned.limits.names, "body": planned.notes, **motion.diagnostics},
                )
            self._event("plan", plan=ident, state=meta["state"])
            return {
                "plan_id": ident,
                "state": meta["state"],
                "failures": report.findings(blocking=True),
                "warnings": report.findings(blocking=False),
                "duration_s": round(motion.duration, 2),
                "end_posture": motion.end_posture.value,
                "body": planned.notes,
            }

    def execute(self, plan: str | None = None) -> dict:
        """Append a ready plan (default: the latest) to the robot; returns its run."""
        with self._command:
            self._open()
            plan = plan or self._latest("plan")
            path = self.files.get(plan, "plan")
            meta = read_json(path / "plan.json")
            if meta["state"] != "ready":
                raise WbcError(ErrorCode.PLAN_REJECTED, f"{plan} failed its checks; see its failures")
            robot, anchor = self._ready("execute"), meta["anchor"]
            if (anchor["epoch"], anchor["revision"]) != (robot.epoch, robot.revision):
                raise WbcError(ErrorCode.PLAN_STALE, "the reference changed since planning; plan again")
            with self._state:
                run = self.runs.create(plan, robot, meta["end"])
            self._append(run, retry_busy=True)
            if self.runs.runs[run["run_id"]]["state"] == RunState.UNKNOWN.value:
                self.poll()  # usually settles it at once
            with self._state:
                run = self.runs.get(run["run_id"])
            self._event("execute", plan=plan, run=run["run_id"], state=run["state"])
            return public(run)

    def grip(self, fingers: dict, hold_s: float = GRIP_S) -> dict:
        """Close or open hands' fingers now (side -> "open"/"closed"), then hold still ``hold_s`` (at least GRIP_S) so
        they move: returns the hold's run (with ``fingers``, every hand's state). The robot must be idle; the hand
        must have fingers."""
        if current_hand().grip is None:
            raise invalid(f"the {current_hand().name} hands have no finger control")
        if not fingers or not set(fingers) <= set(SIDES) or not set(fingers.values()) <= {"open", "closed"}:
            raise invalid('fingers maps "left" and/or "right" to "open" or "closed"')
        if not GRIP_S <= hold_s <= GRIP_S + 10.0:
            raise invalid(f"hold_s must be {GRIP_S}..{GRIP_S + 10.0} s")
        with self._command:
            self._open()
            robot = self._ready("grip")
            if robot.queued_s > 0:
                raise WbcError(
                    ErrorCode.PRECONDITION,
                    f"still moving ({robot.queued_s:.1f} s queued); grip when idle",
                    cause=Cause.MOVING,
                )
            state = self.link.hand(robot.epoch, dict(fingers))
            self._update(fingers=state)
            self._event("grip", fingers=dict(fingers))
            # Simulated time runs only with motion queued: a short stand-still lets the fingers close or open. No
            # other command comes between (the command lock is held); any failure here is after the fingers moved.
            try:
                self.observe()
                plan = self._plan(Request(wait_s=hold_s), {"wait_s": hold_s})
                if plan["state"] != "ready":
                    raise WbcError(ErrorCode.PLAN_REJECTED, str(plan["failures"]))
                run = self.execute(plan["plan_id"])
            except WbcError as error:
                raise WbcError(
                    error.code,
                    f"holding still for the fingers failed: {error.message}",
                    **error.details,
                    fingers_commanded=True,
                ) from error
            return {**run, "fingers": state}

    def look(self, eye: dict) -> dict:
        """Set the head camera's mount (``{yaw_deg, pitch_deg}``, wbc.core.eye) for the observations from now on;
        allowed while moving. Returns the setting."""
        setting = _eye(eye, "look")
        with self._command:
            self._open()
            self._update(eye=setting.to_json())
            self._event("look", eye=setting.to_json())
            return setting.to_json()

    def aim(self, eye: dict) -> dict:
        """Start turning the head camera toward ``{yaw_deg, pitch_deg}`` while the body moves, without waiting
        (a real mount takes seconds); the setting is still ``look``'s, and the observation settles the camera on it.
        Returns the setting."""
        setting = _eye(eye, "aim")
        with self._command:
            self._open()
            self.link.aim(setting.to_json())
            return setting.to_json()

    def eye(self) -> Eye:
        """The head camera's current mount setting (the default until ``look``)."""
        return Eye.from_json(self.record.get("eye"))

    def fingers(self) -> dict | None:
        """Each hand's commanded fingers ("open"/"closed"), or None for hands without finger control."""
        if current_hand().grip is None:
            return None
        return {**dict.fromkeys(SIDES, "open"), **(self.record.get("fingers") or {})}

    def _append(self, run: dict, retry_busy: bool) -> None:
        """Send a run's frames; record what is known of the outcome."""
        run_id, deadline = run["run_id"], time.monotonic() + BUSY_RETRY_S
        with np.load(self.files.get(run["plan_id"], "plan") / "motion.npz", allow_pickle=False) as data:
            frames = data["world"][1:]
        while True:
            try:
                segment = self.link.append(segment_id(run), run["epoch"], run["revision"], frames)
            except WbcError as error:
                if error.code == ErrorCode.BUSY and retry_busy and time.monotonic() < deadline:
                    time.sleep(0.05)
                    continue
                with self._state:
                    if error.code in UNCERTAIN or (error.code == ErrorCode.BUSY and not retry_busy):
                        self.runs.unknown(run_id, str(error))
                    else:
                        self.runs.refused(run_id, str(error))
                return
            except Exception as error:  # an unexpected failure says nothing about the append; settle it later
                with self._state:
                    self.runs.unknown(run_id, repr(error))
                return
            with self._state:
                self.runs.accepted(run_id, segment)
            return

    def _resend(self, robot: RobotState) -> None:
        """Settle unknown runs the robot has not seen by sending the same (idempotent) append again."""
        now = time.monotonic()
        with self._state:  # the monitor and a command may both get here
            runs = [r for r in self.runs.to_resend(robot) if now - self._resent.get(r["run_id"], -RESEND_S) >= RESEND_S]
            self._resent.update((run["run_id"], now) for run in runs)
        for run in runs:
            self._append(run, retry_busy=False)

    # ------------------------------------------------------------ inspection
    def show(self, identifier: str) -> dict:
        """An artifact or run as stored, for debugging; file names are relative to the session."""
        out = self.files.describe(identifier)
        for key in ("directory", "file"):
            if key in out:
                out[key] = self._relative(out[key])
        return out

    def log(self, count: int = 50) -> list[dict]:
        return self.journal.tail(count)

    # --------------------------------------------------------------- helpers
    def _open(self) -> None:
        if self.record["closed"]:
            raise WbcError(ErrorCode.SESSION_STATE, "session is closed")

    def _ready(self, action: str) -> RobotState:
        snapshot = self.poll()
        if snapshot.state != SessionState.READY:
            cause = DOWN_CAUSES.get(snapshot.state)
            raise WbcError(
                ErrorCode.SESSION_STATE,
                f"cannot {action}: {snapshot.state.value} {snapshot.detail}",
                **({"cause": cause} if cause else {}),
            )
        return snapshot.robot

    def _latest(self, kind: str) -> str:
        identifier = self.files.latest(kind)
        if identifier is None:
            raise WbcError(ErrorCode.PRECONDITION, f"no {kind} in this session yet")
        return identifier

    def _anchor(self, robot: RobotState):
        """The latest observation, if it still describes where the robot is."""
        obs = observations.load(self.files, self._latest("obs"))
        seen = obs.state
        if seen.epoch != robot.epoch or seen.revision != robot.revision:
            raise WbcError(
                ErrorCode.PRECONDITION,
                f"{obs.id} predates the current reference; observe again",
                cause=Cause.REFERENCE_CHANGED,
            )
        if seen.cursor < seen.tail:
            raise WbcError(
                ErrorCode.PRECONDITION, f"{obs.id} was taken while moving; observe again", cause=Cause.OBSERVED_MOVING
            )
        return obs

    def _tail_intent(self, robot: RobotState) -> tuple[Posture, tuple[int, int], dict | None]:
        """Posture, torso lean and the hands' intents (in W) at the reference tail: from an accepted run, else inferred
        from the pose (a reference we did not plan starts from the controller's standard posture, with no intents)."""
        with self._state:
            known = self.runs.end_state(robot)
        if known is not None:
            return known
        return posture_for_height(float(robot.tail_pose[2])), (0, 0), None

    def _relative(self, path) -> str:
        return str(Path(path).relative_to(self.dir))

    def _update(self, **changes) -> None:
        with self._state:
            self.record = {**self.record, **changes}
            write_json(self.dir / "session.json", self.record)

    def _event(self, op: str, **values) -> None:
        self.journal.write(op, **values)


def _eye(eye, op: str) -> Eye:
    """A head camera setting ``{yaw_deg, pitch_deg}`` for ``op`` (look or aim), checked."""
    if not isinstance(eye, dict) or set(eye) != {"yaw_deg", "pitch_deg"}:
        raise invalid(f"{op}: give eye {{yaw_deg, pitch_deg}}")
    return parse(Eye, eye, "eye")
