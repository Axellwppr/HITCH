"""Runs: one appended plan, followed from the robot's reference cursor to the end.

A run's progress is never reported by the robot; it is derived from the
cursor and the run's frame range ``[start, end]``. An append whose outcome is
unknown is settled by the robot: its segment id shows up in ``segments``
(appended) or ``pending`` (still being handed over); otherwise the session
sends the same append again, which the link answers idempotently, for at most
RESEND_WINDOW_S after the run was created: the plan starts from the robot as it
was then, so an append the robot has not taken by that time is given up (lost)
and never sent later. Records live
in memory and in ``runs/run-N.json``; callers serialise access.
"""

from __future__ import annotations

import time

import numpy as np

from wbc.core.types import RUN_ENDED, Posture, RunState
from wbc.link.protocol import Phase, RobotState, Segment
from wbc.robot.kinematics import Kinematics
from wbc.session.store import Journal, SessionFiles, read_json, write_json

APPENDED = frozenset({RunState.QUEUED, RunState.RUNNING, RunState.DONE})
PUBLIC = ("run_id", "plan_id", "state", "progress", "detail", "max_effector_error_m")
RESEND_WINDOW_S = 10.0  # s after its creation during which an unknown run's append may be sent again
APPEND_TIMEOUT_S = 5.0  # s an append may take (the link client's timeout): the last one is sent this long before


def state_of(run: dict) -> RunState:
    return RunState(run["state"])


def segment_id(run: dict) -> str:
    """The id its frames go to the link under: unique across sessions, since a real robot's link server outlives a
    session and returns the segment it already holds for a repeated id. A run without one goes under its run id."""
    return run.get("segment_id", run["run_id"])


def _resendable(run: dict, margin: float = 0.0) -> bool:
    """Whether an unknown run is still young enough for its append to be sent again, ``margin`` s before the end of
    the window (a record without a creation time is not)."""
    return time.time() - run.get("created", 0.0) <= RESEND_WINDOW_S - margin


def public(run: dict) -> dict:
    """What an agent sees of a run."""
    return {key: run[key] for key in PUBLIC if run.get(key) is not None}


class RunBook:
    def __init__(self, files: SessionFiles, journal: Journal):
        self.files, self.journal = files, journal
        self.runs: dict[str, dict] = {}
        self._kin = None  # created on first use
        folder = files.dir / "runs"
        for path in sorted(folder.glob("run-*.json")) if folder.exists() else ():
            run = read_json(path)
            if state_of(run) == RunState.SUBMITTING:  # its submitter is gone: settle it from the robot
                run = {**run, "state": RunState.UNKNOWN.value, "detail": "submitter exited before the reply"}
            self.runs[run["run_id"]] = run

    def get(self, run_id: str) -> dict:
        self.files.get(run_id, "run")
        return self.runs[run_id]

    def active(self) -> list[dict]:
        return [run for run in self.runs.values() if state_of(run) not in RUN_ENDED]

    def _save(self, run: dict) -> dict:
        previous = self.runs.get(run["run_id"])
        self.runs[run["run_id"]] = run
        write_json(self.files.path(run["run_id"]), run)
        self.files.saved(run["run_id"])
        if previous is None or previous["state"] != run["state"]:
            self.journal.write("run", run_id=run["run_id"], state=run["state"], detail=run.get("detail"))
        return run

    def _set(self, run_id: str, state: RunState, detail: str | None = None, **values) -> dict:
        return self._save({**self.runs[run_id], "state": state.value, "detail": detail, **values})

    def create(self, plan_id: str, robot: RobotState, end: dict) -> dict:
        run_id = self.files.new_id("run")
        return self._save(
            {
                "run_id": run_id,
                "segment_id": f"{self.files.dir.name}/{run_id}",
                "plan_id": plan_id,
                "epoch": robot.epoch,
                "revision": robot.revision,
                "state": RunState.SUBMITTING.value,
                "created": time.time(),
                "end_posture": end["posture"],
                "end_torso": end.get("torso"),
                "end_hands": end.get("hands"),
            }
        )

    def accepted(self, run_id: str, segment: Segment) -> dict:
        return self._set(run_id, RunState.QUEUED, start=segment.start, end=segment.end)

    def refused(self, run_id: str, detail: str) -> dict:
        return self._set(run_id, RunState.REFUSED, detail)

    def unknown(self, run_id: str, detail: str) -> dict:
        return self._set(run_id, RunState.UNKNOWN, detail)

    def update(self, robot: RobotState) -> None:
        """Settle and advance runs from one robot state of the current epoch."""
        for run in self.active():
            if run["epoch"] != robot.epoch:
                self._set(run["run_id"], RunState.LOST, "controller restarted")
            elif robot.phase == Phase.FAULT:
                self._set(run["run_id"], RunState.LOST, "controller fault")
            elif state_of(run) == RunState.UNKNOWN:
                segment = robot.segment(segment_id(run))
                if segment is not None:
                    self.accepted(run["run_id"], segment)
                elif robot.pending != segment_id(run) and not _resendable(run):
                    self._set(run["run_id"], RunState.LOST, "the append did not reach the robot in time")
            elif state_of(run) in (RunState.QUEUED, RunState.RUNNING):
                self._follow(run, robot)

    def to_resend(self, robot: RobotState) -> list[dict]:
        """Unknown runs the robot neither has nor is still taking: send their append again."""
        if not robot.tracking:
            return []
        return [
            run
            for run in self.active()
            if state_of(run) == RunState.UNKNOWN
            and run["epoch"] == robot.epoch
            and _resendable(run, margin=APPEND_TIMEOUT_S)  # it settles before the window closes
            and robot.segment(segment_id(run)) is None
            and robot.pending != segment_id(run)
        ]

    def _follow(self, run: dict, robot: RobotState) -> None:
        if not robot.tracking:
            return
        cursor, start, end = robot.cursor, run["start"], run["end"]
        new = RunState.QUEUED if cursor < start else RunState.RUNNING if cursor < end else RunState.DONE
        changed = {"progress": round(float(np.clip((cursor - start) / max(1, end - start), 0, 1)), 3)}
        if cursor >= start and robot.q is not None:
            changed.update(self._effector_error(run, robot))
        if new != state_of(run):
            detail = changed.pop("detail", run.get("detail"))
            self._set(run["run_id"], new, detail, finished=time.time() if new == RunState.DONE else None, **changed)
        else:  # progress only: memory is enough, the next transition writes it
            self.runs[run["run_id"]] = {**run, **changed}

    def _effector_error(self, run: dict, robot: RobotState) -> dict:
        """Largest distance so far of each hand and foot from its reference, in the pelvis frame (sampled)."""
        self._kin = self._kin or Kinematics()
        measured, reference = self._kin.effectors(robot.q), self._kin.effectors(robot.cursor_pose[7:])
        worst = dict(run.get("max_effector_error_m", {}))
        for name, point in measured.items():
            worst[name] = round(max(worst.get(name, 0.0), float(np.linalg.norm(point - reference[name]))), 3)
        return {"max_effector_error_m": worst}

    def end_state(self, robot: RobotState) -> tuple[Posture, tuple[int, int], dict | None] | None:
        """Posture, torso lean and the hands' intents (in W) at the reference tail, if one of our runs is what ends
        there."""
        ours = [r for r in self.runs.values() if r["epoch"] == robot.epoch and state_of(r) in APPENDED]
        last = max(ours, key=lambda r: r["end"], default=None)
        if last is None or last["end"] != robot.tail:
            return None
        return (
            Posture(last["end_posture"]),
            tuple(last.get("end_torso") or (0, 0)),
            last.get("end_hands"),
        )
