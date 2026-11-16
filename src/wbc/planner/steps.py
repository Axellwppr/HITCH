"""The motion skills (``walk``, ``hold``) run from a state of a plan, and the finishing of a whole plan; the request
planner (``wbc.planner.request``) builds on both.

Frames: parameters, the start pose and the result are in the observation frame O; ``frame`` maps O to the reference
world W. A skill runs in the step's own frame S (its start pose's ground projection, heading 0): ``run_skill`` maps
its parameters there, runs it, maps the result back to O and keeps the arms at their joint angles; the result's end
state starts what follows. ``finish`` pads the stationary tail, runs the robot checks and maps O to W. Skills
therefore always start at the origin facing +x, whatever came before them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from wbc.checks import CheckReport, Limits, run_checks
from wbc.core import limits as L
from wbc.core.errors import ErrorCode, WbcError, invalid
from wbc.core.frames import SE2, continuous_quaternions
from wbc.core.layout import ARMS, DT
from wbc.core.params import parse, reframe
from wbc.core.types import MotionDraft, StartState
from wbc.skills.hold import HoldParams, hold
from wbc.skills.walk import WalkParams, restate_attempts, walk

SKILLS = {"walk": (WalkParams, walk), "hold": (HoldParams, hold)}  # name -> (parameters, skill)


@dataclass
class Planned:
    motion: MotionDraft  # plan frame
    world: np.ndarray  # same frames in W
    report: CheckReport
    start: StartState
    limits: Limits
    notes: list[dict]  # what the plan result reports as "body" (the in-place part, the walk, the hands)
    hands: dict | None = None  # each hand's intent at the end (StartState.hands), in the plan frame


def pad_tail(motion: MotionDraft, horizon: int) -> MotionDraft:
    """Make sure the last ``horizon + 1`` frames are identical (the controller looks ahead)."""
    frames = motion.frames
    still = 1
    while still < len(frames) and np.allclose(frames[-1 - still], frames[-1], atol=1e-9):
        still += 1
    return motion.held(horizon + 1 - still)


def end_state(draft: MotionDraft, before: StartState) -> StartState:
    """Where the next step starts: the draft's last pose and what it ends in (the torso as it was, if the draft does
    not set it)."""
    torso = before.torso if draft.end_torso is None else draft.end_torso
    return StartState(draft.frames[-1], draft.end_posture, torso)


def lock_arms(draft: MotionDraft, start: StartState) -> MotionDraft:
    """The draft with both arms at their starting joint angles throughout (they move with the torso)."""
    frames = draft.frames.copy()
    frames[:, ARMS] = start.pose[ARMS]
    return replace(draft, frames=frames)


def finish(
    motion: MotionDraft,
    *,
    start: StartState,
    frame: SE2,
    horizon: int,
    end_torso: tuple[int, int],
    notes: list[dict],
    fixed_support: np.ndarray | None = None,
    fixed_until: int | None = None,
) -> Planned:
    """Pad the stationary tail, run the robot checks and map the motion to W; ``notes`` is what the result reports
    (``Planned.notes``). ``fixed_support`` (a pose) pins both feet to its placement up to frame ``fixed_until``
    (default: throughout)."""
    motion = pad_tail(motion, horizon)
    motion = replace(motion, frames=continuous_quaternions(motion.frames), end_torso=tuple(end_torso))
    limits = Limits(tuple(i for i, _ in motion.limit_stages), tuple(p for _, p in motion.limit_stages))
    report = run_checks(motion, limits, fixed_support=fixed_support, fixed_until=fixed_until)
    return Planned(motion, frame.apply_poses(motion.frames), report, start, limits, notes)


def hand_notes(motion: MotionDraft) -> list[dict]:
    """What a hand plan changed that the result reports: stretched waypoint intervals and the body heights a height
    search chose (times from the plan's start)."""
    notes = []
    timing = motion.diagnostics.get("hand_timing")
    if timing is not None:
        notes.append(
            {
                "kind": "hand_timing",
                "segments": [
                    {
                        "requested_duration_s": s["requested_duration_s"],
                        "duration_s": s["duration_s"],
                        "end_time_s": s["end_frame"] * DT,
                    }
                    for s in timing["segments"]
                ],
            }
        )
    adjustment = motion.diagnostics.get("height_adjustment")
    if adjustment is not None and any(
        abs(s["selected_height_m"] - adjustment["anchor_height_m"]) > 1e-7 for s in adjustment["segments"]
    ):
        notes.append(
            {
                "kind": "hand_height",
                "anchor_height_m": adjustment["anchor_height_m"],
                "waypoints": [
                    {"time_s": segment["end_frame"] * DT, "height_m": segment["selected_height_m"]}
                    for segment in adjustment["segments"]
                    if "selected_height_m" in segment and segment["phase"] == "waypoint"
                ],
            }
        )
    return notes


def run_skill(name: str, parameters: dict, state: StartState, path: str) -> MotionDraft:
    """Run skill ``name`` with ``parameters`` (JSON; errors name ``path``) from ``state`` in its own frame S, and return
    its draft in the plan frame, the arms at their joint angles, retaining internal stage limits."""
    if name not in SKILLS:
        raise invalid(f"unknown skill {name!r}; available: {sorted(SKILLS)}")
    schema, skill = SKILLS[name]
    params = parse(schema, parameters, path)
    to_plan = SE2.of_pose(state.pose)
    to_step = to_plan.inverse()
    local = StartState(to_step.apply_poses(state.pose), state.posture, state.torso)
    params = reframe(params, to_step)
    try:
        draft = skill(local, params)
    except WbcError as error:
        restate_attempts(error.details.get("attempts", ()), to_plan)  # the walker's, in S
        raise
    gap = float(np.max(np.abs(draft.frames[0] - local.pose)))
    if gap > L.START_TOLERANCE:
        raise WbcError(ErrorCode.INTERNAL, f"skill {name} did not start at the start pose (gap {gap:.2e})")
    frames = to_plan.apply_poses(draft.frames)
    frames[0] = state.pose  # exact, so consecutive steps join without round-off
    stages = draft.limit_stages or ((0, L.DEFAULT),)
    return lock_arms(replace(draft, frames=frames, limit_stages=stages), state)
