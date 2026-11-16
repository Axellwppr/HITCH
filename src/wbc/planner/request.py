"""The agent's whole-body request (docs/REFERENCE.md): an in-place part (pelvis height, torso lean and turn, and both
hands, all starting together on planted feet), then a walk with the arms at their joint angles, then a stand-still.

The in-place body comes from ``helpers.body``, each hand's path from ``planner.hands``, the arms over the body from
``planner.arms``; the walk and the stand-still are the ``walk`` and ``hold`` skills (``planner.steps``), whose checks
and padding the whole plan shares. Fingers (``grip``) close or open after the in-place part, outside the planned
motion (the session's ``grip``); a request with a grip has no walk. Everything is in the plan frame O; ``frame`` maps
it to the reference world W.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from wbc.core import limits as L
from wbc.core.errors import ErrorCode, WbcError, invalid
from wbc.core.frames import SE2
from wbc.core.layout import DT
from wbc.core.params import to_json
from wbc.core.request import Request
from wbc.core.types import MotionDraft, StartState, phases_from_labels
from wbc.helpers.body import body_draft, body_seconds
from wbc.helpers.gait import CONSERVATIVE, SEARCH_HEIGHT, WALK_HEIGHT
from wbc.helpers.motion import concatenate
from wbc.skills.walking import Stance

from . import arms, hands
from .steps import Planned, end_state, finish, hand_notes, run_skill

STANCE_FREE_M = 0.10  # a height change up to this needs no collected stance (the span of the height search)
HEIGHT_SAME_M = 0.005  # a requested height this close to the current one leaves the pelvis where it is
SWING_S = CONSERVATIVE[1]  # the walk's slowest default swing; the slower retry swings L.SLOWER times it


def plan(request: Request, *, start: StartState, frame: SE2, horizon: int = 9, slower: bool = False) -> Planned:
    """Plan ``request`` from ``start``; ``slower`` stretches the default body timing and the walk's swing by
    ``limits.SLOWER``."""
    if not request.moves:
        raise invalid("nothing to plan: the request only moves fingers")
    drafts, notes, state, intents = [], [], start, start.hands
    if request.in_place:
        draft, in_place_notes, intents = _in_place(request, start, slower)
        drafts.append(draft)
        notes += in_place_notes
        state = end_state(draft, start)
    if request.walk is not None:
        _check_walk(state)
        offset = sum(len(d.frames) - 1 for d in drafts)
        parameters = {k: v for k, v in to_json(request.walk).items() if v not in (None, [])}
        if slower:
            parameters["swing_s"] = round(SWING_S * L.SLOWER, 3)
        draft = _step("walk", parameters, state, offset)
        if "walker" in draft.diagnostics:
            notes.append({"kind": "walk", "start_s": offset * DT, **draft.diagnostics["walker"]})
        drafts.append(draft)
        state = end_state(draft, state)
    if request.wait_s is not None:
        offset = sum(len(d.frames) - 1 for d in drafts)
        drafts.append(_step("hold", {"duration_s": request.wait_s}, state, offset))
    if request.walk is not None:
        intents = None  # the arms locked as they walked: each hand's reference pose is its intent again
    motion = concatenate(drafts)
    if len(motion.frames) + horizon > L.SEGMENT_FRAMES:  # with the stationary tail finish() adds
        raise invalid(
            f"the motion lasts {len(motion.frames) * DT:.1f} s; at most {(L.SEGMENT_FRAMES - horizon) * DT:.0f} s "
            "go to the controller at once: split it into shorter requests",
            reason="REQUEST_LIMIT",
            limit="duration_s",
            requested=round(len(motion.frames) * DT, 2),
            maximum=round((L.SEGMENT_FRAMES - horizon) * DT, 2),
        )
    # both feet stay where they are through the in-place part (and the stand-still after it, without a walk)
    fixed = start.pose if request.in_place or request.walk is None else None
    until = len(drafts[0].frames) - 1 if request.in_place and request.walk is not None else None
    planned = finish(
        motion,
        start=start,
        frame=frame,
        horizon=horizon,
        end_torso=state.torso,
        notes=notes,
        fixed_support=fixed,
        fixed_until=until,
    )
    return replace(planned, hands=intents or None)


# ------------------------------------------------------------------- in place
def _in_place(request: Request, start: StartState, slower: bool) -> tuple[MotionDraft, list[dict], dict]:
    """The body change and both hands from the same start, on planted feet; the notes the result reports; each
    hand's intent at the end (hands.end_intents)."""
    height = start.height if request.height is None else float(request.height)
    torso = start.torso if request.torso is None else (request.torso.pitch_deg, request.torso.yaw_deg)
    seconds, body_end = 0.0, None
    if abs(height - start.height) <= HEIGHT_SAME_M:
        height = start.height  # the height it has: no body change for a rounding difference
    if abs(height - start.height) > 1e-9 or tuple(torso) != tuple(start.torso):
        _check_body(start, height)
        seconds = request.body_duration_s or body_seconds(start.height, height, start.torso, torso)
        if slower:
            seconds = min(8.0, seconds * L.SLOWER)
        body = body_draft(start, height, torso, seconds)
        body_end = len(body.frames) - 1
    else:
        body = _still(start)
    if request.auto:
        low, high = SEARCH_HEIGHT
        if tuple(start.torso) != (0, 0) or not arms.candidate_heights(start.height):
            raise WbcError(
                ErrorCode.PRECONDITION,
                f'height "auto" needs the robot standing, the torso in its standard posture and the pelvis '
                f"{low:.2f}–{high:.2f} m high; it is {start.posture.value}, the torso leaned {start.torso[0]}° and "
                f"turned {start.torso[1]}°, the pelvis {start.height:.2f} m",
                reason="AUTO_HEIGHT",
            )
    plan = hands.compile_hands(body, request.hand_commands, body_end, start.hands)
    try:
        motion = arms.solve(plan, auto_height=request.auto, body_end=body_end)
    except WbcError as error:
        hands.explain(error, plan, request.hand_commands)
        raise
    timing = motion.diagnostics.get("hand_timing")
    if body_end is not None and timing is not None:  # the body change as played, on the shared clock
        seconds = float(np.searchsorted(np.asarray(timing["source_frames"]), body_end - 1e-8)) * DT
    notes = [
        {
            "kind": "in_place",
            "duration_s": motion.duration,
            "body_s": seconds,
            "height_m": [start.height, float(motion.frames[-1, 2]) if request.auto else height],
            "torso_deg": [list(start.torso), list(torso)],
        },
        *hand_notes(motion),
    ]
    intents = hands.end_intents(plan, motion, start.hands)
    return replace(motion, end_torso=tuple(torso)), notes, intents


def _check_body(start: StartState, height: float) -> None:
    """Refuse a height change the start's stance cannot make (docs/REFERENCE.md 3.6, 4.2)."""
    if abs(height - start.height) > STANCE_FREE_M:
        stance = Stance.read(start)
        if not stance.flat or not stance.aligned:
            raise WbcError(
                ErrorCode.PRECONDITION,
                f"a height change over {100 * STANCE_FREE_M:.0f} cm needs flat, parallel, side-by-side feet; collect "
                "them first with a short walk (target [0, 0])",
                reason="STANCE_REQUIRED",
                stance=stance.summary(),
            )


def _still(start: StartState) -> MotionDraft:
    """A body that stays where it is (the hands set the length)."""
    return MotionDraft(
        np.repeat(start.pose[None], 2, axis=0),
        phases_from_labels(["start", "hold"]),
        start.posture,
        support=np.ones((2, 2), bool),
        limit_stages=((0, L.DEFAULT),),
        end_torso=tuple(start.torso),
    )


# ----------------------------------------------------------- walk, stand still
def _check_walk(state: StartState) -> None:
    """The walk starts where the in-place part ends: torso standard, pelvis in the walking range (docs 3.4)."""
    if tuple(state.torso) != (0, 0):
        raise WbcError(
            ErrorCode.PRECONDITION,
            f"the walk needs the torso in its standard posture; it would start leaned {state.torso[0]}° and turned "
            f"{state.torso[1]}°: add torso {{pitch_deg: 0, yaw_deg: 0}} to this request (it runs before the walk)",
            reason="WALK_TORSO",
        )
    low, high = WALK_HEIGHT
    if not low - 1e-3 <= state.height <= high + 1e-3:
        raise WbcError(
            ErrorCode.PRECONDITION,
            f"the walk starts with the pelvis {low:.2f}–{high:.2f} m high; it would start {state.height:.2f} m high: "
            "add a height in that range to this request (it runs before the walk)",
            reason="WALK_HEIGHT",
        )


def _step(skill: str, parameters: dict, state: StartState, offset: int) -> MotionDraft:
    """One skill from ``state`` with the arms at their joint angles; errors say where in the request it failed."""
    try:
        return run_skill(skill, parameters, state, skill)
    except WbcError as error:
        error.details.update(part=skill, plan_frame_offset=offset, frame_scope="skill_or_recovery_attempt")
        raise
