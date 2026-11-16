"""Checked lower-body planning with one bounded collect-and-retry fallback.

Planning only: nothing is executed here and the hands are not planned. Every
target stays in the caller's reference frame, including after collection moves
the root.
"""

from dataclasses import dataclass, field, replace
from itertools import pairwise

import numpy as np
from scipy.spatial.transform import Rotation

from wbc.checks import CheckReport, Limits, run_checks
from wbc.core.errors import ErrorCode, WbcError, invalid
from wbc.core.frames import wrap
from wbc.core.layout import SIDES
from wbc.core.params import parse, to_json
from wbc.core.types import MotionDraft, StartState
from wbc.helpers.gait import WALK_HEIGHT, nominal_height
from wbc.helpers.motion import concatenate

from .footsteps import WalkGoal
from .generate import generate as walk
from .recovery import Recovery, choose, failure_reason
from .stance import Stance
from .validation import check, profile


@dataclass
class Attempt:
    stage: str
    kind: str
    parameters: dict
    start: StartState
    motion: MotionDraft | None = None
    report: CheckReport | None = None
    error: dict | None = None

    @property
    def passed(self):
        return self.report is not None and self.report.passed

    def to_dict(self):
        # Agent-facing diagnostics include failures, not pages of passing
        # checks. In-process debugging still has the complete ``report``.
        checks = (
            None
            if self.report is None
            else dict(
                passed=self.report.passed,
                failures=self.report.failures,
                checks=self.report.findings(blocking=True),
            )
        )
        return dict(
            stage=self.stage,
            kind=self.kind,
            parameters=self.parameters,
            passed=self.passed,
            error=self.error,
            checks=checks,
        )


@dataclass
class WalkerPlan:
    motion: MotionDraft
    report: CheckReport
    stages: list[dict]
    attempts: list[Attempt] = field(repr=False)
    recovery: Recovery = Recovery((), "not_needed")

    @property
    def recovered(self):
        return any(s["stage"] == "collect" for s in self.stages)

    def feedback(self):
        return dict(
            recovered=self.recovered,
            recovery=self.recovery.to_dict(),
            stages=[s["stage"] for s in self.stages],
            end_height_m=float(self.motion.frames[-1, 2]),
            height_range_m=[float(self.motion.frames[:, 2].min()), float(self.motion.frames[:, 2].max())],
            accuracy=[
                dict(
                    stage=a.stage,
                    checks=[
                        r.to_dict()
                        for r in a.report.results
                        if r.name
                        in {
                            "ik_accuracy",
                            "root_target",
                            "root_heading",
                            "landing_position",
                            "landing_heading",
                        }
                    ],
                )
                for a in self.attempts
                if a.passed and a.stage in {s["stage"] for s in self.stages}
            ],
        )


def _endpoint(attempt):
    draft = attempt.motion
    return StartState(draft.frames[-1], draft.end_posture)


def _attempt(start, params, stage, attempts):
    parameters = to_json(params)
    attempt = Attempt(stage, "walk", parameters, start)
    attempts.append(attempt)
    try:
        attempt.motion = walk(start, params)
        attempt.report = check(start, attempt.motion, parameters)
    except WbcError as exc:
        # Expected geometric failures can trigger recovery. Malformed requests
        # and unexpected internal errors are not disguised as a bad stance.
        if exc.details.get("reason") not in {
            "STANCE_REQUIRED",
            "TARGET_NOT_REACHED",
            "REQUEST_LIMIT",
        }:
            raise
        attempt.error = exc.to_dict()
    return attempt


def _reject(reason, stance, attempts, recovery=Recovery((), "not_attempted"), *, stage=None):
    messages = {
        "STANCE_REQUIRED": "Reposition the feet before changing height or continuing this request.",
        "REQUEST_LIMIT": "The request exceeds the path, duration or collection budget; split it into shorter motions.",
        "TARGET_NOT_REACHED": "The requested path/pose did not pass reach checks; try a shorter step or a less extreme height.",
        "SPEED_LIMIT": "The generated motion exceeds a speed limit; reduce the step or request a longer swing/height duration.",
        "COLLISION": "Robot parts intersect; change the path or body pose.",
        "UPPER_BODY_LIMIT": "An upper-body check failed; collecting feet cannot certify hand feasibility.",
        "INVALID_TRAJECTORY": "The generated reference is discontinuous or invalid; inspect the planner.",
        "WALK_HEIGHT_REQUIRED": f"Walking/collection requires height {WALK_HEIGHT[0]}..{WALK_HEIGHT[1]} m; "
        "change height first.",
        "FLAT_SUPPORT_REQUIRED": "Start from a stationary reference with both feet flat on the ground and uncrossed.",
    }
    raise WbcError(
        ErrorCode.PLAN_REJECTED,
        messages[reason],
        reason=reason,
        stage=stage or (attempts[-1].stage if attempts else "precondition"),
        recovery=recovery.to_dict(),
        stance=stance.summary(),
        attempts=[a.to_dict() for a in attempts],
    )


def _finish(selected, attempts, recovery=Recovery((), "not_needed")):
    # Reject a broken boundary rather than patching/interpolating it.
    for before, after in pairwise(selected):
        if not np.array_equal(before.motion.frames[-1], after.motion.frames[0]):
            raise RuntimeError("walker composition changed a reference boundary")
    motion = concatenate([a.motion for a in selected])
    starts = np.cumsum([0] + [len(a.motion) - 1 for a in selected[:-1]]).tolist()
    limits = Limits(tuple(starts), tuple(profile(a.start) for a in selected))
    report = selected[0].report if len(selected) == 1 else run_checks(motion, limits)
    if not report.passed:
        boundary = Attempt("composed", "walk", {}, selected[0].start, motion, report)
        attempts.append(boundary)
        _reject(failure_reason(boundary), Stance.read(selected[0].start), attempts, recovery)
    stages = [
        dict(
            stage=a.stage,
            kind=a.kind,
            parameters=a.parameters,
            start=i,
            stop=i + len(a.motion),
        )
        for i, a in zip(starts, selected, strict=True)
    ]
    motion = replace(motion, limit_stages=tuple(zip(starts, limits.profiles, strict=True)))
    return WalkerPlan(motion, report, stages, attempts, recovery)


def plan(start: StartState, *, recover: bool | None = None, **walk_parameters) -> WalkerPlan:
    """Return an entirely checked walk (``walk_parameters``: a ``WalkGoal``), or reject without executing anything.

    Recovery defaults on for root destinations and off for explicit landings. An
    explicit bool overrides this choice. Parsed goals are immutable: collection
    never shrinks a step, changes a destination or retunes timing.
    """
    if recover is not None and not isinstance(recover, bool):
        raise invalid("recover must be a boolean or null")
    params = parse(WalkGoal, walk_parameters)
    enabled = recover if recover is not None else not params.footsteps
    attempts = []
    stance = Stance.read(start)
    if not stance.flat or stance.width <= 0:
        _reject("FLAT_SUPPORT_REQUIRED", stance, attempts)
    if not WALK_HEIGHT[0] - 1e-6 <= nominal_height(start.height) <= WALK_HEIGHT[1] + 1e-6:
        _reject("WALK_HEIGHT_REQUIRED", stance, attempts)
    if params.footsteps:
        # Bind omitted orientations to the original start, not the recovered
        # pose. Foot targets stay in the caller's original reference frame.
        yaw = stance.frame.yaw
        relative = Rotation.from_matrix(stance.rotations).as_euler("xyz")[:, 2]
        steps = []
        for step in params.footsteps:
            side = SIDES.index(step.side)
            yaw = yaw if step.root_yaw is None else step.root_yaw
            foot_yaw = float(yaw + relative[side]) if step.yaw is None else step.yaw
            steps.append(replace(step, root_yaw=float(yaw), yaw=foot_yaw))
            relative[side] = float(wrap(foot_yaw - yaw))
        params = replace(params, footsteps=tuple(steps))
    direct = _attempt(start, params, "direct", attempts)
    if direct.passed:
        return _finish([direct], attempts)
    reason = failure_reason(direct)
    recovery = choose(enabled=enabled, reason=reason, stance=stance)
    if not recovery.steps:
        _reject(reason, stance, attempts, recovery)

    selected, state = [], start
    collection_stance = Stance.read(state)
    collected = None
    for reverse in (False, True):
        try:
            collection = parse(WalkGoal, collection_stance.collection(state.height, reverse=reverse))
        except WbcError as exc:
            if exc.details.get("reason") != "REQUEST_LIMIT":
                raise
            attempts.append(Attempt("collect", "walk", {}, state, error=exc.to_dict()))
            _reject("REQUEST_LIMIT", stance, attempts, recovery)
        candidate = _attempt(state, collection, "collect", attempts)
        if candidate.passed:
            collected = candidate
            break
        # Both foot orders are predefined candidates. Obstacles may block one
        # but not the other. No recursive recovery or obstacle routing follows.
        if failure_reason(candidate) not in {
            "TARGET_NOT_REACHED",
            "SPEED_LIMIT",
            "COLLISION",
        }:
            break
    if collected is None:
        _reject(failure_reason(attempts[-1]), stance, attempts, recovery)
    selected.append(collected)
    state = _endpoint(collected)
    retry = params
    if params.target is not None:
        retry = replace(
            params,
            turn=float(wrap(stance.frame.yaw + params.turn - Stance.read(state).frame.yaw)),
        )
    retried = _attempt(state, retry, "retry", attempts)
    if not retried.passed:
        _reject(failure_reason(retried), stance, attempts, recovery)
    return _finish([*selected, retried], attempts, recovery)
