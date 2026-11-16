"""Solve both arms over the in-place body (``planner.hands.ArmPlan``), with one smooth clock for body and arms and,
for ``height: "auto"``, a bounded search over fixed-foot pelvis heights.

Body IK pins both feet; arm IK reads the resulting body at every frame (``helpers.arm_path``). Candidate heights
always refer to the request's initial height. Each waypoint interval gets at most five independent trials. This
module never collects feet or approaches a hand target.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from multiprocessing import get_context

import numpy as np

from wbc.checks import CheckReport, Limits, check_geometry, check_ik, check_root, check_speed
from wbc.core import limits as L
from wbc.core.errors import ErrorCode, WbcError
from wbc.core.layout import ARMS, DT, SIDES
from wbc.core.types import MotionDraft, StartState, phases_from_labels, posture_for_height
from wbc.helpers import arm_path
from wbc.helpers.gait import SEARCH_HEIGHT
from wbc.helpers.motion import concatenate
from wbc.helpers.path_clock import allocate
from wbc.helpers.posture import fixed_feet

from .hands import ArmPlan

OFFSETS = (0.0, 0.05, 0.10, -0.05, -0.10)
# A height candidate's failed whole-robot check, as the reason a plan's failed check gives (docs/REFERENCE.md 4.3);
# its other checks (stance_slip) fail as TRAJECTORY_CHECK_FAILED
CHECK_REASONS = {
    name: L.CHECK_REASONS[name]
    for name in ("self_collision", "ground_penetration", "fixed_feet_position", "fixed_feet_rotation", "ik_residual")
}
# Palm pose error (position m, rotation rad) good enough to keep the body where it is: millimetres gained by crouching
# are not worth the posture.
GOOD_ENOUGH = (0.005, np.deg2rad(10.0))


def candidate_heights(height: float) -> tuple[float, ...]:
    """Clip to the search range, deduplicate boundaries, prefer smaller shifts."""
    low, high = SEARCH_HEIGHT
    if not low - 1e-6 <= height <= high + 1e-6:
        return ()
    height = float(np.clip(height, low, high))
    values = {float(np.clip(height + delta, low, high)) for delta in OFFSETS}
    return tuple(sorted(values, key=lambda h: (abs(h - height), h)))


def _eligible(base: MotionDraft) -> bool:
    if not candidate_heights(float(base.frames[0, 2])):
        return False
    return base.support is None or bool(base.support.all())


@dataclass(frozen=True)
class _Window:
    path: object
    offset: int

    @property
    def kind(self):
        return self.path.kind

    @property
    def moves(self):
        return getattr(self.path, "moves", True)

    def tolerance(self, k):
        return self.path.tolerance(k + self.offset)

    @property
    def pieces(self):
        return [piece._replace(first=piece.first - self.offset) for piece in self.path.pieces]

    def at(self, k, pose):
        return self.path.at(k + self.offset, pose)


@dataclass
class _Trial:
    height: float
    motion: MotionDraft | None
    error: dict | None

    def feedback(self):
        return {
            "height_m": self.height,
            "passed": self.motion is not None,
            **({"quality": self.motion.diagnostics.get("hand_quality")} if self.motion else {}),
            **({"error": self.error} if self.error else {}),
        }


def _body(start: StartState, height: float, count: int) -> MotionDraft:
    if abs(height - start.height) >= 1e-7:
        return fixed_feet(start, height_m=height, duration_s=count * DT, label="reach_height")
    frames = np.repeat(start.pose[None], count + 1, axis=0)
    return MotionDraft(
        frames,
        phases_from_labels(["reach_height"] * len(frames)),
        posture_for_height(height),
        support=np.ones((len(frames), 2), bool),
        limit_stages=((0, L.DEFAULT),),
    )


def _trial(start, height, begin, stop, paths, posture_weight, anchor) -> _Trial:
    """One complete interval, including body feasibility, arms and whole-robot checks."""
    try:
        body = _body(start, height, stop - begin)
        windows = {side: _Window(path, begin) for side, path in paths.items()}
        result = arm_path.solve(body, windows, posture_weight=posture_weight, suggestions=False)
        report = _geometry_checks(result, anchor)
        if not report.passed:
            findings = report.findings(True)
            for finding in findings:
                if "frame" in finding:
                    finding["frame"] += begin
            names = ", ".join(dict.fromkeys(f["name"] for f in findings))
            return _Trial(
                height,
                None,
                {
                    "reason": CHECK_REASONS.get(findings[0]["name"], "TRAJECTORY_CHECK_FAILED"),
                    "message": f"the whole-robot checks failed: {names}",
                    "frame_scope": "plan",
                    "findings": findings,
                },
            )
        return _Trial(height, result, None)
    except WbcError as error:
        detail = error.to_dict()
        if "frame" in detail:
            detail.update(frame=int(detail["frame"]) + begin, frame_scope="plan")
            detail["time_s"] = detail["frame"] * DT
        return _Trial(height, None, detail)


def solve(plan: ArmPlan, *, auto_height: bool, body_end: int | None = None) -> MotionDraft:
    """Both arms over ``plan.body`` (a ``planner.hands.ArmPlan``), retimed onto one smooth clock with the body;
    ``auto_height`` lets a still body change its pelvis height per waypoint interval; ``body_end`` (the frame a
    body change ends) splits the clock like a waypoint end."""
    cuts = () if body_end is None else (body_end,)
    if all(path.kind == "joints" for path in plan.paths.values()) and _unchanged(plan):
        return _time(plan.body, plan.paths, cuts)  # the body change gets the same clock as with moving arms
    return _time(_solve(plan, adjust_height=auto_height), plan.paths, cuts)


def _unchanged(plan: ArmPlan) -> bool:
    """Whether joint paths only hold the arms the body already carries."""
    targets = np.array([[plan.paths[s].at(k, pose) for s in SIDES] for k, pose in enumerate(plan.body.frames)])
    return bool(np.allclose(targets.reshape(len(plan.body), -1), plan.body.frames[:, ARMS], atol=1e-6))


def _geometry_checks(motion, anchor):
    """One candidate FK pass for interactions of independently solved limbs.

    Joint bounds and palm tolerances are established by IK. Timing is allocated
    on the selected path. The assembled public plan retains its general checks,
    which also cover joins and controller padding.
    """
    geometry, collisions = check_geometry(motion, Limits((0,), (L.DEFAULT,)), fixed_support=anchor)
    return CheckReport([check_ik(motion.diagnostics), *geometry], collisions)


def _rate_limit(error, cuts, length, paths, body_end=None):
    """The clock's interval failure, restated in the request's own waypoints and times.

    The clock splits the path at every waypoint end of either hand, so its
    interval index means nothing to the caller. Name the waypoints that span
    the interval and give its window in seconds from the start of the request.
    """
    bounds = sorted({0, length - 1, *cuts})
    segment = error.details["segment"]
    start, end = bounds[segment], bounds[segment + 1]
    names = [
        f"hands.{side}.waypoints[{index}]"
        for side, path in sorted(paths.items())
        for index, (first, count, *_) in enumerate(_pieces(path))
        if first <= start and end <= first + count
    ]
    if body_end is not None and end <= body_end:
        names.append("the height and torso change")
    span = ", ".join(names) or "the hold after the last waypoint"
    required, budget = error.details["required_duration_s"], error.details["requested_duration_s"]
    more = max(0.05, required / 2 - budget)
    return WbcError(
        error.code,
        f"The motion from {start * DT:.2f} s to {end * DT:.2f} s ({span}) needs {required:.2f} s, "
        f"more than twice its {budget:.2f} s. Give it at least {more:.2f} s more (the waypoint's duration_s, or "
        "body_duration_s for the height and torso change), or shorten the path.",
        reason="HAND_PATH_RATE_LIMIT",
        stage=error.details.get("stage", "hand_timing"),
        waypoints=names,
        window_s=[round(start * DT, 2), round(end * DT, 2)],
        required_duration_s=round(required, 2),
    )


def _pieces(path) -> list:
    """The intervals a path's clock is cut at: its pieces, unless it only holds one pose (a hand following the body)."""
    return list(getattr(path, "pieces", ())) if getattr(path, "moves", True) else []


def _time(motion, paths, extra_cuts=()):
    limits = Limits((0,), (L.DEFAULT,))
    if all(r.passed for r in check_speed(motion.frames, limits) + check_root(motion.frames, limits)):
        return motion
    cuts = {first + count for p in paths.values() for first, count, *_ in _pieces(p)}
    cuts.update(round(p.seconds / DT) for p in paths.values() if hasattr(p, "seconds"))
    cuts.update(extra_cuts)
    interior = [k for k in cuts if 0 < k < len(motion)]
    try:
        timed = allocate(motion.frames, interior)
    except WbcError as error:
        if error.details.get("reason") == "HAND_PATH_RATE_LIMIT" and "segment" in error.details:
            raise _rate_limit(error, interior, len(motion), paths, max(extra_cuts, default=None)) from None
        raise
    source = timed.source_frames
    if len(source) == len(motion) and np.max(abs(source - np.arange(len(motion)))) < 1e-8:
        return motion
    diagnostics = deepcopy(motion.diagnostics)
    diagnostics["hand_timing"] = dict(segments=timed.segments, source_frames=source.tolist())
    worst = diagnostics.get("ik_worst", {})
    if "frame" in worst:
        worst["frame"] = timed.frame_at(worst["frame"])
    for row in diagnostics.get("hand_reach", {}).values():
        row["frame"] = timed.frame_at(row["frame"])
    for segment in diagnostics.get("height_adjustment", {}).get("segments", ()):
        for key in ("start_frame", "end_frame"):
            segment["source_" + key] = segment[key]
            segment[key] = timed.frame_at(segment[key])
        for candidate in segment["candidates"]:
            if "error" in candidate:
                candidate["error"]["frame_scope"] = "candidate_geometry"
    indices = np.clip(np.floor(source + 1e-8).astype(int), 0, len(motion) - 1)
    result = replace(
        motion,
        frames=timed.frames,
        phases=phases_from_labels(motion.labels()[indices]),
        support=None if motion.support is None else motion.support[indices],
        keyframes=tuple(sorted({timed.frame_at(k) for k in motion.keyframes})),
        limit_stages=tuple((timed.frame_at(k), p) for k, p in motion.limit_stages),
        diagnostics=diagnostics,
    )
    result = arm_path.validate_timing(result, paths, source, motion.frames)
    return result


@lru_cache(maxsize=1)
def _workers() -> ProcessPoolExecutor:
    """One worker per height candidate, started once and reused across requests (Python closes them when the host
    exits). Processes avoid serial Python FK/task work under the GIL; spawn avoids inheriting renderer/ONNX threads
    from a running host.

    Workers never render, so they start without a GL backend (``MUJOCO_GL=disable``), whose GPU driver setup
    serializes across processes. MuJoCo reads the variable when it is imported, and a spawned worker imports it before
    any pool initializer runs: it re-imports the host's main module (``python -m wbc.session.host``, which imports the
    planner) as ``__mp_main__``, and the initializer's own module. So the variable must be in the environment the
    workers are started with: it is set here, while every worker is started, and restored at once. The session host
    plans under its command lock, so nothing else plans meanwhile.
    """
    pool = ProcessPoolExecutor(max_workers=len(OFFSETS), mp_context=get_context("spawn"))
    saved = os.environ.get("MUJOCO_GL")
    os.environ["MUJOCO_GL"] = "disable"
    try:
        for future in [pool.submit(int) for _ in OFFSETS]:  # each submit starts a worker while none is idle
            future.result()
    finally:
        if saved is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = saved
    return pool


def _in_workers(arguments: list[tuple]) -> list:
    """``_trial`` of each argument tuple in the worker pool; a pool that lost a worker (killed, crashed) is replaced
    and the trials run once more."""

    def run() -> list:
        futures = [_workers().submit(_trial, *args) for args in arguments]
        return [future.result() for future in futures]

    try:
        return run()
    except BrokenProcessPool:
        if _workers.cache_info().currsize:  # the cached pool, not a new one
            _workers().shutdown(wait=False, cancel_futures=True)
        _workers.cache_clear()
        return run()


def _choose(trials, height):
    """The smallest body adjustment with good-enough palms (ties: the higher body); if none, Pareto quality in
    1 mm / 2 degree bins, then smallest body adjustment."""
    enough = [t for t in trials if np.all(np.asarray(t.motion.diagnostics["hand_quality"]) <= GOOD_ENOUGH)]
    if enough:
        return min(enough, key=lambda t: (round(abs(t.height - height), 6), -t.height))
    quality = [np.floor(np.asarray(t.motion.diagnostics["hand_quality"]) / [0.001, np.deg2rad(2)]) for t in trials]
    front = [i for i, q in enumerate(quality) if not any(np.all(other <= q) and np.any(other < q) for other in quality)]
    return trials[min(front, key=lambda i: (abs(trials[i].height - height), quality[i][1], quality[i][0]))]


def _solve(command: ArmPlan, *, adjust_height: bool) -> MotionDraft:
    """Compile O targets once; select whole waypoint intervals from the same start."""
    base, paths = command.body, command.paths
    eligible = adjust_height and _eligible(base) and any(p.kind == "pose" and p.moves for p in paths.values())
    if not eligible:
        return arm_path.solve(base, paths, posture_weight=command.posture_weight)
    anchor = float(base.frames[0, 2])
    heights = candidate_heights(anchor)
    cuts = sorted(
        {
            first + count
            for path in paths.values()
            if path.kind == "pose"
            for first, count, *_ in _pieces(path)
            if 0 < first + count < len(base)
        }
    )
    moving_end = max(cuts)
    if moving_end < len(base) - 1:
        cuts.append(len(base) - 1)
    state = StartState(base.frames[0], base.end_posture)
    segments, attempts, begin = [], [], 0
    adjustment = dict(anchor_height_m=anchor, range_m=list(SEARCH_HEIGHT), foot_policy="fixed", segments=attempts)
    for stop in cuts:
        candidates = (state.height,) if begin >= moving_end else heights
        arguments = [(state, h, begin, stop, paths, command.posture_weight, base.frames[0]) for h in candidates]
        if len(arguments) == 1:
            trials = [_trial(*arguments[0])]
        else:
            trials = _in_workers(arguments)
        feasible = [trial for trial in trials if trial.motion is not None]
        entry = dict(
            start_frame=begin,
            end_frame=stop,
            phase="waypoint" if begin < moving_end else "hold",
            candidates=[t.feedback() for t in trials],
        )
        attempts.append(entry)
        if not feasible:
            detail = dict(trials[0].error)
            code, message = (
                detail.pop("code", ErrorCode.PLAN_REJECTED),
                detail.pop("message", "No body height candidate passed."),
            )
            if detail.get("reason") in ("HAND_REACH_LIMIT", "HAND_POSE_UNREACHABLE"):
                frame = int(detail.get("frame", begin))
                body = _body(state, trials[0].height, stop - begin)
                pose = body.frames[min(max(frame - begin, 0), len(body) - 1)]
                detail["suggestion"] = arm_path.approach_hint(pose, paths, frame)
            detail["height_adjustment"] = adjustment
            raise WbcError(code, message, **detail)
        chosen = _choose(feasible, state.height)
        entry["selected_height_m"] = chosen.height
        segments.append(chosen.motion)
        state = StartState(chosen.motion.frames[-1], chosen.motion.end_posture)
        begin = stop
    combined = concatenate(segments)
    combined.diagnostics.update(height_adjustment=adjustment)
    return combined
