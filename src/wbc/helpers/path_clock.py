"""Smooth time allocation for an already solved, fixed-support body/arm path.

The positive clock density w(s)=dt/ds bounds every joint and root speed by
w >= |dq/ds| / vmax. Cubic B-splines make w C2. Each waypoint interval fixes
w, w' and w'' at its boundaries, so independent small solves join smoothly.
First minimise duration (at least the requested budget), then minimise the
whole interval's density curvature at that duration. No IK is run here.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, CubicSpline
from scipy.optimize import LinearConstraint, brentq, linprog, minimize
from scipy.spatial.transform import Rotation, RotationSpline

from wbc.core import limits as L
from wbc.core.errors import ErrorCode, WbcError
from wbc.core.layout import DT

SPEED_MARGIN = 0.01  # the share of each speed limit the clock keeps free


@dataclass
class TimedPath:
    frames: np.ndarray
    source_frames: np.ndarray
    segments: list[dict]

    def frame_at(self, source):
        return int(round(np.interp(source, self.source_frames, np.arange(len(self.frames)))))


def _interval(source, demand, boundary, requested, index):
    """Small global spline solve; boundary triples impose C2 joins."""
    # Resolve rate peaks near waypoint boundaries while smoothing the entire
    # interval. This is numerical resolution, independent of robot/task direction.
    knots = np.r_[[0.0] * 3, np.linspace(0, requested, 25), [requested] * 3]
    basis = BSpline(knots, np.eye(len(knots) - 4), 3)
    free = np.arange(3, len(knots) - 7)
    fixed = np.zeros(len(knots) - 4)
    fixed[:3], fixed[-3:] = boundary
    sampled = basis(source)
    matrix, lower = sampled[:, free], demand * (1 + 1e-6) - sampled @ fixed
    primitive = basis.antiderivative()
    integral = primitive(requested) - primitive(0)
    area, constant = integral[free], float(integral @ fixed)
    optimum = linprog(
        area,
        A_ub=np.vstack([-matrix, -area]),
        b_ub=np.r_[-lower, constant - requested],
        bounds=(1e-6, None),
        method="highs",
    )
    if not optimum.success:
        raise WbcError(
            ErrorCode.PLAN_REJECTED,
            "Could not construct a smooth path clock.",
            reason="HAND_TIMING_SOLVE_FAILED",
            segment=index,
            detail=optimum.message,
        )
    minimum = constant + float(area @ optimum.x)
    count = max(round(requested / DT), int(np.ceil((minimum - 1e-8) / DT)))
    duration = count * DT
    if count > 2 * round(requested / DT):
        raise WbcError(
            ErrorCode.PLAN_REJECTED,
            "This path needs more than twice the waypoint interval's time budget. Give it more time or change the path.",
            reason="HAND_PATH_RATE_LIMIT",
            stage="hand_timing",
            segment=index,
            requested_duration_s=requested,
            maximum_duration_s=2 * requested,
            required_duration_s=duration,
            minimum_smooth_duration_s=minimum,
            evidence="minimum_for_the_solved_path_and_C2_clock_basis",
        )
    # Rounding upward to 50 Hz leaves a feasible initial point: all basis
    # functions are nonnegative, so adding time cannot violate a speed bound.
    initial = optimum.x + max(0.0, duration - minimum) / area.sum()
    bends = basis(source, 2)
    h = bends[:, free].T @ bends[:, free]
    g = bends[:, free].T @ (bends @ fixed)
    scale = max(1.0, float(np.diag(h).max()))
    h, g = h / scale, g / scale
    smooth = minimize(
        lambda c: 0.5 * c @ h @ c + g @ c,
        initial,
        jac=lambda c: h @ c + g,
        method="SLSQP",
        bounds=[(1e-6, None)] * len(free),
        constraints=[
            LinearConstraint(matrix, lower, np.inf),
            LinearConstraint(area, duration - constant, duration - constant),
        ],
        options=dict(maxiter=100, ftol=1e-10),
    )
    if not smooth.success or np.min(matrix @ smooth.x - lower) < -1e-7:
        raise WbcError(
            ErrorCode.PLAN_REJECTED,
            "Smooth time allocation did not converge.",
            reason="HAND_TIMING_SOLVE_FAILED",
            segment=index,
            detail=smooth.message,
        )
    coefficients = fixed.copy()
    coefficients[free] = smooth.x
    clock = BSpline(knots, coefficients, 3).antiderivative()
    origin = float(clock(0))
    times = np.arange(count + 1) * DT
    positions = np.r_[
        0.0,
        [brentq(lambda s, target: float(clock(s)) - origin - target, 0.0, requested, args=(t,)) for t in times[1:-1]],
        requested,
    ]
    return positions, minimum


def allocate(frames, cuts, profile=L.DEFAULT):
    """Retain each waypoint budget when possible; stretch each interval at most 2x.

    `cuts` are original waypoint/end frame indices, including any stationary tail.
    A single source coordinate resamples root, waist, legs and both arms together.
    The caller rechecks FK, contacts and all trajectory limits after resampling.
    """
    source = np.arange(len(frames)) * DT
    cuts = np.array(sorted({0, len(frames) - 1, *map(int, cuts)}))
    if cuts[0] < 0 or cuts[-1] >= len(frames):
        raise ValueError("clock boundaries must be inside the path")
    joints = CubicSpline(source, frames[:, 7:], bc_type="clamped")
    root = CubicSpline(source, frames[:, :3], bc_type="clamped")
    turn = RotationSpline(source, Rotation.from_quat(frames[:, 3:7], scalar_first=True))
    # the clock is set against the limits less SPEED_MARGIN: the 50 Hz frames are checked by their differences, which
    # can exceed the spline's speed at the collocation points
    margin = 1.0 - SPEED_MARGIN
    joint_limits = np.r_[np.full(12, profile.leg_speed), np.full(17, profile.upper_speed)] * margin

    def demand(t):
        return np.maximum.reduce(
            (
                np.max(abs(joints(t, 1)) / joint_limits, axis=-1),
                np.linalg.norm(root(t, 1), axis=-1) / (profile.root_speed * margin),
                np.linalg.norm(turn(t, 1), axis=-1) / (profile.root_turn * margin),
            )
        )

    boundaries = np.maximum(1.0, demand(source[cuts]) * (1 + 2e-6))
    samples, segments, offset = [np.array([0.0])], [], 0
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:], strict=True)):
        seconds = float((b - a) * DT)
        # Quarter-tick collocation controls spline speed between output samples;
        # the final 50 Hz motion is independently speed-checked by the planner.
        grid = np.linspace(0, seconds, max(4 * (b - a), 96) + 1)
        rates = demand(source[a] + grid)
        if rates.max() <= 1 and np.all(boundaries[i : i + 2] == 1):
            part, minimum = np.arange(b - a + 1) * DT, seconds
        else:
            part, minimum = _interval(grid, rates, boundaries[i : i + 2], seconds, i)
        count = len(part) - 1
        samples.append(source[a] + part[1:])
        segments.append(
            dict(
                source_start_frame=int(a),
                source_end_frame=int(b),
                start_frame=offset,
                end_frame=offset + count,
                requested_duration_s=seconds,
                duration_s=count * DT,
                minimum_smooth_duration_s=minimum,
            )
        )
        offset += count
    samples = np.concatenate(samples)
    poses = np.column_stack((root(samples), turn(samples).as_quat(scalar_first=True), joints(samples)))
    # Every original waypoint pose is retained exactly, including the initial ref.
    poses[0] = frames[0]
    for segment in segments:
        poses[segment["end_frame"]] = frames[segment["source_end_frame"]]
    return TimedPath(poses, samples / DT, segments)
