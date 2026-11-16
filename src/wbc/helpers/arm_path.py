"""One warm-started arm pass on a prescribed body; diagnose only failed samples."""

from dataclasses import replace
from functools import partial

import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core import limits as L
from wbc.core.errors import ErrorCode, WbcError
from wbc.core.layout import ARMS, DT, JOINTS, SIDES, arm
from wbc.helpers.arm_ik import POSITION_TOL, POSTURE_WEIGHT, ArmIK
from wbc.robot.kinematics import Kinematics, body_rotation


def _feedback(result, pose):
    """The solver's feedback, its displacement in the plan's observation frame O."""
    out = result.feedback()
    out["position_error_O_m"] = (body_rotation(pose) @ out.pop("position_error_body_m")).tolist()
    return out


def _at_limit(solver, joints) -> list[str]:
    """The arm joints of a solution within 0.02 rad of their range: what stopped the arm."""
    low, high = solver.geometry.limits.T
    names = JOINTS[solver.geometry.slice.start - 7 : solver.geometry.slice.stop - 7]
    near = (joints - low < 0.02) | (high - joints < 0.02)
    return [name.removesuffix("_joint") for name, flag in zip(names, near, strict=True) if flag]


def _reach(solver, pose, target):
    """The arm's reach check, its shoulder and target in O."""
    out = solver.geometry.reach(pose, target)
    rotation = body_rotation(pose)
    for name in ("shoulder", "target"):
        out[f"{name}_O_m"] = (pose[:3] + rotation @ out.pop(f"{name}_body_m")).tolist()
    return out


def _translation_hint(pose, goals, solvers):
    """One common horizontal approach for the palm targets fixed in O, validated (one hand's or both hands'; the
    other arm keeps its joint angles).

    This is a kinematic suggestion, never an automatic walk or a scene/path
    clearance guarantee. It is deliberately omitted if the hands disagree.
    """
    sides = [s for s in SIDES if s in goals]
    if not sides:
        return None
    vectors = np.array([goals[s][0][:2] - solvers[s].geometry.shoulder_in_body(pose)[:2] for s in sides])
    norms = np.linalg.norm(vectors, axis=1)
    if min(norms) < 1e-6:
        return None
    directions = vectors / norms[:, None]
    if len(sides) == 2 and directions[0] @ directions[1] < 0.8:
        return None
    direction = directions.mean(axis=0)
    direction /= np.linalg.norm(direction)
    # Coarse 2 cm suggestions, at most half a metre. Check both full palm
    # orientations and retain 4 cm of geometric extension reserve.
    for distance in np.arange(0.02, 0.501, 0.02):
        shift = np.r_[distance * direction, 0.0]
        results = {}
        for side in sides:
            position, rotation = goals[side]
            target = position - shift
            solver = solvers[side]
            if solver.geometry.reach(pose, target)["distance_m"] > solver.geometry.comfort_reach:
                break
            result = solver.solve(pose, target, rotation, solver.nominal)
            if not result.feasible:
                break
            results[side] = result
        if len(results) == len(sides):
            # The single-arm solutions must also coexist with each other and the other arm. The independent
            # solvers deliberately do not move or constrain the opposite arm.
            combined = pose.copy()
            for s in sides:
                combined[arm(s)] = results[s].joints
            kin = solvers[SIDES[0]].kin
            kin.forward(combined)
            robot = set(kin.collisions)
            if any(
                c.geom[0] in robot and c.geom[1] in robot and c.dist < -L.SELF_PENETRATION for c in kin.data.contact
            ):
                continue
            return dict(
                translation_O_m=(body_rotation(pose) @ shift).tolist(),
                verified="the palm poses at the same frozen body configuration",
                limitation="Choose a clear walking path, then observe, perceive and re-plan the hands after moving.",
                arms={s: _feedback(r, pose) for s, r in results.items()},
            )
    return None


def approach_hint(pose, paths, frame):
    """One verified approach suggestion after all body candidates are exhausted."""
    goals = {side: path.at(frame, pose) for side, path in paths.items() if path.kind == "pose" and path.moves}
    return _translation_hint(pose, goals, {side: ArmIK(side) for side in SIDES})


def _pose_error(reason, frame, pose, goals, solvers, failures, *, suggestions=True):
    """Raise the failure of the palms at ``frame``, with an approach hint when ``suggestions``."""
    hint = _translation_hint(pose, goals, solvers) if suggestions else None
    raise WbcError(
        ErrorCode.PLAN_REJECTED,
        "A palm target is farther from its shoulder than the arm reaches."
        if reason == "HAND_REACH_LIMIT"
        else "An arm cannot hold its palm pose: position, orientation, joint limits or clearance from the body.",
        reason=reason,
        stage="hand_pose",
        frame=int(frame),
        time_s=float(frame * DT),
        body_height_m=float(pose[2]),
        arms=failures,
        suggestion=hint,
        evidence="geometric_upper_bound"
        if reason == "HAND_REACH_LIMIT"
        else "warm_start_did_not_reach_pose_tolerance; not_a_global_infeasibility_proof",
    )


def solve(base, plans, *, posture_weight=POSTURE_WEIGHT, suggestions=True):
    """Both arms over ``base`` (root, waist and legs as given, each frame's actual body pose) along ``plans``.

    Mechanical reach is checked over the full path before continuation. A geometrically feasible sample the arm
    cannot reach in one frame passes to the common clock (``planner.arms``), which slows the path.
    """
    frames = base.frames.copy()
    solvers = {s: ArmIK(s, posture_weight=posture_weight) for s in SIDES}
    for s, solver in solvers.items():
        low, high = solver.geometry.limits.T
        q = frames[0, arm(s)]
        if np.any(q < low - 1e-6) or np.any(q > high + 1e-6):
            raise WbcError(
                ErrorCode.PRECONDITION,
                "Starting arm reference is outside physical joint limits.",
                reason="HAND_START_LIMIT",
                stage="hand_start",
                side=s,
            )
    goals = {s: [p.at(k, pose) for k, pose in enumerate(frames)] for s, p in plans.items()}
    palms = [s for s in SIDES if plans[s].kind == "pose"]
    resting = [s for s in SIDES if plans[s].kind == "rest"]
    fixed = [s for s in palms if getattr(plans[s], "moves", True)]  # targets fixed in O
    reach = {}
    violations = []
    for s in palms:
        rows = [_reach(solvers[s], pose, goals[s][k][0]) for k, pose in enumerate(frames)]
        worst = int(np.argmax([r["distance_m"] for r in rows]))
        reach[s] = dict(frame=worst, **rows[worst])
        violations += [(k, s, r) for k, r in enumerate(rows) if r["distance_m"] > r["absolute_radius_m"] + 1e-9]
    if violations:
        # Report the worst target, rather than a barely-over-limit early sample.
        k, _, _ = max(violations, key=lambda x: x[2]["distance_m"] - x[2]["absolute_radius_m"])
        failures = {s: _reach(solvers[s], frames[k], goals[s][k][0]) for s in palms}
        _pose_error(
            "HAND_REACH_LIMIT",
            k,
            frames[k],
            {s: goals[s][k] for s in fixed},
            solvers,
            failures,
            suggestions=suggestions,
        )

    errors = {side: [] for side in palms}
    step = L.DEFAULT.upper_speed * DT * (1 - 1e-6)
    for k in range(1, len(frames)):
        failures = {}
        for side in SIDES:
            sl = arm(side)
            previous = frames[k - 1, sl]
            same_goal = (
                all(np.array_equal(a, b) for a, b in zip(goals[side][k], goals[side][k - 1], strict=True))
                if side in palms
                else np.array_equal(goals[side][k], goals[side][k - 1])
            )
            if k > 1 and same_goal and np.array_equal(frames[k, : ARMS.start], frames[k - 1, : ARMS.start]):
                # Reuse the already accepted pose for a true hold. Continuing
                # secondary optimization here would turn a stationary tail into
                # another arm movement and require extra controller padding.
                frames[k, sl] = previous
                if side in palms:
                    errors[side].append(errors[side][-1])
                continue
            if side not in palms and side not in resting:
                frames[k, sl] = goals[side][k]
                continue
            solver = solvers[side]
            options = {"bounds": (previous - step, previous + step)}
            if side in resting:
                fit = partial(solver.rest, frames[k], goals[side][k], previous)
            else:
                angle_limit = plans[side].tolerance(k)
                fit = partial(solver.solve, frames[k], *goals[side][k], previous, rotation_tolerance=angle_limit)
            result = fit(**options)
            if not result.feasible:
                # One diagnostic at the failed sample separates insufficient
                # time from geometry. Successful samples are never solved twice.
                geometric = fit()
                if geometric.feasible:
                    result = geometric
                elif side in resting:
                    raise WbcError(
                        ErrorCode.PLAN_REJECTED,
                        "The resting arm cannot clear the body; adjust the body pose.",
                        reason="COLLISION",
                        stage="hand_rest",
                        frame=k,
                        side=side,
                        minimum_body_clearance_m=geometric.clearance,
                    )
                else:
                    failures[side] = {
                        **_feedback(geometric, frames[k]),
                        **_reach(solver, frames[k], goals[side][k][0]),
                        "joints_at_limit": _at_limit(solver, geometric.joints),
                    }
            frames[k, sl] = result.joints
            if side in palms:
                errors[side].append((np.linalg.norm(result.position_error), result.rotation_error))
        if failures:
            _pose_error(
                "HAND_POSE_UNREACHABLE",
                k,
                frames[k],
                {s: goals[s][k] for s in fixed},
                solvers,
                failures,
                suggestions=suggestions,
            )
    rows = [np.asarray(samples) for samples in errors.values() if samples]
    quality = np.max([np.maximum(np.quantile(row, 0.95, axis=0), row[-1]) for row in rows] or [[0, 0]], axis=0)
    # Palm errors exist only where an arm had a palm target; joint-only arms report none.
    accuracy = {}
    if rows:
        max_position, max_rotation = np.concatenate(rows).max(axis=0)
        accuracy = {"hand_position_error_m": max_position, "hand_rotation_error_rad": max_rotation}
    keyframes = set(base.keyframes)
    for s in palms:
        for first, n, *_ in plans[s].pieces:
            keyframes.update((first + n // 2, first + n))
    return replace(
        base,
        frames=frames,
        diagnostics={
            **base.diagnostics,
            "hand_reach": reach,
            **accuracy,
            "hand_posture_weight": posture_weight,
            "hand_quality": [float(v) for v in quality],
        },
        keyframes=tuple(sorted(k for k in keyframes if 0 <= k < len(frames))),
    )


# Resampling interpolates between solved frames, so a frame solved at the edge of the IK tolerance (a palm near the
# arm's reach) can come out a few hundredths of a millimetre past it. A resampled frame is held to the tolerance, or,
# beyond it, to the solved frames it lies between plus this much; an interpolation that leaves the path (an IK branch
# jump between frames) is still refused.
RESAMPLE_SLACK_M = 0.0005
RESAMPLE_SLACK_RAD = np.radians(1.0)


def _palm_error(kin, pose, path, source, side):
    """(position error m, rotation error rad) of a palm against its path at ``source`` (a fractional solved frame)."""
    kin.forward(pose)
    heading = body_rotation(pose)
    position, rotation = path.at(source, pose)
    site = kin.palms[SIDES.index(side)]
    error = position - heading.T @ (kin.data.site_xpos[site] - pose[:3])
    angle = Rotation.from_matrix(rotation @ (heading.T @ kin.data.site_xmat[site].reshape(3, 3)).T).magnitude()
    return float(np.linalg.norm(error)), float(angle)


def validate_timing(base, plans, source_frames, solved):
    """Recheck palm FK after resampling ``solved`` (the frames before it); whole-body geometry is checked once by the
    planner."""
    if not any(path.kind == "pose" for path in plans.values()):
        return base  # no palm targets: nothing to recheck, no palm error to report
    kin = Kinematics()
    max_position, max_rotation = 0.0, 0.0
    for k, (pose, source) in enumerate(zip(base.frames, source_frames, strict=True)):
        for side, path in plans.items():
            if path.kind != "pose":
                continue
            distance, angle = _palm_error(kin, pose, path, source, side)
            limit = path.tolerance(int(np.floor(source)))
            if distance > POSITION_TOL or angle > limit:
                bracket = [
                    _palm_error(kin, solved[j], path, j, side)
                    for j in {int(np.floor(source)), min(int(np.ceil(source)), len(solved) - 1)}
                ]
                if (
                    distance > max(POSITION_TOL, *(b[0] for b in bracket)) + RESAMPLE_SLACK_M
                    or angle > max(limit, *(b[1] for b in bracket)) + RESAMPLE_SLACK_RAD
                ):
                    raise WbcError(
                        ErrorCode.PLAN_REJECTED,
                        "Time-resampled path did not retain the palm constraints.",
                        reason="HAND_TIMING_GEOMETRY",
                        stage="hand_timing",
                        frame=k,
                        side=side,
                        source_frame=float(source),
                        position_error_m=distance,
                        rotation_error_deg=float(np.degrees(angle)),
                        solved_position_error_m=max(b[0] for b in bracket),
                        solved_rotation_error_deg=float(np.degrees(max(b[1] for b in bracket))),
                    )
            max_position = max(max_position, distance)
            max_rotation = max(max_rotation, angle)
    return replace(
        base,
        diagnostics={
            **base.diagnostics,
            "hand_position_error_m": max_position,
            "hand_rotation_error_rad": max_rotation,
        },
    )
