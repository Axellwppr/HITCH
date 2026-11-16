"""The in-place body: pelvis height plus torso lean and twist, both feet planted (docs/REFERENCE.md 3.2).

At every pelvis height the body keeps a standard posture (``gait.balance_pose``: the torso upright down to 0.62 m,
then pitched forward as little as the ankles allow); the walk shares it. A request may lean the torso further forward
and turn it at the waist. The waist pitch takes the lean first, up to its limit, then the pelvis (hip flexion); the CoM
stays at the standard point over the feet (``gait.stand_balance``), so the pelvis moves back. The twist is the waist
yaw. The lean and turn a request sets are recorded as state (``StartState.torso``, kept in the run records), not read
back from joint angles; ``torso_limits`` bounds them by pelvis height, checked at every frame of a transition that
changes height and torso together. The walk needs the torso returned first, and nothing returns it automatically.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core import limits as L
from wbc.core.errors import ErrorCode, WbcError
from wbc.core.layout import DT, JOINTS, ROOT_QUAT, UPPER, WAIST
from wbc.core.torso import torso_limits
from wbc.core.types import MotionDraft, StartState, phases_from_labels, posture_for_height
from wbc.helpers.gait import balance_pose, stand_balance
from wbc.helpers.interp import frames_for, smooth
from wbc.helpers.posture import BodyGoal, FixedSupport, actual_arms, balance_start, transition

WAIST_PITCH_MAX = 0.50  # rad: the most waist pitch a lean takes, inside the 0.52 rad joint limit
LEAN_RATE = np.radians(25.0)  # rad/s of torso pitch or yaw change
LEAN_MIN_S = 1.0  # shortest torso change
HEIGHT_RATE_S = 4.0  # s per metre of height change
HEIGHT_MIN_S = 1.2  # shortest height change
_WAIST_PITCH = JOINTS.index("waist_pitch_joint")
_WAIST_YAW = JOINTS.index("waist_yaw_joint")


def split_lean(height: float, pitch: float) -> tuple[float, float]:
    """(pelvis pitch, waist pitch) in rad for a lean ``pitch`` (rad) beyond the standard posture at ``height``: the
    waist first, up to its limit, then the hips."""
    pelvis0, waist0 = balance_pose(height)
    waist = float(np.clip(waist0 + pitch, -0.2, WAIST_PITCH_MAX))
    return pelvis0 + pitch - (waist - waist0), waist


def torso_offset(pose) -> tuple[float, float]:
    """(lean, turn) in degrees of a pose's torso beyond the standard posture of its pelvis height (measured)."""
    pose = np.asarray(pose, float)
    pelvis0, waist0 = balance_pose(float(pose[2]))
    pelvis = Rotation.from_quat(pose[ROOT_QUAT], scalar_first=True).as_euler("ZYX")[1]
    lean = (pelvis - pelvis0) + (pose[7 + _WAIST_PITCH] - waist0)
    return float(np.degrees(lean)), float(np.degrees(pose[7 + _WAIST_YAW]))


def torso_pitch(pose) -> float:
    """The torso's forward pitch from vertical in degrees: pelvis pitch plus waist pitch (measured)."""
    pose = np.asarray(pose, float)
    pelvis = Rotation.from_quat(pose[ROOT_QUAT], scalar_first=True).as_euler("ZYX")[1]
    return float(np.degrees(pelvis + pose[7 + _WAIST_PITCH]))


def body_seconds(height0: float, height1: float, torso0, torso1) -> float:
    """Default duration of an in-place body change: the slower of the height and the torso change (0: none)."""
    seconds = 0.0
    if abs(height1 - height0) > 1e-6:
        seconds = max(HEIGHT_MIN_S, HEIGHT_RATE_S * abs(height1 - height0))
    change = max(abs(torso1[0] - torso0[0]), abs(torso1[1] - torso0[1]))
    if change:
        seconds = max(seconds, LEAN_MIN_S, np.radians(change) / LEAN_RATE)
    return float(seconds)


def check_limits(height0: float, height1: float, torso0, torso1, count: int) -> None:
    """Refuse a transition whose lean or turn exceeds the bound of the pelvis height at any of its ``count`` frames
    (all three follow the same smooth profile)."""
    for k in range(count + 1):
        u = smooth(k / max(count, 1))
        height = (1 - u) * height0 + u * height1
        lean = (1 - u) * torso0[0] + u * torso1[0]
        turn = (1 - u) * torso0[1] + u * torso1[1]
        lean_max, turn_max = torso_limits(height, lean)
        if lean > lean_max + 1e-6 or abs(turn) > turn_max + 1e-6:
            raise WbcError(
                ErrorCode.PRECONDITION,
                f"at {k * DT:.2f} s the pelvis is {height:.2f} m high, where the torso leans at most {lean_max}° and, "
                f"with a {lean:.0f}° lean, turns at most {turn_max}°; it would lean {lean:.0f}° and turn {turn:.0f}°",
                reason="TORSO_LIMIT",
                time_s=round(k * DT, 2),
                height_m=round(height, 3),
                lean_deg=round(lean, 1),
                turn_deg=round(turn, 1),
                lean_limit_deg=lean_max,
                turn_limit_deg=turn_max,
            )


def body_draft(start: StartState, height: float, torso, seconds: float) -> MotionDraft:
    """The body moving on its planted feet from its pose to ``height`` with the lean and twist ``torso`` (degrees)
    beyond that height's standard posture, arms at their starting joint angles (the hands are solved afterwards);
    a draft in the start's frame.

    Height, lean and turn follow one smooth profile, and every frame's pelvis pitch and waist come from
    ``split_lean`` of that frame's height and lean, so the posture is the one ``check_limits`` checks (the standard
    posture bends at every knot of ``gait.STANCE``: a straight blend of the end postures would lean differently
    between them). Where
    the start pose differs from its own nominal posture, the difference fades out over the transition."""
    count = frames_for(seconds)
    check_limits(start.height, height, start.torso, torso, count)
    frame, local = balance_start(start)  # plan the balance with the nominal arm mass
    neutral = local.pose
    solver = FixedSupport(neutral, placeholder_arms=True)  # the balance arms; the real ones are solved afterwards
    waist = slice(WAIST.start - UPPER.start, WAIST.stop - UPPER.start)  # yaw, roll, pitch among the upper joints
    upper0 = neutral[UPPER].copy()
    pelvis0, waist0 = split_lean(start.height, np.radians(start.torso[0]))
    # what the start pose has beyond its nominal posture: the attitude's residual and the waist's
    residual = (
        Rotation.from_quat(neutral[ROOT_QUAT], scalar_first=True) * Rotation.from_euler("Y", -pelvis0)
    ).as_rotvec()
    waist_residual = upper0[waist] - np.array([np.radians(start.torso[1]), 0.0, waist0])
    com_goal = float(solver.feet[:, 0].mean() + stand_balance(height))

    def target(u):
        h = (1 - u) * start.height + u * height
        lean = np.radians((1 - u) * start.torso[0] + u * torso[0])
        turn = np.radians((1 - u) * start.torso[1] + u * torso[1])
        pelvis, waist_pitch = split_lean(h, lean)
        attitude = Rotation.from_rotvec((1 - u) * residual) * Rotation.from_euler("Y", pelvis)
        upper = upper0.copy()
        upper[waist] = np.array([turn, 0.0, waist_pitch]) + (1 - u) * waist_residual
        return BodyGoal(h, attitude.as_quat(scalar_first=True), upper, (1 - u) * solver.com_x + u * com_goal)

    poses, labels, residual_error = transition(local, target, count, "body", solver)
    frames = actual_arms(frame, local, poses, start)
    return MotionDraft(
        frames,
        phases_from_labels(["start", *labels]),
        posture_for_height(height),
        support=np.ones((len(frames), 2), bool),
        limit_stages=((0, L.DEFAULT),),
        diagnostics=residual_error.diagnostics(),
        end_torso=(int(torso[0]), int(torso[1])),
    )
