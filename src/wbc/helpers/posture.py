"""Fixed-support transitions: both feet stay planted while the body moves.

One solver for the in-place body (``helpers.body``), the height candidates of a hand request's height search
(``planner.arms``): feet 6D + CoM x, with the root height and attitude prescribed per frame. ``fixed_feet`` is the
pelvis height change of that search.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core import limits as L
from wbc.core.errors import invalid
from wbc.core.frames import SE2, continuous_quaternions, heading
from wbc.core.layout import ARMS, ROOT_QUAT, UPPER, WAIST
from wbc.core.types import MotionDraft, StartState, phases_from_labels, posture_for_height
from wbc.helpers.gait import MAX_HEIGHT, balance_pose, stand_balance
from wbc.helpers.ik import ComXY, Part, TaskIK, Worst, feet_tasks
from wbc.helpers.interp import frames_for, slerp, smooth
from wbc.robot.kinematics import Kinematics
from wbc.robot.model import BALANCE_ARM_JOINTS

# The lowest pelvis height a fixed-foot height change is asked for (the request's height range starts here); the
# stance and fixed-foot checks still apply. The walking gait's own floor is helpers.gait.MIN_HEIGHT.
CROUCH_FLOOR_M = 0.30


@dataclass(frozen=True)
class BodyGoal:
    """Desired body configuration at progress u in [0, 1] of a transition."""

    height: float
    attitude: np.ndarray  # wxyz
    upper: np.ndarray  # 17 waist+arm joints
    com: np.ndarray | float | None = None  # CoM x (scalar) or xy target


class FixedSupport:
    """Solve body poses with both feet pinned to their start placement."""

    def __init__(self, start_pose, *, placeholder_arms: bool = False):
        """``placeholder_arms``: the poses carry the nominal balance arms for their mass only (ik.TaskIK)."""
        self.ik = TaskIK(joints=range(12), root="x", placeholder_arms=placeholder_arms)
        self.kin = self.ik.kin
        self.kin.forward(start_pose)
        self.feet = self.kin.foot_positions()
        self.rotations = self.kin.foot_rotations()
        self.com_x = float(self.kin.com()[0])

    def solve(self, pose, w: BodyGoal) -> tuple[np.ndarray, Part | None]:
        pose = pose.copy()
        pose[ROOT_QUAT] = w.attitude
        pose[UPPER] = w.upper
        tasks = feet_tasks(self.feet, self.rotations)
        if w.com is not None:
            com = np.atleast_1d(np.asarray(w.com, float))
            tasks.append(ComXY("com", 1.0, com, axes=tuple(range(len(com)))))
        result = self.ik.solve(pose, tasks, height=w.height, joint_step=0.07, root_step=0.015)
        return result.pose, result.worst(("feet",))


def transition(
    start: StartState, target: Callable[[float], BodyGoal], count: int, label: str, solver: FixedSupport | None = None
) -> tuple[list[np.ndarray], list[str], Worst]:
    """Frames 1..count of a smooth transition (frame 0 is the start pose)."""
    solver = solver or FixedSupport(start.pose)
    pose = start.pose.copy()
    frames, labels, residual = [], [], Worst()
    for tick in range(1, count + 1):
        pose, error = solver.solve(pose, target(smooth(tick / count)))
        residual.add(error, tick)
        frames.append(pose)
        labels.append(label)
    return frames, labels, residual


def _balance_attitude(height, yaw: float):
    """Balanced pelvis pitch for ``height``, keeping the current heading."""
    return Rotation.from_euler("ZY", [yaw, balance_pose(height)[0]]).as_quat(scalar_first=True)


def body_transition(start: StartState, height: float, upper_goal, seconds: float, label: str):
    solver = FixedSupport(start.pose, placeholder_arms=True)  # the body is solved with the balance arms
    attitude = slerp(start.pose[ROOT_QUAT], _balance_attitude(height, heading(start.pose[ROOT_QUAT])))
    com_goal = float(solver.feet[:, 0].mean() + stand_balance(height))
    upper0 = start.pose[UPPER].copy()

    def target(u):
        return BodyGoal(
            (1 - u) * start.height + u * height,
            attitude(u),
            (1 - u) * upper0 + u * upper_goal,
            (1 - u) * solver.com_x + u * com_goal,
        )

    return transition(start, target, frames_for(seconds), label, solver)


def balance_start(start: StartState) -> tuple[SE2, StartState]:
    """``start`` in its own heading frame with the nominal balance arms (their mass only), and that frame."""
    frame = SE2.of_pose(start.pose)
    neutral = frame.inverse().apply_poses(start.pose)
    neutral[ARMS] = BALANCE_ARM_JOINTS
    return frame, StartState(neutral, start.posture)


def actual_arms(frame: SE2, local: StartState, poses, start: StartState) -> np.ndarray:
    """The frames of a plan from ``balance_start`` (``poses`` after ``local``) back in the frame of ``start``, its
    arms at their starting joint angles."""
    frames = continuous_quaternions(frame.apply_poses(np.vstack([local.pose, poses])))
    frames[:, ARMS] = start.pose[ARMS]
    frames[0] = start.pose
    return frames


def fixed_feet(start: StartState, *, height_m: float, duration_s: float, label: str = "height") -> MotionDraft:
    """Build an unpadded body interval on the current feet, with nominal arm mass.

    This primitive preserves staggered feet too. Callers decide whether to collect
    feet first, solve the actual arms afterwards, and validate the full trajectory.
    """
    if not np.isfinite(height_m) or not CROUCH_FLOOR_M <= height_m <= MAX_HEIGHT:
        raise invalid(f"height_m must be {CROUCH_FLOOR_M}..{MAX_HEIGHT} m")
    frame, local = balance_start(start)
    upper = local.pose[UPPER].copy()
    upper[WAIST.start - UPPER.start : WAIST.stop - UPPER.start] = [
        0,
        0,
        balance_pose(height_m)[1],
    ]
    poses, labels, residual = body_transition(local, height_m, upper, duration_s, label)
    frames = actual_arms(frame, local, poses, start)
    return MotionDraft(
        frames,
        phases_from_labels(["start", *labels]),
        posture_for_height(height_m),
        support=np.ones((len(frames), 2), bool),
        limit_stages=((0, L.DEFAULT),),
        diagnostics={
            **residual.diagnostics(),
            "foot_width_m": _foot_width(local.pose),
            "height_target_m": height_m,
        },
    )


def _foot_width(pose) -> float:
    """The ankles' lateral distance in the pose's heading frame, read as ``skills.walking.Stance.width`` reads it."""
    kin = Kinematics()
    kin.forward(SE2.of_pose(pose).inverse().apply_poses(pose))
    feet = kin.foot_positions()
    return float(feet[0, 1] - feet[1, 1])
