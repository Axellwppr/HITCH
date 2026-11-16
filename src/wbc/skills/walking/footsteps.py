"""The walker's goal in radians (``WalkGoal``, ``Placement``) and automatic placements; no trajectory IK or execution.

The public walk (``skills.walk.WalkParams``, integer degrees) converts to a ``WalkGoal`` once, after the planner has
mapped it into the step's frame; a recovery attempt's goal is restated in degrees (``skills.walk.describe_attempt``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core import limits as L
from wbc.core.errors import invalid
from wbc.core.frames import wrap
from wbc.core.layout import ROOT_QUAT, SIDES
from wbc.core.params import param
from wbc.helpers.gait import GaitProfile, balance_pose, nominal_height
from wbc.helpers.support import ANKLE_HEIGHT

STANCE_WIDTH_M = 0.20  # the walk's terminal stance
TURN_PER_STEP_RAD = 0.5  # largest heading change per automatic footstep


@dataclass(frozen=True)
class Placement:
    """One footstep (``core.request.Footstep`` in radians)."""

    side: Literal["left", "right"]
    xy: tuple[float, ...] = param(length=2, doc="ankle position in the observation frame, m", frame="xy")
    yaw: float | None = param(
        None,
        doc="foot yaw in the observation frame, rad; default keeps its angle to the pelvis",
        frame="yaw",
    )
    root_yaw: float | None = param(
        None,
        doc="pelvis yaw in the observation frame reached during this step, rad",
        frame="yaw",
    )


@dataclass(frozen=True)
class WalkGoal:
    target: tuple[float, ...] | None = param(
        None,
        length=2,
        doc="root destination [x, y] in the observation frame",
        frame="xy",
    )
    turn: float = param(
        0.0,
        range=(-np.pi, np.pi),
        doc="heading change by the end of the walk, rad (left +)",
    )
    footsteps: tuple[Placement, ...] = param((), length=(0, 12), doc="explicit placements instead of target")
    swing_s: float | None = param(None, range=(0.4, 1.8), doc="seconds per swing")
    swing_height_m: float = param(0.045, range=(0.03, 0.28))

    def __post_init__(self):
        if (self.target is None) == (not self.footsteps):
            raise ValueError("give exactly one of target or footsteps")
        if self.footsteps and self.turn:
            raise ValueError("turn goes with target; with footsteps give each step's root_yaw")


def destination_midpoint(start, target_xy, yaw_goal, gait, width, solver, origin):
    """Calibrate the final foot centre against the requested pelvis destination."""
    height = nominal_height(start.height)
    rotation = Rotation.from_euler("z", yaw_goal)
    target_mid = origin + target_xy
    # Calibrate the root-vs-foot-centre offset at the destination posture, then
    # translate the stance so the final root lands on the requested point.
    terminal_feet = np.array(
        [np.r_[target_mid, ANKLE_HEIGHT] + rotation.apply([0, (0.5 - s) * width, 0]) for s in range(2)]
    )
    guess = start.pose.copy()
    guess[:2] = target_xy
    guess[ROOT_QUAT] = (rotation * Rotation.from_euler("y", balance_pose(height)[0])).as_quat(scalar_first=True)
    terminal, worst = solver.solve(
        guess,
        terminal_feet,
        np.repeat(rotation.as_matrix()[None], 2, 0),
        target_mid + rotation.apply([gait.forward_balance, 0, 0])[:2],
        height,
    )
    if worst.value > L.IK_RESIDUAL:
        raise invalid(
            f"walk destination posture is not reachable ({worst.description})",
            reason="TARGET_NOT_REACHED",
        )
    return target_mid + target_xy - terminal[:2]


def turning(start, target_xy, yaw_goal, gait, width, solver, origin):
    """Alternate short arcs about the stance centre; pelvis bisects foot headings.

    Both feet finish at the calibrated destination. A translation-style lateral
    progress constraint is unnecessary here: the stance axes rotate with the
    feet. Full robot collision and planted-foot checks still validate the result.
    """
    target_mid = destination_midpoint(start, target_xy, yaw_goal, gait, width, solver, origin)
    count = max(2, int(np.ceil(abs(yaw_goal) / TURN_PER_STEP_RAD)) + 1)
    first = 0 if yaw_goal > 0 else 1
    solver.kin.forward(start.pose)
    headings = Rotation.from_matrix(solver.kin.foot_rotations()).as_euler("xyz")[:, 2]
    steps = []
    for i in range(count):
        u = min(1.0, (i + 1) / (count - 1))
        angle = u * yaw_goal
        side = (first + i) % 2
        headings[side] = angle
        mid = (1 - u) * origin + u * target_mid
        xy = mid + Rotation.from_euler("z", angle).apply([0, (0.5 - side) * width, 0])[:2]
        steps.append(Placement(SIDES[side], tuple(xy), angle, float(headings.mean())))
    return steps


def automatic(start, target_xy, yaw_goal, gait: GaitProfile, width, solver, origin):
    """Alternate footsteps from the current stance to a root destination."""
    target_mid = destination_midpoint(start, target_xy, yaw_goal, gait, width, solver, origin)
    rotation = Rotation.from_euler("z", yaw_goal)
    delta = target_mid - origin
    # Lateral stepping spends two landings on one advance: open, then close.
    advances = max(gait.step_count(delta), 2 * abs(delta[1]) / gait.lateral_advance)
    count = max(
        2,
        int(np.ceil(advances)) + 1,
        int(np.ceil(abs(yaw_goal) / TURN_PER_STEP_RAD)) + 1,
    )
    solver.kin.forward(start.pose)
    initial_feet = solver.kin.foot_positions().copy()
    initial_yaws = Rotation.from_matrix(solver.kin.foot_rotations()).as_euler("xyz")[:, 2]
    if abs(delta[1]) > abs(delta[0]):
        first = 0 if delta[1] > 0 else 1  # open the stance with the outside foot
    elif np.linalg.norm(delta) > 1e-9:
        first = int(np.argmin((initial_feet[:, :2] - origin) @ delta))  # trailing foot
    else:
        first = 0 if yaw_goal >= 0 else 1
    initial_offsets = initial_feet[:, :2] - origin - np.array([[0, width / 2], [0, -width / 2]])
    minimum_width = min(width, initial_feet[0, 1] - initial_feet[1, 1])
    if minimum_width <= 0:
        raise invalid(
            "automatic walk requires an uncrossed starting stance",
            reason="STANCE_REQUIRED",
        )
    placed = initial_feet[:, :2].copy()
    steps = []
    for i in range(count):
        u = min(1.0, (i + 1) / (count - 1))
        angle = u * yaw_goal
        side = (first + i) % 2
        mid = origin + u * delta
        xy = mid + Rotation.from_euler("z", angle).apply([0, (0.5 - side) * width, 0])[:2]
        xy += (1 - u) * initial_offsets[side]
        # Keep the inside foot behind the outside foot in lateral progress.
        # Projection preserves at least the smaller of the actual initial and
        # default stance widths. Thus sideways walking opens then closes the
        # stance, instead of sending the trailing foot through the other foot.
        lateral_axis = Rotation.from_euler("z", angle).apply([0, 1, 0])[:2]
        sign = 1 if side == 0 else -1
        gap = sign * float((xy - placed[1 - side]) @ lateral_axis)
        xy += sign * max(0.0, minimum_width - gap) * lateral_axis
        placed[side] = xy
        # Turn each foot the same way as the root: wrapping the whole difference
        # would send it the other way round when yaw_goal is pi.
        foot_yaw = initial_yaws[side] + u * float(yaw_goal - wrap(initial_yaws[side]))
        steps.append(Placement(SIDES[side], tuple(xy), foot_yaw, angle))
    # Lateral non-crossing can delay the inside foot. Finish both actual feet
    # at the requested terminal stance, rather than stopping one step short.
    for side in [first, 1 - first]:
        xy = target_mid + rotation.apply([0, (0.5 - side) * width, 0])[:2]
        if np.linalg.norm(placed[side] - xy) > 1e-8:
            steps.append(Placement(SIDES[side], tuple(xy), yaw_goal, yaw_goal))
            placed[side] = xy
    return steps
