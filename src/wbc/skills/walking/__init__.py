"""Body reference generation, stance recovery and nominal-arm mass prior.

No execution: the planner restores the actual arms and checks the robot geometry. ``plan_body`` is the walk skill's
entry; the request planner also uses ``Stance`` (the feet as they stand).
"""

from dataclasses import replace

from wbc.core.layout import ARMS
from wbc.core.types import StartState
from wbc.robot.model import BALANCE_ARM_JOINTS

from .stance import Stance
from .walker import plan

__all__ = ["Stance", "plan_body"]


def plan_body(start: StartState, **parameters):
    """Plan with the nominal arm mass distribution, then restore the actual arms (held at their joint angles).

    The temporary nominal arms never reach the controller. Robot geometry checks run on the actual arms.
    """
    neutral = start.pose.copy()
    neutral[ARMS] = BALANCE_ARM_JOINTS
    body_start = replace(start, pose=neutral)
    body = plan(body_start, **parameters)
    frames = body.motion.frames.copy()
    frames[:, ARMS] = start.pose[ARMS]
    frames[0] = start.pose
    return replace(
        body.motion,
        frames=frames,
        diagnostics={**body.motion.diagnostics, "walker": body.feedback()},
    )
