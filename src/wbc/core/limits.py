"""The pass/fail thresholds every plan is checked against.

A skill names one of the profiles below; it keeps only the bounds of what it
can build (path length, reach) next to its own code.
Values are deliberately generous: they reject references that are clearly
broken.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LimitProfile:
    name: str
    leg_speed: float  # rad/s, any leg joint
    upper_speed: float  # rad/s, any waist/arm joint
    stance_slip: float  # m/s, a foot marked as support
    reason: str
    root_speed: float = 2.0  # m/s of the pelvis; skills peak below 1 m/s, so this only catches jumps
    root_turn: float = 4.0  # rad/s of the pelvis orientation


DEFAULT = LimitProfile("default", 4.0, 4.0, 0.02, "quasi-static gait, posture and hand motion")
# Walking: limits for a dynamic gait, standing or crouched.
WALK = LimitProfile("walk", 8.0, 4.0, 0.02, "dynamic standing gait", root_turn=8.0)
LOW_GAIT = LimitProfile("low_gait", 8.0, 4.0, 0.02, "crouched gait needs fast knee/ankle flexion", root_turn=8.0)

JOINT_LIMIT_TOLERANCE = 0.01  # rad beyond the model range
IK_RESIDUAL = 0.01  # m or rad, worst task residual a skill may report
SELF_PENETRATION = 0.005  # m between robot collision geoms
SELF_COLLISION_MARGIN = 0.005  # m the IK keeps between robot collision geoms (absorbs linearisation)
GROUND_PENETRATION = 0.01  # m below the ground plane
START_TOLERANCE = 1e-6  # frames[0] must reproduce the start pose
STATE_AGE = 2.0  # s, robot state older than this makes the session stale
SEGMENT_FRAMES = 4096  # frames the controller takes in one append (its motion channel's MAX_FRAMES)
BALANCE_MARGIN = 0.0  # m, CoM inside the support hull (warning only)
QUATERNION_NORM = 1e-3  # a root quaternion's norm differs from 1 by less than this (finite check)
FIXED_FOOT_POSITION = 0.001  # m a foot planted through the in-place part may move
FIXED_FOOT_ROTATION = 0.001  # rad it may turn

# The walker's own accuracy gates (skills.walking.validation), on top of the checks above. Generator-exact: a step
# meets them to the millimetre, so a miss means the generator failed, not that the robot tracks worse.
WALKER_UNCHANGED = 1e-10  # the start pose, and the upper body the walker must not move, kept to rounding
WALKER_IK_ACCURACY = 0.001  # scaled foot/CoM residual of the body IK
WALKER_SOLE_PENETRATION = 0.001  # m a sole may sink below the ground
WALKER_LANDING_POSITION = 0.001  # m between an explicit footstep's ankle XY and where it lands
WALKER_LANDING_HEADING = 0.001  # rad between an explicit footstep's yaw and the landed foot
WALKER_ROOT_TARGET = 0.005  # m between a walk's pelvis destination and where it ends
WALKER_ROOT_HEADING = 0.001  # rad between a walk's final heading and the one asked for

# The reason each robot check stands for in a refused plan (docs/REFERENCE.md 4.3; the agent's result names it).
CHECK_REASONS = {
    "ik_residual": "BODY_UNREACHABLE",
    "self_collision": "SELF_COLLISION",
    "ground_penetration": "GROUND_CONTACT",
    "joint_limits": "JOINT_LIMIT",
    "leg_speed": "SPEED_LIMIT",
    "upper_speed": "SPEED_LIMIT",
    "root_speed": "SPEED_LIMIT",
    "root_turn": "SPEED_LIMIT",
    "fixed_feet_position": "FEET_MOVED",
    "fixed_feet_rotation": "FEET_MOVED",
    "stance_slip": "FEET_MOVED",
    "finite": "INVALID_TRAJECTORY",
}
# The checks that only say a plan is too fast: slower timing can fix them.
SPEED_CHECKS = frozenset(name for name, reason in CHECK_REASONS.items() if reason == "SPEED_LIMIT")
SLOWER = 1.5  # a request refused only for speed is planned once more with this much slower body and walk timing
