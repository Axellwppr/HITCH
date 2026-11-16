"""The walk's body reference in the caller's frame, from a parsed goal."""

from dataclasses import replace

from wbc.core.errors import invalid
from wbc.core.frames import SE2, continuous_quaternions
from wbc.core.params import reframe
from wbc.core.types import StartState
from wbc.helpers.gait import MAX_HEIGHT, MIN_HEIGHT

from .trajectory import WalkGoal, step_motion


def generate(start: StartState, params: WalkGoal):
    """Internal typed entry: caller has already parsed the original request.

    Re-expressing a goal after collection must not reapply input bounds to a
    derived relative turn (e.g. 0.6 plus the collection's tiny yaw residual).
    Trajectory checks still apply to the complete result.
    """
    to_parent = SE2.of_pose(start.pose)
    to_local = to_parent.inverse()
    local = StartState(to_local.apply_poses(start.pose), start.posture)
    if not MIN_HEIGHT - 1e-6 <= local.height <= MAX_HEIGHT + 1e-6:
        raise invalid(f"walk: pelvis height must be {MIN_HEIGHT:.2f}..{MAX_HEIGHT:.2f} m, got {local.height:.3f}")
    draft = step_motion(local, reframe(params, to_local))
    frames = continuous_quaternions(to_parent.apply_poses(draft.frames))
    frames[0] = start.pose
    return replace(draft, frames=frames)
