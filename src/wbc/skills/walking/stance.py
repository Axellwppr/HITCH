"""Read the reference stance and collect feet by actual, bounded steps.

All geometry is heading-relative; collection keeps the foot midpoint, not a
root XY target whose calibration changes with the body's stance.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core.errors import invalid
from wbc.core.frames import SE2, wrap
from wbc.core.layout import SIDES
from wbc.helpers.gait import gait_profile
from wbc.helpers.support import ANKLE_HEIGHT
from wbc.robot.kinematics import Kinematics
from wbc.robot.model import nominal_pose

from .footsteps import STANCE_WIDTH_M

ALIGNMENT_M = 0.01  # m of stagger (and of width, for a standard or resting stance) still counted as side by side
ALIGNMENT_RAD = 0.05
# Feet within this of the ground count as flat (a measured stance is compared, so it allows for a few millimetres
# of measurement).
GROUND_M = 0.012


@lru_cache(maxsize=1)
def resting_widths():
    """The generator's terminal stance and the robot's nominal stance.

    A normal stance need not match the collection target to the millimetre.
    This interval describes when walking recovery has nothing useful to fix;
    it is not an IK reach limit or a promise that every squat is feasible.
    """
    kin = Kinematics()
    kin.forward(nominal_pose())
    feet = kin.foot_positions()
    return sorted((STANCE_WIDTH_M, float(feet[0, 1] - feet[1, 1])))


@dataclass(frozen=True)
class Stance:
    frame: SE2
    feet: np.ndarray  # in root-heading coordinates
    rotations: np.ndarray
    com: np.ndarray

    @classmethod
    def read(cls, start):
        frame = SE2.of_pose(start.pose)
        kin = Kinematics()
        kin.forward(frame.inverse().apply_poses(start.pose))
        return cls(frame, kin.foot_positions(), kin.foot_rotations(), kin.com())

    @property
    def width(self):
        return float(self.feet[0, 1] - self.feet[1, 1])

    @property
    def flat(self):
        tilt = np.arccos(np.clip(self.rotations[:, 2, 2], -1, 1))
        return bool(abs(self.feet[:, 2] - ANKLE_HEIGHT).max() <= GROUND_M and tilt.max() <= ALIGNMENT_RAD)

    @property
    def aligned(self):
        return bool(
            self.width > 0
            and abs(self.feet[0, 0] - self.feet[1, 0]) <= ALIGNMENT_M
            and Rotation.from_matrix(self.rotations).magnitude().max() <= ALIGNMENT_RAD
        )

    @property
    def standard(self):
        return self.aligned and abs(self.width - STANCE_WIDTH_M) <= ALIGNMENT_M

    @property
    def resting(self):
        low, high = resting_widths()
        return self.aligned and low - ALIGNMENT_M <= self.width <= high + ALIGNMENT_M

    def summary(self):
        return dict(
            width_m=self.width,
            stagger_m=float(self.feet[0, 0] - self.feet[1, 0]),
            flat=self.flat,
            aligned=self.aligned,
            standard=self.standard,
            resting=self.resting,
        )

    def collection(self, height, *, reverse=False):
        """At most 12 landings using the existing gait's step sizes.

        Prefer swinging the foot farther from the projected CoM (less loaded).
        The other order is a bounded alternative, not a parameter search.
        """
        mid = self.feet[:, :2].mean(axis=0)
        goals = mid + np.array([[0, STANCE_WIDTH_M / 2], [0, -STANCE_WIDTH_M / 2]])
        delta = goals - self.feet[:, :2]
        gait = gait_profile(height)
        rounds = max(
            1,
            int(np.ceil(np.max(abs(delta) / [gait.forward_advance, gait.lateral_advance]))),
        )
        if rounds > 6:
            raise invalid(
                "collection needs more than 12 landings",
                reason="REQUEST_LIMIT",
                limit="collection_landings",
                maximum=12,
                requested=2 * rounds,
            )
        first = int(np.argmax(np.linalg.norm(self.feet[:, :2] - self.com[:2], axis=1)))
        if reverse:
            first = 1 - first
        yaws = Rotation.from_matrix(self.rotations).as_euler("xyz")[:, 2]
        steps = []
        for n in range(1, rounds + 1):
            u = n / rounds
            for side in [first, 1 - first]:
                if np.linalg.norm(delta[side]) < 1e-6 and abs(yaws[side]) < 1e-6:
                    continue
                xy = self.frame.apply_xy(self.feet[side, :2] + u * delta[side])
                steps.append(
                    dict(
                        side=SIDES[side],
                        xy=xy.tolist(),
                        yaw=float(wrap(self.frame.yaw + (1 - u) * yaws[side])),
                        root_yaw=self.frame.yaw,
                    )
                )
        return {"footsteps": steps}
