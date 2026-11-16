"""G1 arm geometry, measured at the palm site (not at the fingertips), and the arm postures a hand plan starts from:
at rest (the nominal arm, or close to it) or hanging as in the simulator's start pose."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from wbc.core.layout import SIDES, arm
from wbc.paths import INITIAL_REFERENCE
from wbc.robot.hands import HANDS, current_hand
from wbc.robot.kinematics import Kinematics, body_rotation
from wbc.robot.model import NOMINAL_JOINTS, descendants, self_collision_pairs

HOME_TOLERANCE = 0.05  # rad, arms count as home within this of the nominal pose
NEAR_REST = 0.3  # rad: an arm this close to its rest posture is one that rests (it may have made room for the knees)


def nominal_arm(side: str) -> np.ndarray:
    return NOMINAL_JOINTS[arm(side).start - 7 : arm(side).stop - 7]


def arm_is_home(pose, side: str, tolerance: float = HOME_TOLERANCE) -> bool:
    return float(np.max(np.abs(np.asarray(pose)[arm(side)] - nominal_arm(side)))) <= tolerance


@lru_cache(maxsize=8)
def _start_arm(hand: str, side: str) -> tuple:
    """An arm's joint angles in the simulator's start pose for a hand (hands.Hand.rest of the initial reference)."""
    return tuple(HANDS[hand].rest(np.load(INITIAL_REFERENCE, allow_pickle=False)["qpos"][0])[arm(side)])


def arm_is_hanging(pose, side: str, tolerance: float = NEAR_REST) -> bool:
    """Whether an arm hangs as it does at the start: not its rest posture, but not holding anything either."""
    hanging = np.asarray(_start_arm(current_hand().name, side))
    return float(np.max(np.abs(np.asarray(pose)[arm(side)] - hanging))) <= tolerance


@dataclass
class ArmGeometry:
    side: str
    kin: Kinematics

    def __post_init__(self):
        m = self.kin.model
        self.slice = arm(self.side)
        self.cols = np.arange(self.slice.start - 1, self.slice.stop - 1)
        self.site = self.kin.palms[SIDES.index(self.side)]
        self.shoulder = m.body(f"{self.side}_shoulder_pitch_link").id
        self.limits = m.jnt_range[1 + self.slice.start - 7 : 1 + self.slice.stop - 7].copy()
        # Triangle inequality along the actual kinematic chain. Deliberately
        # loose: orientation and joint limits can only reduce this distance.
        b = int(m.site_bodyid[self.site])
        length = np.linalg.norm(m.site_pos[self.site])
        while b != self.shoulder:
            length += np.linalg.norm(m.body_pos[b])
            b = int(m.body_parentid[b])
        self.max_reach = float(length)
        self.comfort_reach = min(0.45, self.max_reach)
        # Each 7-DOF solve avoids its own arm and the frozen body. Opposite-arm
        # collisions are checked on the complete motion, not solved by secretly
        # moving the other arm or depending on which side was solved first.
        other = "right" if self.side == "left" else "left"
        moving = descendants(m, self.shoulder)
        opposite = descendants(m, m.body(f"{other}_shoulder_pitch_link").id)
        self.pairs = []
        for a, b in self_collision_pairs(m):
            ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            if ba not in moving and bb not in moving:
                continue
            if ba in opposite or bb in opposite:
                continue
            self.pairs.append((a, b))

    def shoulder_in_body(self, pose):
        self.kin.forward(pose)
        return body_rotation(pose).T @ (self.kin.data.xpos[self.shoulder] - pose[:3])

    def reach(self, pose, position):
        shoulder = self.shoulder_in_body(pose)
        return dict(
            shoulder_body_m=shoulder.tolist(),
            target_body_m=np.asarray(position).tolist(),
            distance_m=float(np.linalg.norm(position - shoulder)),
            comfortable_radius_m=self.comfort_reach,
            absolute_radius_m=self.max_reach,
        )
