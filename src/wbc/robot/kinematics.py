"""Forward kinematics on the shared model."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core.frames import heading
from wbc.core.layout import SIDES
from wbc.robot.model import collision_geoms, lowest_point, robot_model


class Kinematics:
    """A model + data pair with named handles. Not thread-safe; create one per user.

    Without a model it reads the shared robot model (it never changes a model).
    """

    def __init__(self, model: mujoco.MjModel | None = None):
        self.model = model if model is not None else robot_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.feet = [m.body(f"{s}_ankle_roll_link").id for s in SIDES]
        self.palms = [m.site(f"{s}_palm").id for s in SIDES]
        self.pelvis = m.body("pelvis").id
        self.collisions = collision_geoms(m)

    def forward(self, pose) -> mujoco.MjData:
        self.data.qpos[: len(pose)] = pose
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def foot_positions(self) -> np.ndarray:
        return self.data.xpos[self.feet].copy()

    def foot_rotations(self) -> np.ndarray:
        return self.data.xmat[self.feet].reshape(2, 3, 3).copy()

    def com(self) -> np.ndarray:
        return self.data.subtree_com[self.pelvis].copy()

    def lowest_collision(self) -> tuple[float, int]:
        heights = [lowest_point(self.model, self.data, g) for g in self.collisions]
        index = int(np.argmin(heights))
        return heights[index], self.collisions[index]

    def effectors(self, joints) -> dict[str, np.ndarray]:
        """Palm and ankle positions in the pelvis frame for these 29 joint angles."""
        self.forward(np.r_[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, np.asarray(joints, float)])
        out = {f"{s}_hand": self.data.site_xpos[p].copy() for s, p in zip(SIDES, self.palms, strict=True)}
        out.update({f"{s}_foot": self.data.xpos[f].copy() for s, f in zip(SIDES, self.feet, strict=True)})
        return out

    def palm_in_body(self, pose, side: int) -> tuple[np.ndarray, np.ndarray]:
        """Palm position and rotation in the body frame of ``pose``."""
        self.forward(pose)
        r = body_rotation(pose)
        site = self.palms[side]
        position = r.T @ (self.data.site_xpos[site] - np.asarray(pose[:3]))
        return position, r.T @ self.data.site_xmat[site].reshape(3, 3)


def body_rotation(pose) -> np.ndarray:
    """Heading-only rotation: the pelvis heading frame (palm paths, arm IK)."""
    return Rotation.from_euler("z", heading(np.asarray(pose)[3:7])).as_matrix()


def grounded_pose(joints, imu_quat_wxyz, kinematics: Kinematics | None = None) -> np.ndarray:
    """Pose in the observation frame O from encoders and IMU alone.

    Yaw is removed (O is heading-aligned at the pelvis), and the root height is
    chosen so that the lowest robot collision surface touches z = 0.
    """
    kin = kinematics or Kinematics()
    attitude = Rotation.from_quat(np.asarray(imu_quat_wxyz, float), scalar_first=True)
    level = Rotation.from_euler("z", -heading(imu_quat_wxyz)) * attitude
    pose = np.r_[0.0, 0.0, 0.0, level.as_quat(scalar_first=True), np.asarray(joints, float)]
    kin.forward(pose)
    lowest, _ = kin.lowest_collision()
    pose[2] = -lowest
    return pose


def link_transform(kin: Kinematics, pose, link: str) -> np.ndarray:
    """4x4 world transform of a robot link at ``pose``."""
    kin.forward(pose)
    body = kin.model.body(link).id
    transform = np.eye(4)
    transform[:3, :3] = kin.data.xmat[body].reshape(3, 3)
    transform[:3, 3] = kin.data.xpos[body]
    return transform
