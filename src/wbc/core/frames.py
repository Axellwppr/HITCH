"""Planar frames: W (the controller's reference world), O (an observation), S (a planned step's start).

They share the gravity-aligned z axis and the ground height, so a frame is a
planar transform (x, y, yaw). ``SE2.apply_*`` maps coordinates expressed in
the frame into its parent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core.layout import ROOT_QUAT


def heading(quat_wxyz) -> float:
    """Yaw of the body x axis projected on the ground (falls back to the y axis)."""
    matrix = Rotation.from_quat(np.asarray(quat_wxyz, float), scalar_first=True).as_matrix()
    if np.linalg.norm(matrix[:2, 0]) > 0.1:
        return float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return float(np.arctan2(-matrix[0, 1], matrix[1, 1]))


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class SE2:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    @classmethod
    def of_pose(cls, pose) -> SE2:
        """Frame on the ground below a root pose, x along its heading."""
        pose = np.asarray(pose, float)
        return cls(float(pose[0]), float(pose[1]), heading(pose[ROOT_QUAT]))

    @property
    def rotation(self) -> Rotation:
        return Rotation.from_euler("z", self.yaw)

    def inverse(self) -> SE2:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return SE2(-(c * self.x + s * self.y), s * self.x - c * self.y, -self.yaw)

    def compose(self, child: SE2) -> SE2:
        """Return parent<-child given self = parent<-mid and child = mid<-child."""
        x, y = self.apply_xy([child.x, child.y])
        return SE2(float(x), float(y), float(wrap(self.yaw + child.yaw)))

    def apply_xy(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        x, y = xy[..., 0], xy[..., 1]
        return np.stack([c * x - s * y + self.x, s * x + c * y + self.y], axis=-1)

    def apply_points(self, points) -> np.ndarray:
        points = np.asarray(points, float)
        out = points.copy()
        out[..., :2] = self.apply_xy(points[..., :2])
        return out

    def apply_poses(self, poses) -> np.ndarray:
        """Map [..., 36] poses (root position, root attitude); joints unchanged."""
        poses = np.asarray(poses, float)
        flat = poses.reshape(-1, poses.shape[-1]).copy()
        flat[:, :3] = self.apply_points(flat[:, :3])
        attitude = Rotation.from_quat(flat[:, ROOT_QUAT], scalar_first=True)
        flat[:, ROOT_QUAT] = (self.rotation * attitude).as_quat(scalar_first=True)
        return flat.reshape(poses.shape)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.yaw]

    @classmethod
    def from_list(cls, value) -> SE2:
        x, y, yaw = (float(v) for v in value)
        return cls(x, y, yaw)


def continuous_quaternions(poses) -> np.ndarray:
    """Flip quaternion signs so consecutive samples stay in one hemisphere."""
    poses = np.array(poses, float, copy=True)
    quats = poses[:, ROOT_QUAT]
    for i in range(1, len(quats)):
        if quats[i] @ quats[i - 1] < 0:
            quats[i] = -quats[i]
    poses[:, ROOT_QUAT] = quats
    return poses
