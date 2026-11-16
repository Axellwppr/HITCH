"""Where the head camera is for a setting of its two-axis mount (``wbc.core.eye.Eye``; the D455 on two ST3215
servos), as a transform only: the mount's design geometry is the constants below, and nothing of it is added to the
robot model.

The mount sits on ``torso_link`` at ``MOUNT``, turned ``MOUNT_TILT_DEG`` about y (the assembly's axes lean 0.72
degrees against the torso). ``yaw`` turns about the mount's z axis through the yaw axis, ``YAW_OFFSET`` from the mount
(left positive); ``pitch`` turns down about the yawed y axis through the pitch axis, ``PITCH_OFFSET`` from the yaw axis
in the yawed frame (down positive). The optical centre is the D455's RGB lens (11.5 mm right of the camera's centre
line), ``OPTICAL_OFFSET`` from the pitch axis in the pitch body, where the camera looks ``OPTICAL_DOWN_DEG`` down.

Our pitch is the view's angle below the torso's forward direction: 0 looks straight ahead along the torso (with yaw
0), the default 45 down. The mount's pitch servo has its zero at 50 degrees down (``JOINT_ZERO_DEG``), so the servo
angle is ``pitch - JOINT_ZERO_DEG`` (``servo_angles``); the pitch body's fixed turn (``PITCH_SET_DEG``) and the
mount's tilt make the two agree at yaw 0. Both angles are relative to the torso: leaning the torso forward turns the
view down with it. The mount's limits are yaw ±48 and pitch 0-63 (vision/config.json); the model uses less
(``wbc.core.eye``).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core.eye import Eye

# torso_link frame, metres: the mount, its yaw axis, its pitch axis and the camera, each in its parent
MOUNT = np.array([0.0076864148932334, 0.0, 0.3823833123616])
MOUNT_TILT_DEG = -0.7248449788842486  # about the torso's y
YAW_OFFSET = np.array([0.0938, 0.0, 0.019158474645501])  # in the mount frame
PITCH_OFFSET = np.array([0.0245, 0.0, 0.0384])  # in the yawed frame
PITCH_SET_DEG = 5.724844978884216  # the pitch body's fixed turn about y, with the joint at zero
JOINT_ZERO_DEG = 50.0  # the view's angle down with the pitch joint at zero (yaw 0, torso upright)
OPTICAL_OFFSET = np.array([0.039067649660557, -0.0115, -0.02209708691208])  # in the pitch body
OPTICAL_DOWN_DEG = 45.0  # the camera's tilt in the pitch body
# RGB image: the D455's 90 degree horizontal field at 640 x 400 with square pixels (64 degrees vertical)
WIDTH, HEIGHT = 640, 400
HFOV_DEG = 90.0
FOCAL = (WIDTH / 2) / np.tan(np.radians(HFOV_DEG) / 2)
FOVY_DEG = float(np.degrees(2 * np.arctan((HEIGHT / 2) / FOCAL)))
# MuJoCo camera axes in the camera's frame: image right is -y, image up is +z, the camera looks along +x
_MUJOCO_AXES = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
OPTICAL_FROM_MUJOCO = np.diag([1.0, -1.0, -1.0])  # optical: x right, y down, z forward


def servo_angles(eye: Eye) -> tuple[float, float]:
    """The mount's servo angles (yaw, pitch; degrees) for a setting: pitch counted from the joint's zero."""
    return float(eye.yaw_deg), float(eye.pitch_deg) - JOINT_ZERO_DEG


def mujoco_in_torso(eye: Eye) -> tuple[np.ndarray, np.ndarray]:
    """(position, rotation) of the MuJoCo camera (looking along -z, +y up) in the torso_link frame."""
    yaw_joint, pitch_joint = servo_angles(eye)
    mount = Rotation.from_euler("y", MOUNT_TILT_DEG, degrees=True)
    yawed = mount * Rotation.from_euler("z", yaw_joint, degrees=True)
    pitched = yawed * Rotation.from_euler("y", PITCH_SET_DEG + pitch_joint, degrees=True)
    position = MOUNT + mount.apply(YAW_OFFSET) + yawed.apply(PITCH_OFFSET) + pitched.apply(OPTICAL_OFFSET)
    camera = pitched * Rotation.from_euler("y", OPTICAL_DOWN_DEG, degrees=True)
    return position, camera.as_matrix() @ _MUJOCO_AXES


def optical_in_torso(eye: Eye) -> np.ndarray:
    """4x4 pose of the optical frame (x right, y down, z forward) in the torso_link frame."""
    position, rotation = mujoco_in_torso(eye)
    pose = np.eye(4)
    pose[:3, :3] = rotation @ OPTICAL_FROM_MUJOCO
    pose[:3, 3] = position
    return pose
