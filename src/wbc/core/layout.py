"""The single pose layout: root xyz + root quat wxyz + 29 G1 joints, sampled at 50 Hz."""

FPS = 50
DT = 1.0 / FPS
POSE_SIZE = 36

JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Joint order on the controller wire (policy order). Only the robot link uses it.
WIRE_JOINTS = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# canonical[i] == wire[WIRE_TO_CANONICAL[i]]
WIRE_TO_CANONICAL = tuple(WIRE_JOINTS.index(name) for name in JOINTS)
# wire[i] == canonical[CANONICAL_TO_WIRE[i]]
CANONICAL_TO_WIRE = tuple(JOINTS.index(name) for name in WIRE_JOINTS)

SIDES = ("left", "right")


def _qpos_slice(first: str, last: str) -> slice:
    return slice(7 + JOINTS.index(first), 7 + JOINTS.index(last) + 1)


ROOT_POS = slice(0, 3)
ROOT_QUAT = slice(3, 7)
LEGS = _qpos_slice("left_hip_pitch_joint", "right_ankle_roll_joint")
WAIST = _qpos_slice("waist_yaw_joint", "waist_pitch_joint")
UPPER = _qpos_slice("waist_yaw_joint", "right_wrist_yaw_joint")
ARMS = _qpos_slice("left_shoulder_pitch_joint", "right_wrist_yaw_joint")


def arm(side: str) -> slice:
    return _qpos_slice(f"{side}_shoulder_pitch_joint", f"{side}_wrist_yaw_joint")
