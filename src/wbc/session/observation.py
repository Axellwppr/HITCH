"""Observations: one robot state (+ optional RGB-D) frozen as an artifact.

The observation frame O is heading-aligned at the pelvis with z = 0 on the
ground. O maps to the reference world W by the IMU heading and the reference
root xy at the same tick (xy tracking error is not observable without
odometry; the closed loop absorbs it).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from wbc.core.errors import Cause, ErrorCode, WbcError
from wbc.core.frames import SE2, heading
from wbc.helpers.body import torso_offset, torso_pitch
from wbc.link.protocol import RobotLink, RobotState, VisionFrame
from wbc.robot.kinematics import Kinematics, grounded_pose
from wbc.session import depth_refine
from wbc.session.perception import Camera, camera_in_observation
from wbc.session.store import SessionFiles, read_json, write_json
from wbc.session.views import body_description

log = logging.getLogger(__name__)
VISION_RETRIES = 3  # an image taken while the body moved is taken again at most this often
STILL_JOINT_RAD = 0.005  # a joint that moved more than this while the image was taken moved the camera
STILL_ATTITUDE_RAD = 0.005


@dataclass(frozen=True)
class Observation:
    path: Path

    @property
    def id(self) -> str:
        return self.path.name

    def file(self, name: str) -> Path:
        return self.path / name

    @cached_property
    def record(self) -> dict:
        return read_json(self.file("observation.json"))

    @cached_property
    def state(self) -> RobotState:
        return RobotState.from_dict(read_json(self.file("state.json")))

    @property
    def to_world(self) -> SE2:
        return SE2.from_list(self.record["to_world"])

    @property
    def has_vision(self) -> bool:
        return self.record.get("camera") is not None

    @property
    def camera(self) -> Camera:
        return Camera.from_dict(self.record["camera"])

    def rgb(self) -> np.ndarray:
        return np.asarray(Image.open(self.file("rgb.png")))

    def depth(self) -> np.ndarray:
        return np.load(self.file("depth.npy"), allow_pickle=False)

    def images(self) -> list[Path]:
        """The head camera image, if there is one."""
        return [self.file("rgb.png")] if self.has_vision else []


def capture(
    link: RobotLink, files: SessionFiles, tail_intent, since: Observation | None = None, eye: dict | None = None
) -> Observation:
    """Read state, vision, state again; keep the state read just before the image.

    ``tail_intent(state)`` gives the posture, torso lean and the hands' intents (in W) at the reference tail; they
    describe the body when the robot is idle there (an intent: the palm pose a hand's requests last asked for, with
    how far the reference palm is from it). The body is given as measured and as the controller's reference at this
    moment, both in this O; ``since`` (an earlier observation) adds where this O lies in that one's. ``eye`` is the
    head mount's setting to look with (wbc.robot.eye); the camera's pose comes with the image, so a fixed camera that
    ignores it is described as it is.
    """
    attempts = 0
    while True:
        attempts += 1
        before = link.state()
        if not before.tracking:
            raise WbcError(
                ErrorCode.SESSION_STATE,
                f"cannot observe: controller is {before.phase.value}",
                cause=Cause.CONTROLLER_DOWN,
            )
        vision, vision_error = None, None
        started = time.monotonic()
        try:
            vision = link.vision(eye)
        except WbcError as error:
            vision_error = str(error)
        vision_s = time.monotonic() - started
        after = link.state() if vision is not None else before
        # The camera's pose comes from the state read before the image: if the body moved while the image was taken
        # (a simulator still settling at its start), take it again.
        if vision is None or attempts > VISION_RETRIES or _still(before, after):
            break
    kin = Kinematics()
    pose = grounded_pose(before.q, before.quat_wxyz, kin)
    to_world = SE2(float(before.cursor_pose[0]), float(before.cursor_pose[1]), heading(before.quat_wxyz))
    body = body_description(pose, kin)
    lean, turn = torso_offset(pose)
    body["torso_measured"] = {"lean_deg": round(lean), "turn_deg": round(turn), "pitch_deg": round(torso_pitch(pose))}
    reference = body_description(to_world.inverse().apply_poses(before.cursor_pose), kin)
    body["reference"] = {key: reference[key] for key in ("pelvis", "feet", "hands")}
    if before.queued_s == 0:
        posture, torso, intents = tail_intent(before)
        body["posture"] = posture.value
        body["torso"] = {"lean_deg": int(torso[0]), "turn_deg": int(torso[1])}  # as the last request set it
        for side, intent in (intents_in(intents, to_world.inverse()) or {}).items():
            body["reference"]["hands"][side]["intent"] = _intent(intent, side, reference["hands"][side])
    record = {
        "body": body,
        "since": None if since is None else frame_change(since, before.epoch, to_world),
        "pose": pose.tolist(),
        "to_world": to_world.to_list(),
        "vision_seconds": vision_s,
        "vision_attempts": attempts,
        "ticks_during_vision": int(after.tick - before.tick),
        "camera": None,
        "vision_error": vision_error,
    }
    with files.create("obs") as (path, identifier):
        write_json(path / "state.json", before.to_dict())
        if vision is not None:
            _save_vision(path, record, vision, pose, kin, eye)
        write_json(path / "observation.json", record)
    return Observation(files.get(identifier, "obs"))


def _still(before, after) -> bool:
    """Whether the body stood still between two state reads (joints and IMU attitude)."""
    if before.q is None or after.q is None:
        return True
    if np.max(np.abs(np.asarray(after.q) - np.asarray(before.q))) > STILL_JOINT_RAD:
        return False
    a, b = np.asarray(before.quat_wxyz, float), np.asarray(after.quat_wxyz, float)
    return 2 * np.arccos(min(1.0, abs(float(a @ b)))) <= STILL_ATTITUDE_RAD


def _save_vision(path, record, vision: VisionFrame, pose, kin, eye: dict | None = None) -> None:
    if vision.rgb.shape[:2] != vision.depth.shape:
        raise WbcError(ErrorCode.INTERNAL, "RGB and depth are not aligned")
    camera = camera_in_observation(kin, pose, vision.link, vision.link_to_camera, vision.intrinsics, vision.depth.shape)
    forward = camera.pose[:3, 2]  # the optical axis in O
    # The mount's setting, relative to the torso, as it was asked for (absent for a fixed camera): a real mount's
    # servos stop a tick or few off it (44.9 for 45), which the camera's pose above already holds; the model is told
    # the setting, as in simulation, and the measured angles are kept apart.
    setting = None if vision.eye is None else (eye or vision.eye)
    if setting is not None and vision.eye != setting:
        record["eye_measured"] = vision.eye
    record["eye"] = {
        **(setting or {}),
        "down_deg": round(float(np.degrees(np.arcsin(-forward[2]))), 1),  # below the horizontal
        "heading_deg": round(float(np.degrees(np.arctan2(forward[1], forward[0]))), 1),  # in O, left +
    }
    depth = vision.depth
    url = depth_refine.service()  # a real camera's depth through LingBot-Depth (unset: unchanged, as in simulation)
    if url:
        try:
            refined, provenance, record["depth_refine"] = depth_refine.refine(url, vision.rgb, vision.depth)
        except Exception as error:  # the measured depth is kept, and the observation reports a camera failure
            record["depth_refine"] = {"service": url, "error": repr(error)}
            record["vision_error"] = f"depth repair failed: {error}"
            log.warning("depth refinement failed, raw depth kept: %r", error)
        else:
            np.save(path / "depth_raw.npy", vision.depth.astype(np.float32), allow_pickle=False)
            np.save(path / "depth_provenance.npy", provenance, allow_pickle=False)
            record["raw_valid_depth_fraction"] = float(np.isfinite(vision.depth).mean())
            depth = refined
    record.update(
        camera=camera.to_dict(),
        camera_source=vision.source_id,
        camera_frame=vision.frame_id,
        valid_depth_fraction=float(np.isfinite(depth).mean()),
    )
    Image.fromarray(vision.rgb).save(path / "rgb.png")
    np.save(path / "depth.npy", depth.astype(np.float32), allow_pickle=False)


def frame_change(since: Observation, epoch: str, to_world: SE2) -> dict:
    """Where a new O (``to_world``, in the reference world of ``epoch``) lies in an earlier observation's O: its origin
    and turn there. Both come from the controller's reference and the IMU heading; a restarted controller (another
    epoch) has another reference world, and then nothing can be said."""
    if since.state.epoch != epoch:
        return {"observation_id": since.id, "epoch_changed": True}
    change = since.to_world.inverse().compose(to_world)
    return {
        "observation_id": since.id,
        "origin_xy": [round(change.x, 3) + 0.0, round(change.y, 3) + 0.0],
        "turn_deg": round(float(np.degrees(change.yaw)), 1) + 0.0,
    }


def intents_in(intents: dict | None, frame: SE2) -> dict | None:
    """Hand intents ({"position", "rotation"} per hand) mapped by ``frame`` (O to W, or its inverse)."""
    if not intents:
        return None
    turn = frame.rotation.as_matrix()
    return {
        side: {
            "position": frame.apply_points(np.asarray(i["position"], float)).tolist(),
            "rotation": (turn @ np.asarray(i["rotation"], float)).tolist(),
        }
        for side, i in intents.items()
    }


def _intent(intent: dict, side: str, reference: dict) -> dict:
    """An intent as the observation gives a palm (wbc.session.views), and how far the reference palm is from it."""
    rotation = np.asarray(intent["rotation"], float)
    normal = rotation[:, 1] * (-1 if side == "left" else 1)
    fingers = rotation[:, 0]
    planned = np.column_stack(
        [
            np.asarray(reference["finger_direction"], float),
            np.asarray(reference["normal"], float) * (-1 if side == "left" else 1),
        ]
    )
    x, y = planned[:, 0] / np.linalg.norm(planned[:, 0]), planned[:, 1] / np.linalg.norm(planned[:, 1])
    angle = Rotation.from_matrix(rotation.T @ np.column_stack([x, y, np.cross(x, y)])).magnitude()
    return {
        "palm_xyz": [round(float(v), 3) for v in intent["position"]],
        "normal": [round(float(v), 2) for v in normal],
        "finger_direction": [round(float(v), 2) for v in fingers],
        "from_reference_m": round(float(np.linalg.norm(np.asarray(reference["palm_xyz"]) - intent["position"])), 3),
        "from_reference_deg": round(float(np.degrees(angle)), 1),
    }


def load(files: SessionFiles, identifier: str) -> Observation:
    return Observation(files.get(identifier, "obs"))
