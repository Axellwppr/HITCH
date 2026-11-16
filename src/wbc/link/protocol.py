"""The robot link: six operations between the harness and a controller.

``info`` (static); ``state`` and ``vision`` (reads; ``vision`` may first set the head camera's mount); ``aim`` (start
turning the head camera, returning at once); ``append`` (the body write); ``hand`` (finger commands: each hand open
or closed, applied at once). Appended frames are W-frame canonical poses that continue the reference tail; the link
adds nothing and checks nothing beyond what the controller channel itself requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from wbc.core.layout import FPS
from wbc.core.wire import pack, unpack


class Phase(str, Enum):
    TRACKING = "tracking"
    NOT_TRACKING = "not_tracking"
    FAULT = "fault"


@dataclass(frozen=True)
class LinkInfo:
    epoch: str
    backend: str  # "sim" | "real" | "fake"
    fps: int
    joints: tuple[str, ...]
    horizon: int  # frames the policy looks ahead; the controller holds the tail that long before an append
    paused: bool = False  # simulator holds time while the reference is idle

    def to_dict(self) -> dict:
        return {**self.__dict__, "joints": list(self.joints)}

    @classmethod
    def from_dict(cls, value: dict) -> LinkInfo:
        return cls(**{**value, "joints": tuple(value["joints"])})


@dataclass(frozen=True)
class Segment:
    segment_id: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RobotState:
    epoch: str
    phase: Phase
    tick: int = 0
    age_s: float | None = None  # seconds since the last robot state; None when not tracking
    # Measured (present only while tracking).
    q: np.ndarray | None = None  # [29] canonical
    dq: np.ndarray | None = None
    quat_wxyz: np.ndarray | None = None
    gyro: np.ndarray | None = None
    q_target: np.ndarray | None = None  # PD target actually sent
    # Reference buffer.
    cursor: int = 0  # frame being tracked
    tail: int = 0  # last frame index
    revision: int = 0  # number of accepted appends; append must quote it
    cursor_pose: np.ndarray | None = None  # [36] pose at the cursor, W frame
    tail_pose: np.ndarray | None = None  # [36] pose at the tail, W frame
    segments: tuple[Segment, ...] = ()
    pending: str | None = None  # segment id handed to the controller but not yet acknowledged

    @property
    def tracking(self) -> bool:
        return self.phase == Phase.TRACKING

    @property
    def queued_s(self) -> float:
        """Reference time still ahead of the cursor; 0 when idle."""
        return (self.tail - self.cursor) / FPS

    def segment(self, segment_id: str) -> Segment | None:
        return next((s for s in self.segments if s.segment_id == segment_id), None)

    def to_dict(self) -> dict:
        out = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
            elif key == "segments":
                out[key] = [s.to_dict() for s in value]
            elif isinstance(value, Phase):
                out[key] = value.value
            else:
                out[key] = value
        return out

    @classmethod
    def from_dict(cls, value: dict) -> RobotState:
        arrays = {"q", "dq", "quat_wxyz", "gyro", "q_target", "cursor_pose", "tail_pose"}
        kwargs = {}
        for key, item in value.items():
            if key in arrays:
                kwargs[key] = None if item is None else np.asarray(item, float)
            elif key == "segments":
                kwargs[key] = tuple(Segment(**s) for s in item)
            elif key == "phase":
                kwargs[key] = Phase(item)
            else:
                kwargs[key] = item
        return cls(**kwargs)


@dataclass(frozen=True)
class VisionFrame:
    rgb: np.ndarray  # [H, W, 3] uint8
    depth: np.ndarray  # [H, W] float32 metres, NaN when missing
    intrinsics: dict  # fx, fy, cx, cy; optional distortion: {model, coeffs}
    link: str  # robot link the camera is mounted on
    link_to_camera: np.ndarray  # 4x4, optical frame (x right, y down, z forward)
    source_id: str = ""
    frame_id: int = 0
    eye: dict | None = None  # the mount's {yaw_deg, pitch_deg} the image was taken at; None: a fixed camera

    def to_wire(self) -> dict:
        return {
            "rgb": pack(self.rgb),
            "depth": pack(self.depth),
            "intrinsics": self.intrinsics,
            "link": self.link,
            "link_to_camera": np.asarray(self.link_to_camera).tolist(),
            "source_id": self.source_id,
            "frame_id": self.frame_id,
            "eye": self.eye,
        }

    @classmethod
    def from_wire(cls, value: dict) -> VisionFrame:
        return cls(
            unpack(value["rgb"]),
            unpack(value["depth"]),
            value["intrinsics"],
            value["link"],
            np.asarray(value["link_to_camera"], float),
            value["source_id"],
            value["frame_id"],
            value.get("eye"),
        )


class RobotLink(Protocol):
    def info(self) -> LinkInfo: ...

    def state(self) -> RobotState: ...

    def vision(self, eye: dict | None = None) -> VisionFrame | None: ...  # eye: {yaw_deg, pitch_deg} to look with

    def aim(self, eye: dict) -> None: ...  # start turning the camera toward eye and return at once

    def append(self, segment_id: str, epoch: str, revision: int, frames: np.ndarray) -> Segment: ...

    def hand(self, epoch: str, fingers: dict[str, str]) -> dict[str, str]: ...  # side -> "open"/"closed"; all sides
