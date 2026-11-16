"""HTTP client for a robot link server (wbc_link_server.launch.control) and a vision service."""

from __future__ import annotations

import numpy as np

from wbc.core.jsonhttp import request
from wbc.core.wire import pack
from wbc.link.protocol import LinkInfo, RobotState, Segment, VisionFrame

AIM_TIMEOUT_S = 2.0  # an aim only writes the servos' goals


class HttpLink:
    def __init__(self, control: str, vision: str | None = None, timeout: float = 5.0):
        self.control = control.rstrip("/")
        self.vision_url = vision.rstrip("/") if vision else None
        self.timeout = timeout

    def info(self) -> LinkInfo:
        return LinkInfo.from_dict(request(self.control + "/info", timeout=self.timeout))

    def state(self) -> RobotState:
        return RobotState.from_dict(request(self.control + "/state", timeout=self.timeout))

    def vision(self, eye: dict | None = None) -> VisionFrame | None:
        if self.vision_url is None:
            return None
        payload = None if eye is None else {"eye": eye}  # a POST asks for the view; a fixed camera ignores it
        return VisionFrame.from_wire(request(self.vision_url + "/vision", payload, timeout=max(self.timeout, 10.0)))

    def aim(self, eye: dict) -> None:
        """Send the head toward ``eye`` and return at once (a real mount turns while the body moves)."""
        if self.vision_url is not None:
            request(self.vision_url + "/aim", {"eye": eye}, timeout=AIM_TIMEOUT_S)

    def append(self, segment_id: str, epoch: str, revision: int, frames: np.ndarray) -> Segment:
        payload = {
            "segment_id": segment_id,
            "epoch": epoch,
            "revision": int(revision),
            "frames": pack(np.asarray(frames, np.float64)),
        }
        return Segment(**request(self.control + "/append", payload, timeout=self.timeout))

    def hand(self, epoch: str, fingers: dict[str, str]) -> dict[str, str]:
        return request(self.control + "/hand", {"epoch": epoch, "fingers": fingers}, timeout=self.timeout)
