"""In-memory controller emulation, for trying the harness without a robot or simulator.

The reference buffer, cursor, revision and segment bookkeeping behave like the
real link; tracking is perfect unless an offset is injected. Time advances
only through ``advance`` or, with ``realtime=True``, with the wall clock.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core.errors import ErrorCode, WbcError
from wbc.core.layout import FPS, JOINTS, POSE_SIZE
from wbc.link.protocol import LinkInfo, Phase, RobotState, Segment, VisionFrame


class FakeLink:
    """A ``RobotLink`` in memory. ``vision(link) -> VisionFrame | None`` provides camera frames (default: none)."""

    def __init__(
        self,
        initial_pose: np.ndarray,
        *,
        horizon: int = 9,
        realtime: bool = False,
        vision: Callable[[FakeLink], VisionFrame | None] | None = None,
    ):
        self.horizon = horizon
        self.realtime = realtime
        self.vision_source = vision
        self.eye: dict | None = None  # what the session last asked to look with (vision)
        self.aimed: dict | None = None  # where it last asked the camera to turn (aim)
        self.phase = Phase.TRACKING
        self.yaw_offset = 0.0  # IMU heading minus reference heading
        self.joint_offset = np.zeros(len(JOINTS))
        self.fail_next: WbcError | None = None  # raised by the next append, nothing applied
        self.lose_next_reply = False  # next append is applied but reported as UNREACHABLE
        self.defer_next = False  # next append waits for ``acknowledge`` (a slow or paused controller)
        self.deferred: tuple | None = None
        self.appends = 0
        self.fingers = {"left": "open", "right": "open"}
        self._reset(np.asarray(initial_pose, float))

    def _reset(self, pose: np.ndarray) -> None:
        self.epoch = uuid.uuid4().hex[:12]
        self.buffer = np.repeat(pose[None], self.horizon + 1, axis=0)
        self.cursor = float(len(self.buffer) - 1)  # idle at the end of a stationary reference, as after startup
        self.revision = 0
        self.segments: list[Segment] = []
        self._clock = time.monotonic()

    def restart(self) -> None:
        """Simulate a controller restart: new epoch, buffer reset to the current pose."""
        self._reset(self.buffer[self._index()].copy())

    def advance(self, frames: float) -> None:
        self.cursor = min(self.cursor + frames, len(self.buffer) - 1)

    def run_to_end(self) -> None:
        self.cursor = len(self.buffer) - 1

    def _index(self) -> int:
        if self.realtime:
            now = time.monotonic()
            self.advance((now - self._clock) * FPS)
            self._clock = now
        return int(self.cursor)

    def info(self) -> LinkInfo:
        return LinkInfo(self.epoch, "fake", FPS, JOINTS, self.horizon)

    def state(self) -> RobotState:
        index = self._index()
        if self.phase != Phase.TRACKING:
            return RobotState(self.epoch, self.phase)
        current = self.buffer[index]
        quat = Rotation.from_euler("z", self.yaw_offset) * Rotation.from_quat(current[3:7], scalar_first=True)
        return RobotState(
            self.epoch,
            Phase.TRACKING,
            tick=index,
            age_s=0.0,
            q=current[7:] + self.joint_offset,
            dq=np.zeros(len(JOINTS)),
            quat_wxyz=quat.as_quat(scalar_first=True),
            gyro=np.zeros(3),
            q_target=current[7:].copy(),
            cursor=index,
            tail=len(self.buffer) - 1,
            revision=self.revision,
            cursor_pose=current.copy(),
            tail_pose=self.buffer[-1].copy(),
            segments=tuple(self.segments[-32:]),
            pending=None if self.deferred is None else self.deferred[0],
        )

    def vision(self, eye: dict | None = None) -> VisionFrame | None:
        self.eye = eye  # a vision callback may read it
        return None if self.vision_source is None else self.vision_source(self)

    def aim(self, eye: dict) -> None:
        self.aimed = eye

    def hand(self, epoch: str, fingers: dict[str, str]) -> dict[str, str]:
        if epoch != self.epoch:
            raise WbcError(ErrorCode.EPOCH_CHANGED, "controller restarted; reconnect")
        self.fingers.update(fingers)
        return dict(self.fingers)

    def append(self, segment_id: str, epoch: str, revision: int, frames: np.ndarray) -> Segment:
        existing = next((s for s in self.segments if s.segment_id == segment_id), None)
        if existing is not None:
            return existing
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error
        if epoch != self.epoch:
            raise WbcError(ErrorCode.EPOCH_CHANGED, "controller restarted")
        if self.phase != Phase.TRACKING:
            raise WbcError(ErrorCode.NOT_TRACKING, "controller is not tracking")
        if revision != self.revision:
            raise WbcError(ErrorCode.STALE_TAIL, f"revision {revision} != {self.revision}")
        frames = np.asarray(frames, float)
        if frames.ndim != 2 or frames.shape[1] != POSE_SIZE or len(frames) == 0:
            raise WbcError(ErrorCode.INVALID_SEGMENT, "expected [N, 36] frames")
        if self.deferred is not None:
            raise WbcError(ErrorCode.BUSY, "previous append not yet acknowledged")
        if self.defer_next:
            self.defer_next, self.deferred = False, (segment_id, frames)
            raise WbcError(ErrorCode.PENDING, "not acknowledged yet; the segment id shows up in state")
        return self._apply(segment_id, frames)

    def acknowledge(self) -> None:
        """The controller finally takes the deferred append."""
        (segment_id, frames), self.deferred = self.deferred, None
        self._apply(segment_id, frames)

    def _apply(self, segment_id: str, frames: np.ndarray) -> Segment:
        index = self._index()
        guard = max(0, self.horizon + 1 - (len(self.buffer) - 1 - index))
        start = len(self.buffer) + guard
        self.buffer = np.vstack([self.buffer, np.repeat(self.buffer[-1:], guard, axis=0), frames])
        self.revision += 1
        self.appends += 1
        segment = Segment(segment_id, start, len(self.buffer) - 1)
        self.segments.append(segment)
        if self.lose_next_reply:
            self.lose_next_reply = False
            raise WbcError(ErrorCode.UNREACHABLE, "reply lost")
        return segment
