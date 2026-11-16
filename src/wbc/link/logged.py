"""A link wrapper that journals every write and every failure."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from wbc.core.errors import WbcError
from wbc.link.protocol import LinkInfo, RobotLink, RobotState, Segment, VisionFrame


class Journal(Protocol):
    def write(self, op: str, **values) -> None: ...


class LoggedLink:
    """Journals appends, reads of info/vision, and failures (``journal.write(op, **values)``).

    ``state`` is read ten times a second: only its changes between working and failing are recorded.
    """

    def __init__(self, link: RobotLink, journal: Journal):
        self.link, self.journal = link, journal
        self._state_error = None

    def __getattr__(self, name: str) -> Any:  # anything else the wrapped link offers, unjournaled
        return getattr(self.link, name)

    def info(self) -> LinkInfo:
        return self._call("info", self.link.info)

    def vision(self, eye: dict | None = None) -> VisionFrame | None:
        return self._call("vision", lambda: self.link.vision(eye), **({"eye": eye} if eye else {}))

    def aim(self, eye: dict) -> None:
        return self._call("aim", lambda: self.link.aim(eye), eye=eye)

    def append(self, segment_id: str, epoch: str, revision: int, frames: np.ndarray) -> Segment:
        entry = {"segment_id": segment_id, "revision": revision, "frames": len(frames)}
        return self._call("append", lambda: self.link.append(segment_id, epoch, revision, frames), **entry)

    def hand(self, epoch: str, fingers: dict[str, str]) -> dict[str, str]:
        return self._call("hand", lambda: self.link.hand(epoch, fingers), fingers=fingers)

    def state(self) -> RobotState:
        try:
            result = self.link.state()
        except WbcError as error:
            if error.to_dict() != self._state_error:
                self._state_error = error.to_dict()
                self.journal.write("link.state", error=self._state_error)
            raise
        if self._state_error is not None:
            self._state_error = None
            self.journal.write("link.state", recovered=True)
        return result

    def _call(self, name: str, operation: Callable[[], Any], **entry) -> Any:
        started = time.monotonic()
        try:
            result = operation()
        except WbcError as error:
            self.journal.write(f"link.{name}", **entry, seconds=time.monotonic() - started, error=error.to_dict())
            raise
        self.journal.write(f"link.{name}", **entry, seconds=time.monotonic() - started)
        return result
