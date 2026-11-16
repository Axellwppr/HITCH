"""Finger commands from the link server to a simulator bridge.

The link's ``hand`` operation (``wbc_link_server.launch.control``) writes ``hands.json`` in the controller runtime: each
hand ``open`` or ``closed``, with a serial. A bridge polls it and drives its finger actuators; the 50 Hz body
control is untouched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from wbc.core.layout import SIDES
from wbc.link.fingers import FILE, STATES, all_open

log = logging.getLogger(__name__)


class FingerCommands:
    """Polls ``runtime/hands.json``; ``changes()`` returns the sides whose command changed since the last call."""

    def __init__(self, runtime: Path):
        self.path = Path(runtime) / FILE
        self.current = all_open()
        self.serial = -1

    def changes(self) -> dict[str, str]:
        """The file is read as it stands. Raises ValueError for a file that is not a hand command: the link writes it atomically (``write_json_atomic``), so a malformed one
        is a bug, never a half-written file, and must not pass for "no change"."""
        try:
            text = self.path.read_text()
        except FileNotFoundError:  # nothing commanded yet
            return {}
        try:
            value = json.loads(text)
            serial, hands = value["serial"], value["hands"]
            if (
                not isinstance(serial, int)
                or not isinstance(hands, dict)
                or not set(hands) <= set(SIDES)
                or not set(hands.values()) <= set(STATES)
            ):
                raise ValueError("expected {serial: int, hands: {left|right: open|closed}}")
        except (ValueError, KeyError, TypeError) as error:
            log.error("malformed hand command %s: %r (%s)", self.path, text[:200], error)
            raise ValueError(f"malformed hand command {self.path}: {error}") from error
        if serial == self.serial:
            return {}
        self.serial = serial
        changed = {side: state for side, state in hands.items() if self.current.get(side) != state}
        self.current.update(changed)
        return changed
