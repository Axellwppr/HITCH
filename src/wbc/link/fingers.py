"""The link's finger commands: each hand ``open`` or ``closed``.

A simulator's link server (``hand``) writes them to ``hands.json`` in the controller runtime as ``{"serial": int,
"hands": {side: state}}``; the simulator bridge polls the file (``wbc_link_server.bridges.fingers``).
"""

from __future__ import annotations

from wbc.core.layout import SIDES

FILE = "hands.json"
STATES = ("open", "closed")


def all_open() -> dict[str, str]:
    """Every hand open: the command before any is given."""
    return dict.fromkeys(SIDES, "open")
