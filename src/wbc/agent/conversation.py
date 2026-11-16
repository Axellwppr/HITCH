"""The conversation as the loop keeps it: provider-neutral, append-only.

Backends render it into their wire format on every request; nothing here
knows about JSON payloads. Images are kept as files of the episode (the exact
bytes sent). Text the loop writes (tool results, the first observation) keeps
the session results it was rendered from in ``data``, so an episode can be
rendered again. A model's reply keeps, beside the parts the loop reads, the
backend's own record of it (``raw``: e.g. the Responses API output items with
their encrypted reasoning), which the backend sends back unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Image:
    path: Path  # the file in the episode directory
    label: str  # where it came from, e.g. "observations/obs-1/rgb.png"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # exactly as the model wrote it; parsed by the tools


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    text: str = ""
    images: tuple[Image, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None  # tool messages: the call they answer
    raw: object = None  # assistant messages: the reply as the backend received it, sent back as is
    reasoning_text: str = ""  # assistant messages: readable reasoning (e.g. its summary), for transcripts
    data: object = None  # tool / user messages: the session results behind the text; recorded, never sent

    def to_record(self, root: Path) -> dict:
        """A JSON-ready record, image paths relative to ``root``."""
        record = {k: v for k, v in asdict(self).items() if v not in (None, "", (), [])}
        if self.images:
            record["images"] = [{"path": str(i.path.relative_to(root)), "label": i.label} for i in self.images]
        return record
