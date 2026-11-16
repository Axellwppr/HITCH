"""An episode directory, ``data/episodes/ep-…/``: written as it happens (``Episode``: the loop) and read back
(``read``: the transcript).

- ``episode.json``: task, backend profile, session, how it ended, token totals
- ``messages.jsonl``: the conversation, one message per line
- ``requests/NNNN.json``: each request exactly as sent (images by file name) and the raw response, numbered by turn
- ``images/``: every image the model was shown
- ``transcript.md``: the same, for reading (``wbc.agent.transcript``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from wbc.agent.backends import Reply
from wbc.agent.conversation import Message
from wbc.paths import timestamped


@dataclass
class Episode:
    dir: Path
    record: dict
    messages: list[Message] = field(default_factory=list)

    @classmethod
    def create(cls, root: Path, record: dict) -> Episode:
        directory = timestamped(root / "episodes", "ep")
        (directory / "requests").mkdir()
        episode = cls(directory, record)
        episode.save()
        return episode

    def add(self, message: Message) -> None:
        self.messages.append(message)
        with (self.dir / "messages.jsonl").open("a") as stream:
            stream.write(json.dumps(message.to_record(self.dir), ensure_ascii=False) + "\n")

    def request(self, number: int, reply: Reply) -> None:
        record = {"seconds": round(reply.seconds, 2), "usage": reply.usage, "note": reply.note}
        record.update(request=reply.request, response=reply.response)
        (self.dir / "requests" / f"{number:04d}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
        totals = self.record.setdefault("usage", dict.fromkeys(reply.usage, 0))
        for key, value in reply.usage.items():
            totals[key] = totals.get(key, 0) + (value or 0)

    def save(self) -> None:
        (self.dir / "episode.json").write_text(json.dumps(self.record, indent=1, ensure_ascii=False))


@dataclass(frozen=True)
class Recorded:
    """An episode as its files hold it: the record, the messages as JSON, the requests by turn number."""

    dir: Path
    record: dict
    messages: list[dict]
    requests: dict[int, dict]


def read(directory: Path) -> Recorded:
    directory = Path(directory)
    messages = directory / "messages.jsonl"
    lines = messages.read_text(encoding="utf-8").splitlines() if messages.exists() else []
    return Recorded(
        directory,
        read_json(directory / "episode.json"),
        [json.loads(line) for line in lines if line.strip()],
        {int(path.stem): read_json(path) for path in sorted((directory / "requests").glob("[0-9]*.json"))},
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
