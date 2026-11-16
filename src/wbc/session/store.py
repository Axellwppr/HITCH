"""One session directory: numbered artifacts, run records and an append-only journal.

Only the session's host process writes here, so nothing is locked. Artifacts
are directories that appear atomically (staged, then renamed); ids are
numbered within the session (``obs-3``, ``plan-2``, ``run-1``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from wbc.core.errors import ErrorCode, WbcError
from wbc.core.jsonhttp import plain_json, write_json_atomic

KINDS = {"obs": "observations", "plan": "plans", "run": "runs"}
ID_PATTERN = re.compile(r"^(?P<kind>obs|plan|run)-(?P<number>[1-9][0-9]*)$")


def kind_of(identifier: str) -> str:
    match = ID_PATTERN.fullmatch(identifier or "")
    if not match:
        raise WbcError(ErrorCode.INVALID_REQUEST, f"not an id: {identifier!r} (expected obs-N, plan-N or run-N)")
    return match["kind"]


def write_json(path, value) -> None:
    """A session record, indented, written atomically (``core.jsonhttp.write_json_atomic``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, value, end="\n", indent=2, ensure_ascii=False, allow_nan=False, default=plain_json)


def read_json(path):
    return json.loads(Path(path).read_text())


class Journal:
    """Append-only JSON lines; safe to write from the command and monitor threads."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, op: str, **values) -> None:
        line = json.dumps({"time": time.time(), "op": op, **values}, ensure_ascii=False, default=plain_json)
        with self._lock, self.path.open("a") as stream:
            stream.write(line + "\n")

    def tail(self, count: int) -> list[dict]:
        """The last ``count`` entries, read from the end of the file."""
        if count <= 0 or not self.path.exists():
            return []
        with self.path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = block = stream.tell()
            data = b""
            while size > 0 and data.count(b"\n") <= count:
                block = min(65536, size)
                size -= block
                stream.seek(size)
                data = stream.read(block) + data
        lines = [line for line in data.splitlines() if line.strip()][-count:]
        return [json.loads(line) for line in lines]


class SessionFiles:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self._last = {kind: self._highest(kind) for kind in KINDS}  # newest that exists
        self._next = {kind: number + 1 for kind, number in self._last.items()}

    def _highest(self, kind: str) -> int:
        folder = self.dir / KINDS[kind]
        numbers = [
            int(match["number"])
            for entry in (folder.iterdir() if folder.exists() else ())
            if (match := ID_PATTERN.fullmatch(entry.name.removesuffix(".json"))) and match["kind"] == kind
        ]
        return max(numbers, default=0)

    def new_id(self, kind: str) -> str:
        number = self._next[kind]
        self._next[kind] += 1
        return f"{kind}-{number}"

    def latest(self, kind: str) -> str | None:
        number = self._last[kind]
        return f"{kind}-{number}" if number else None

    def saved(self, identifier: str) -> None:
        """Note that ``identifier`` now exists (runs are files written by their owner)."""
        kind, number = kind_of(identifier), int(identifier.rsplit("-", 1)[1])
        self._last[kind] = max(self._last[kind], number)

    def path(self, identifier: str) -> Path:
        kind = kind_of(identifier)
        name = f"{identifier}.json" if kind == "run" else identifier
        return self.dir / KINDS[kind] / name

    def get(self, identifier: str, kind: str | None = None) -> Path:
        if kind is not None and kind_of(identifier) != kind:
            raise WbcError(ErrorCode.INVALID_REQUEST, f"{identifier} is not a {KINDS[kind][:-1]}")
        path = self.path(identifier)
        if not path.exists():
            raise WbcError(ErrorCode.NOT_FOUND, f"no {identifier} in this session")
        return path

    @contextmanager
    def create(self, kind: str):
        """Yield (staging directory, id); the artifact directory appears atomically on success."""
        identifier = self.new_id(kind)
        staging = self.dir / ".staging" / identifier
        if staging.exists():  # left by a host that was killed mid-write; ids are ours alone
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            yield staging, identifier
            destination = self.path(identifier)
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(destination)
            self.saved(identifier)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def describe(self, identifier: str) -> dict:
        path = self.get(identifier)
        if path.is_file():
            return {"id": identifier, "file": str(path), **read_json(path)}
        files = sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())
        main = {"obs": "observation.json", "plan": "plan.json"}[kind_of(identifier)]
        return {"id": identifier, "directory": str(path), "files": files, **read_json(path / main)}
