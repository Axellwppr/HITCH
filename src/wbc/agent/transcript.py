"""``transcript.md``: an episode as a person reads it, from its records."""

from __future__ import annotations

import json
from pathlib import Path

from wbc.agent.episode import read


def write(directory: Path) -> Path:
    """Write ``directory``'s transcript.md, from the episode it holds."""
    episode = read(directory)
    record = episode.record
    out = [f"# {episode.dir.name}: {record['task']}", "", *_summary(record), ""]
    turn = 0
    for message in episode.messages:
        role = message["role"]
        if role == "system":
            words = len(message["text"].split())
            out += ["## System prompt", "", f"<details><summary>{words} words</summary>", "", "````markdown"]
            out += [message["text"], "````", "", "</details>", ""]
        elif role == "user":
            out += ["## User", "", *_quote(message.get("text", "")), *_images(message), ""]
        elif role == "assistant":
            turn += 1
            out += [f"## Turn {turn} · model", "", *_usage(episode.requests.get(turn))]
            sealed = sum(1 for item in message.get("raw") or [] if item.get("encrypted_content"))
            if message.get("reasoning_text") or sealed:
                kept = f" ({sealed} encrypted, sent back with the conversation)" if sealed else ""
                out += [f"**Reasoning**{kept}", "", *_quote(message.get("reasoning_text", "")), ""]
            if message.get("text"):
                out += [message["text"], ""]
            for call in message.get("tool_calls", []):
                out += [f"**→ `{call['name']}`** `{call['id']}`", "", "```json", _pretty(call["arguments"]), "```", ""]
        elif role == "tool":
            out += [f"**← result** `{message.get('tool_call_id')}`", "", *_block(message["text"])]
            out += [*_images(message), ""]
    path = episode.dir / "transcript.md"
    path.write_text("\n".join(out).rstrip() + "\n")
    return path


def _summary(record: dict) -> list[str]:
    stop = record.get("stop", {})
    backend = record.get("backend", {})
    lines = [
        f"- target: `{record['target']}`, session `{record.get('session')}`",
        f"- backend: `{backend.get('name')}` ({backend.get('api')} API), model `{backend.get('model')}`",
        f"- ended: {stop.get('reason')} after {stop.get('turns', '?')} turns"
        + (f", success {stop['success']}" if "success" in stop else ""),
    ]
    if stop.get("summary") or stop.get("detail"):
        lines.append(f"- {stop.get('summary') or stop.get('detail')}")
    check = record.get("check")
    if check:  # the scene's own success check, after the simulator stopped
        verdict = f"success {check['success']}" if "success" in check else f"failed to run ({check.get('error')})"
        lines.append(f"- scene check: {verdict}")
    usage = record.get("usage")
    if usage:
        lines.append(
            f"- tokens: prompt {usage.get('prompt_tokens')} (cached {usage.get('cached_tokens')}), "
            f"completion {usage.get('completion_tokens')} (reasoning {usage.get('reasoning_tokens', 0)})"
        )
    return lines


def _usage(request: dict | None) -> list[str]:
    if request is None:
        return []
    usage = request.get("usage") or (request.get("response") or {}).get("usage") or {}
    note = f" · **{request['note']}**" if request.get("note") else ""
    return [
        f"<sub>prompt {usage.get('prompt_tokens')} tokens (cached {usage.get('cached_tokens')}), completion "
        f"{usage.get('completion_tokens')} (reasoning {usage.get('reasoning_tokens')}), {request['seconds']} s</sub>"
        + note,
        "",
    ]


def _images(message: dict) -> list[str]:
    return [f"![{i['label']}]({i['path']})" for i in message.get("images", [])]


def _quote(text: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in text.splitlines()]


def _block(text: str) -> list[str]:
    """A tool result, as written."""
    return ["```text", text, "```"]


def _pretty(text: str) -> str:
    """Indented JSON, with lists of plain values kept on one line."""
    try:
        value = json.loads(text)
    except ValueError:
        return text
    return _dump(value, 0)


def _dump(value, depth: int) -> str:
    pad = " " * (depth + 1)
    if isinstance(value, dict) and value:
        items = [f"{pad}{json.dumps(k)}: {_dump(v, depth + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * depth + "}"
    if isinstance(value, list) and any(isinstance(v, (dict, list)) and v for v in value):
        if all(isinstance(v, list) and not any(isinstance(x, (dict, list)) for x in v) for v in value):
            return json.dumps(value, ensure_ascii=False)  # e.g. a list of pixels
        return "[\n" + ",\n".join(pad + _dump(v, depth + 1) for v in value) + "\n" + " " * depth + "]"
    return json.dumps(value, ensure_ascii=False)
