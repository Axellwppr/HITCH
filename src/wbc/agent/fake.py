"""A scripted model behind a strict local OpenAI Responses server, for trying the loop.

It answers ``POST /v1/responses`` the way the real API treats a stateless
client, and refuses (HTTP 400) what it would refuse:

- ``store`` not false;
- an earlier reply missing from the input, or not sent back verbatim and in
  order: every output item (reasoning, message, function calls) of every reply
  must come back as it was produced;
- reasoning whose ``encrypted_content`` does not verify (it is signed);
- a function call without its ``function_call_output`` before the next input;
- images that are not PNG data URLs.

It plays one fixed episode, reading the tool results as text the way a model
would: measure the middle of the first camera image, execute a walk toward it,
measure again in the new frame, finish (never claiming success). Its usage reports a
prefix-cache estimate: the leading input identical to the previous request's.

``python -m wbc.agent.fake --port 8901`` serves it on its own.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from wbc.core import logs

IMAGE_TOKENS = 800  # rough cost of one image, for the usage estimate
PNG = b"\x89PNG\r\n\x1a\n"
POINT = re.compile(r"→ \((-?[\d.]+), (-?[\d.]+), (-?[\d.]+)\)")  # a located point, as render writes it


class FakeModel:
    def __init__(self):
        self.key = os.urandom(16)  # signs the encrypted reasoning
        self.outputs: list[list[dict]] = []  # every reply's output items, in order
        self.previous: list[str] = []  # the last request, serialised item by item

    # ------------------------------------------------------------ the API
    def answer(self, payload: dict) -> tuple[int, dict]:
        problem = self.check(payload)
        if problem:
            return 400, {"error": {"message": problem, "type": "invalid_request_error"}}
        items = payload["input"]
        text, calls, thought = self.decide(_results(items), _image_size(items))
        n = len(self.outputs)
        output = [
            {
                "id": f"rs_{n}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": thought}],
                "encrypted_content": self.seal({"reply": n, "thought": thought}),
            }
        ]
        if text:
            output.append(
                {
                    "id": f"msg_{n}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            )
        for i, (name, args) in enumerate(calls):
            output.append(
                {
                    "id": f"fc_{n}_{i}",
                    "type": "function_call",
                    "call_id": f"call_{n}_{i}",
                    "name": name,
                    "arguments": json.dumps(args),
                    "status": "completed",
                }
            )
        self.outputs.append(output)
        return 200, {
            "id": f"resp_{n}",
            "object": "response",
            "status": "completed",
            "model": payload["model"],
            "output": output,
            "usage": self.usage(payload, thought, calls),
        }

    def check(self, payload: dict) -> str | None:
        if payload.get("store") is not False:
            return "this client must not store responses: send store=false"
        items = payload.get("input")
        if not isinstance(items, list) or not isinstance(payload.get("instructions"), str):
            return "expected instructions (a string) and input (a list of items)"
        ours = {item["id"]: n for n, output in enumerate(self.outputs) for item in output}
        replayed, open_calls, i = [], set(), 0
        while i < len(items):
            item = items[i]
            if item.get("id") in ours:  # one of our replies: all of it, verbatim, in order
                n = ours[item["id"]]
                output = self.outputs[n]
                if items[i : i + len(output)] != output:
                    return f"input[{i}]: reply {n} must be sent back verbatim and complete"
                for produced in output:
                    if produced["type"] == "reasoning" and not self.opens(produced["encrypted_content"]):
                        return f"input[{i}]: encrypted reasoning does not verify"
                    if produced["type"] == "function_call":
                        open_calls.add(produced["call_id"])
                replayed.append(n)
                i += len(output)
                continue
            if item.get("type") in ("reasoning", "function_call") or item.get("role") == "assistant":
                return f"input[{i}]: an assistant item this server did not produce"
            if item.get("type") == "function_call_output":
                if item.get("call_id") not in open_calls:
                    return f"input[{i}]: no function call {item.get('call_id')!r} waiting for output"
                open_calls.discard(item["call_id"])
            elif item.get("role") == "user":
                if open_calls:
                    return f"input[{i}]: no output for function calls {sorted(open_calls)}"
            else:
                return f"input[{i}]: unexpected item {json.dumps(item)[:80]}"
            for part in _parts(item):
                if part.get("type") == "input_image" and not _png(part.get("image_url", "")):
                    return f"input[{i}]: images must be PNG data URLs"
            i += 1
        if replayed != list(range(len(self.outputs))):
            return f"replies {sorted(set(range(len(self.outputs))) - set(replayed))} are missing from the input"
        return f"no output for function calls {sorted(open_calls)}" if open_calls else None

    def seal(self, value: dict) -> str:
        body = base64.b64encode(json.dumps(value).encode()).decode()
        return body + "." + hmac.new(self.key, body.encode(), hashlib.sha256).hexdigest()

    def opens(self, sealed: str) -> bool:
        body, _, signature = sealed.partition(".")
        return hmac.compare_digest(signature, hmac.new(self.key, body.encode(), hashlib.sha256).hexdigest())

    def usage(self, payload: dict, thought: str, calls: list) -> dict:
        head = json.dumps([payload["instructions"], payload.get("tools")], sort_keys=True)
        serialised = [head] + [json.dumps(item, sort_keys=True) for item in payload["input"]]
        cost = [len(head) // 4] + [_tokens(item) for item in payload["input"]]
        same = 0
        while same < min(len(serialised), len(self.previous)) and serialised[same] == self.previous[same]:
            same += 1
        self.previous = serialised
        reasoning = len(thought) // 4
        return {
            "input_tokens": sum(cost),
            "input_tokens_details": {"cached_tokens": sum(cost[:same])},
            "output_tokens": reasoning + 20 * len(calls),
            "output_tokens_details": {"reasoning_tokens": reasoning},
            "total_tokens": sum(cost) + reasoning + 20 * len(calls),
        }

    # ---------------------------------------------------------- the script
    def decide(self, results: list[tuple[str, dict, str]], size) -> tuple[str, list, str]:
        """Reply text, tool calls and reasoning, from the tool results so far, read as a model reads them."""
        executed = [text for name, _, text in results if name == "execute"]
        if not results:
            if size is None:
                return (
                    "",
                    [("execute", {"walk": {"target": [0.3, 0.0]}})],
                    (
                        "No camera image came with the first observation, so nothing can be measured. "
                        "Walk 0.3 m forward to test the loop."
                    ),
                )
            u, v = 500, 600
            return (
                "",
                [("perceive", {"points": [[u, v]]})],
                (
                    f"The image is {size[0]}x{size[1]}. I take normalized image point ({u}, {v}) below the centre as the "
                    "thing to approach and measure it in O."
                ),
            )
        name, _, text = results[-1]
        if name == "execute" and "passed its checks" not in text:
            first = text.splitlines()[0]
            return (
                "",
                [("finish", {"success": False, "summary": f"Stopped: {first}"})],
                ("The motion was not executed; a scripted model does not repair plans."),
            )
        if name == "perceive" and not executed:
            point = POINT.search(text)
            x, y = (float(point.group(1)), float(point.group(2))) if point else (0.75, 0.0)
            target = [round(min(max(x - 0.45, 0.1), 0.6), 3), round(min(max(y, -0.3), 0.3), 3)]
            return (
                f"Walking to {target}.",
                [("execute", {"walk": {"target": target}})],
                f"The point is at x={x:.2f}, y={y:.2f} in O. Stop 0.45 m short of it, at most 0.6 m per walk.",
            )
        if name == "execute" and size is not None:
            u, v = 500, 600
            return "", [("perceive", {"points": [[u, v]]})], "Moved; measure the same place again in the new frame."
        summary = "Scripted demo: measured, walked and measured again. A scripted model does not judge the task, "
        summary += "so it does not claim success."
        return "", [("finish", {"success": False, "summary": summary})], "The demo script is complete."


# --------------------------------------------------------------- reading
def _results(items: list[dict]) -> list[tuple[str, dict, str]]:
    """(tool name, arguments, result text) for every answered function call, in order."""
    calls, out = {}, []
    for item in items:
        if item.get("type") == "function_call":
            calls[item["call_id"]] = (item["name"], json.loads(item["arguments"]))
        elif item.get("type") == "function_call_output":
            output = item["output"]
            text = output if isinstance(output, str) else next(p["text"] for p in output if p["type"] == "input_text")
            name, args = calls[item["call_id"]]
            out.append((name, args, text))
    return out


def _parts(item: dict) -> list[dict]:
    content = item.get("output") if item.get("type") == "function_call_output" else item.get("content")
    return content if isinstance(content, list) else []


def _image_size(items: list[dict]) -> tuple[int, int] | None:
    """Width and height of the first camera image sent (from its PNG header)."""
    for item in items:
        parts = _parts(item)
        for i, part in enumerate(parts):
            if part.get("type") == "input_image" and i and parts[i - 1].get("text", "").endswith("rgb.png]"):
                header = base64.b64decode(part["image_url"][22:][:40])
                return struct.unpack(">II", header[16:24])
    return None


def _png(url: str) -> bool:
    try:
        return url.startswith("data:image/png;base64,") and base64.b64decode(url[22:]).startswith(PNG)
    except ValueError:
        return False


def _tokens(item: dict) -> int:
    parts = _parts(item)
    text = json.dumps({k: v for k, v in item.items() if not isinstance(v, list)})
    text += "".join(p.get("text", "") for p in parts)
    return IMAGE_TOKENS * sum(p.get("type") == "input_image" for p in parts) + len(text) // 4


# ---------------------------------------------------------------- server
class FakeServer:
    """The fake model on a local port: ``url`` is a base URL for a Responses client."""

    def __init__(self, model: FakeModel, port: int = 0):
        fake = model

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path.rstrip("/") != "/v1/responses":
                    status, body = 404, {"error": {"message": f"no route {self.path}"}}
                else:
                    payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    status, body = fake.answer(payload)
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8901)
    port = p.parse_args().port
    logs.setup("fake-model")
    server = FakeServer(FakeModel(), port)
    print(f"fake model at {server.url}", flush=True)  # the command's result: where to point a backend
    threading.Event().wait()


if __name__ == "__main__":
    main()
