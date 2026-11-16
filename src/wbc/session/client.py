"""Client of a session host: start one, then talk to it over HTTP only.

A session is its URL. ``connect`` starts a host and returns the URL; every
other call goes to the URL given by the caller (``WBC_SESSION`` for the CLI).
Nothing here reads the session directory.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

from wbc.core.errors import ErrorCode, WbcError
from wbc.core.jsonhttp import fetch, request

STARTUP_S = 180.0
CALL_S = 900.0  # an operation that changes something (planning, a grip's stand-still) may take this long
READS = ("status", "wait", "show", "log")  # safe to repeat while a crashed host restarts
RESTART_GRACE_S = 15.0


def session_url(value: str | None = None) -> str:
    url = value or os.environ.get("WBC_SESSION")
    if not url:
        raise WbcError(
            ErrorCode.INVALID_REQUEST, "no session: set WBC_SESSION to the URL printed by `scripts/wbc connect`"
        )
    return url.rstrip("/")


def connect(
    target: str,
    *,
    vision: str | None = None,
    port: int = 0,
    root: Path | None = None,
    timeout: float = STARTUP_S,
) -> dict:
    """Start a session host for ``target``; returns ``{"session": URL, "session_id": ...}``."""
    command = [sys.executable, "-m", "wbc.session.host", "--connect", target, "--port", str(port)]
    command += ["--vision", vision] if vision else []
    command += ["--root", str(root)] if root else []
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # outlives the command that started it
    )
    try:
        ready, _, _ = select.select([process.stdout], [], [], timeout)
        if not ready:
            raise WbcError(ErrorCode.UNREACHABLE, f"session host did not start within {timeout:.0f} s")
        line = process.stdout.readline()
    except BaseException:  # timed out or interrupted (Ctrl-C): the host stops, with the simulator it started
        process.terminate()
        raise
    process.stdout.close()
    try:
        reply = json.loads(line)
    except ValueError:
        raise WbcError(ErrorCode.INTERNAL, "session host exited without a reply") from None
    if "error" in reply:
        raise WbcError.from_dict(reply["error"])
    return reply


def call(url: str, op: str, **args):
    timeout = CALL_S if op not in READS else 30.0 + float(args.get("timeout", 30.0) if op == "wait" else 0)
    deadline = time.monotonic() + RESTART_GRACE_S
    while True:
        try:
            return request(url + "/call", {"op": op, "args": args}, timeout=timeout)
        except WbcError as error:
            if error.code != ErrorCode.UNREACHABLE or op not in READS or time.monotonic() > deadline:
                raise
            time.sleep(0.5)  # the host may be restarting on the same port


def file(url: str, path: str) -> bytes:
    """A file of the session at ``url`` (paths as results give them, relative to the session)."""
    return fetch(f"{url}/files/{quote(path)}")
