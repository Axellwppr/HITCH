"""Depth repair for the real camera: the LingBot-Depth service (scripts/lingbot).

Set ``WBC_DEPTH_REFINE`` to the service URL (e.g. http://127.0.0.1:18890) for the session's process; every
observation's depth then goes through it before it is saved and measured (``WBC_DEPTH_REFINE_MODE``: ``full``, the
default, the model's depth wherever it gives one; ``fill``, measured depth kept and holes filled). A failed call keeps
the measured depth and reports the observation's camera as failed. Unset, nothing changes, which is how simulation
runs.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import numpy as np

from wbc.core.wire import pack, unpack

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # direct or SSH-forwarded, never a proxy


def service() -> str | None:
    return os.environ.get("WBC_DEPTH_REFINE") or None


def refine(url: str, rgb: np.ndarray, depth: np.ndarray, timeout: float = 30.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """(depth in metres with NaN where none, per-pixel provenance, a record of the call) from the service."""
    mode = os.environ.get("WBC_DEPTH_REFINE_MODE", "full")
    body = json.dumps(
        {"rgb": pack(np.asarray(rgb, np.uint8)), "depth": pack(np.asarray(depth, np.float32)), "mode": mode}
    ).encode()
    request = urllib.request.Request(url.rstrip("/") + "/refine", body, {"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            reply = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"depth service: {error.read().decode(errors='replace')[:300]}") from None
    except OSError as error:  # URLError, a refused connection, a timeout
        raise RuntimeError(f"depth service at {url} unreachable: {getattr(error, 'reason', error)}") from None
    record = {
        "service": url,
        "mode": reply["mode"],
        "model": reply["model"],
        "infer_ms": round(reply["infer_ms"], 1),
        "seconds": round(time.monotonic() - started, 3),
    }
    return unpack(reply["depth"]).astype(np.float32), unpack(reply["provenance"]), record
