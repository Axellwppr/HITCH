"""Child processes: dying with the parent, signalling a process group, waiting for the file a child writes when it
is ready."""

from __future__ import annotations

import ctypes
import os
import signal
import time
from pathlib import Path

PR_SET_PDEATHSIG = 1


def die_with_parent() -> None:
    """For ``Popen(preexec_fn=...)``: the child gets SIGTERM when its parent dies."""
    ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)


def signal_group(pid: int, sig: int) -> bool:
    """``sig`` to the process group ``pid`` leads (signal 0 only asks); whether the group still exists."""
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return False
    return True


def wait_for_file(path: Path, timeout: float, check=None, poll: float = 0.05) -> bool:
    """Wait until ``path`` exists, calling ``check`` (which raises to give up) every ``poll`` s; False after
    ``timeout`` s without it."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        if check is not None:
            check()
        if time.monotonic() > deadline:
            return False
        time.sleep(poll)
    return True
