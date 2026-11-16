"""Diagnostics of every process, one format: ``setup(process, episode)`` once at an entry point, and in each module
``log = logging.getLogger(__name__)`` (a module also run as ``python -m`` spells its dotted name out: there
``__name__`` is ``__main__``).

A line on stderr::

    2026-01-01 14:03:12.345 INFO bridge[81234] ses-20260101-140310 wbc_link_server.bridges.fingers: message

time, level, the process's name and pid, the episode or session it serves (when it knows one: ``session_of`` finds
the session of a process's directory) and the logger. Our packages log from INFO up, other libraries from WARNING up
(httpx and the like would fill the logs at INFO). Only diagnostics go here: what a process prints for a caller
to read (JSON lines, a command's result, ``--help``) stays on stdout, and nothing a model sees or a record stores is
logged instead of written.

Standard library only, so a process can set it up before it imports anything else (another library's handler on the
root logger is left alone).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

OURS = ("wbc", "wbc_link_server", "__main__")
FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s {process}[%(process)d]{episode} %(name)s: %(message)s"
DATE = "%Y-%m-%d %H:%M:%S"


class Handler(logging.StreamHandler):
    """The one handler ``setup`` installs on the root logger (a second ``setup`` replaces it)."""


def ours(record: logging.LogRecord) -> bool:
    """Our packages' records at the handler's level; any other library's from WARNING up."""
    return record.levelno >= logging.WARNING or record.name.split(".")[0] in OURS


def setup(process: str, episode: str | None = None, *, level: int = logging.INFO, stream=None) -> logging.Handler:
    """Send this process's log records to ``stream`` (default: ``sys.stderr`` as it is now) as one line each, named
    ``process`` and, if given, ``episode``. Idempotent: called again (a process that learns its session later), it
    replaces the handler it installed before, so a record is never written twice; other handlers are left alone."""
    root = logging.getLogger()
    for old in [h for h in root.handlers if isinstance(h, Handler)]:
        root.removeHandler(old)
        old.close()
    handler = Handler(sys.stderr if stream is None else stream)
    tag = f" {episode}" if episode else ""
    handler.setFormatter(logging.Formatter(FORMAT.format(process=_literal(process), episode=_literal(tag)), DATE))
    handler.setLevel(level)
    handler.addFilter(ours)
    root.addHandler(handler)
    if root.getEffectiveLevel() > level:  # never raised: a library may have set DEBUG for its own handler
        root.setLevel(level)
    return handler


def session_of(path) -> str | None:
    """The session a process's directory is in: the nearest ``ses-*`` directory on its path (the session host's
    directory, the stack's ``runtime/``, ``runtime/sim``, ``runtime/controller``); None outside a session."""
    if path is None:
        return None
    path = Path(path)
    return next((p.name for p in (path, *path.parents) if p.name.startswith("ses-")), None)


def _literal(text: str) -> str:
    return text.replace("%", "%%")
