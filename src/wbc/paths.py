"""Filesystem locations. Data can be relocated with WBC_DATA; WBC_CHECKPOINT names another tracker bundle."""

import os
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ASSETS = PROJECT / "assets"
G1_ASSETS = ASSETS / "g1"
INITIAL_REFERENCE = ASSETS / "initial" / "start.npz"  # the start pose: qpos of shape (1, 36)
CONTROLLER = PROJECT / "third_party" / "motion_tracking" / "sim2real"  # fetched by scripts/setup
CHECKPOINT = Path(os.environ.get("WBC_CHECKPOINT", CONTROLLER / "config" / "g1" / "ckpts" / "G1_Harness"))


def data_root() -> Path:
    return Path(os.environ.get("WBC_DATA", PROJECT / "data")).expanduser().resolve()


def timestamped(parent: Path, prefix: str) -> Path:
    """A new directory ``parent/PREFIX-YYYYMMDD-HHMMSS``, with ``-2``, ``-3``, ... when one started in the same
    second (sessions, episodes)."""
    base = f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}"
    for suffix in ("", *(f"-{n}" for n in range(2, 100))):
        directory = Path(parent) / (base + suffix)
        try:
            directory.mkdir(parents=True)
            return directory
        except FileExistsError:
            continue
    raise FileExistsError(f"{parent}: {base} and 98 more exist")
