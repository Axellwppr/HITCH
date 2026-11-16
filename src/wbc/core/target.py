"""What a session connects to: ``http://HOST:PORT`` (a robot's link server) or
``sim://SCENE?seed=&heading=&paused=&vision=&viewer=&checkpoint=`` (a simulator this machine starts,
``wbc.link.local``). The client, the session and the agent's prompt read a target the same way here."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse


def options(target: str) -> dict[str, str]:
    """A target's query options, the last value of each."""
    return {key: values[-1] for key, values in parse_qs(urlparse(target).query).items()}


def scene(target: str) -> str:
    """A sim:// target's scene name ('' if it names none)."""
    url = urlparse(target)
    return url.netloc or url.path.strip("/")


def is_sim(target: str) -> bool:
    return urlparse(target).scheme == "sim"


def paused(target: str) -> bool:
    """Whether the simulator holds its time while the robot is idle (``paused=1``)."""
    return options(target).get("paused", "0") in ("1", "true")


def vision(target: str) -> bool:
    """Whether the simulator renders the head camera (on unless ``vision=0``)."""
    return options(target).get("vision", "1") not in ("0", "false")


def viewer(target: str) -> bool:
    """Whether the simulator opens an interactive window (``viewer=1``; needs a display)."""
    return options(target).get("viewer", "0") in ("1", "true")


def with_option(target: str, name: str, value) -> str:
    """``target`` with the query option ``name=value`` added."""
    return f"{target}{'&' if '?' in target else '?'}{name}={value}"
