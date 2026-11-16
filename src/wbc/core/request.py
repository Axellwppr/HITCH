"""The agent's whole-body request as data (docs/REFERENCE.md 2): its fields, their ranges and the rules that need no
robot state. ``wbc.planner.request`` plans it; the agent's tools check a request with ``parse_request`` before
sending it, so these types live here, below both.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from wbc.core.errors import invalid
from wbc.core.eye import Eye
from wbc.core.layout import SIDES
from wbc.core.params import param, parse

AUTO = "auto"
# m: helpers.posture.CROUCH_FLOOR_M and helpers.gait.MAX_HEIGHT, which core cannot import
HEIGHT_RANGE = (0.30, 0.79)

PALM_PRESETS = {  # (palm normal, finger direction) in observation frame O; inward is per side
    "down": ((0, 0, -1), (1, 0, 0)),
    "up": ((0, 0, 1), (1, 0, 0)),
    "forward": ((1, 0, 0), (0, 0, 1)),
}


@dataclass(frozen=True)
class Orientation:
    palm: Literal["inward", "down", "up", "forward"] | None = param(
        None,
        doc="preset in O: inward = palms face each other (left palm toward -y, right toward +y), fingers forward; "
        "down = palm toward the ground, fingers forward; up = palm toward the sky, fingers forward; "
        "forward = palm toward +x, fingers up. Give palm alone, or normal and fingers instead",
    )
    normal: tuple[float, ...] | None = param(
        None, length=3, doc="direction the palm faces in O; with fingers, instead of palm"
    )
    fingers: tuple[float, ...] | None = param(
        None, length=3, doc="direction the fingers point in O, perpendicular to normal (within about 3 degrees)"
    )

    def __post_init__(self):
        preset = self.palm is not None and self.normal is None and self.fingers is None
        vectors = self.palm is None and self.normal is not None and self.fingers is not None
        if not (preset or vectors):
            raise ValueError("orientation needs either palm alone, or normal and fingers together")
        if vectors:  # refused with the arguments, before any planning
            normal, fingers = np.asarray(self.normal, float), np.asarray(self.fingers, float)
            if min(np.linalg.norm(normal), np.linalg.norm(fingers)) < 1e-8:
                raise ValueError("orientation vectors cannot be zero")
            if abs(normal @ fingers) / (np.linalg.norm(normal) * np.linalg.norm(fingers)) > 0.05:
                raise ValueError("palm normal and fingers must be perpendicular")

    def matrix(self, side: str) -> np.ndarray:
        if self.palm is not None:
            inward = (0, -1 if side == "left" else 1, 0)
            normal, fingers = (inward, (1, 0, 0)) if self.palm == "inward" else PALM_PRESETS[self.palm]
        else:
            normal, fingers = self.normal, self.fingers
        normal = np.asarray(normal, float)
        x = np.asarray(fingers, float)
        normal /= np.linalg.norm(normal)
        x /= np.linalg.norm(x)
        x = x - (x @ normal) * normal
        x /= np.linalg.norm(x)
        y = normal * (-1 if side == "left" else 1)  # site +y is the left palm's -normal
        return np.column_stack([x, y, np.cross(x, y)])


@dataclass(frozen=True)
class Waypoint:
    position: tuple[float, ...] = param(
        length=3, doc="palm position [x, y, z] in observation frame O, metres; z is height above ground"
    )
    duration_s: float = param(
        range=(0.4, 8.0),
        doc="time budget from the previous waypoint (or the start), seconds; "
        "hand-only plans redistribute speed smoothly and may extend an interval up to 2x, reported in the result",
    )
    orientation: Orientation | None = param(
        None,
        doc="preferred palm orientation; position takes priority, up to 60 degrees orientation error allowed; omitted continues the previous orientation target",
    )


@dataclass(frozen=True)
class Footstep:
    side: Literal["left", "right"]
    xy: tuple[float, ...] = param(length=2, doc="ankle position in O, m", frame="xy")
    yaw_deg: int | None = param(
        None,
        range=(-180, 180),
        frame="yaw_deg",
        doc="absolute foot heading in O, integer degrees; default keeps its angle to the pelvis",
    )
    root_yaw_deg: int | None = param(
        None,
        range=(-180, 180),
        frame="yaw_deg",
        doc="absolute pelvis heading in O reached during this step, integer degrees",
    )


@dataclass(frozen=True)
class Torso:
    pitch_deg: int = param(
        range=(0, 60), doc="forward lean beyond the standard posture of the pelvis height, degrees, multiples of 5"
    )
    yaw_deg: int = param(range=(-45, 45), doc="turn at the waist, degrees, multiples of 5; positive left")

    def __post_init__(self):
        if self.pitch_deg % 5 or self.yaw_deg % 5:
            raise ValueError("torso pitch_deg and yaw_deg are multiples of 5")


@dataclass(frozen=True)
class HandCommand:
    waypoints: tuple[Waypoint, ...] = param((), length=(1, 12), doc="palm waypoints in O, in order")
    home: bool = param(False, doc="return the arm to its rest posture and keep it there")
    follow: Literal["root", "world", "body"] | None = param(
        None,
        doc="how the hand moves with the body: root, world or body; with waypoints, only while the body changes, "
        "the waypoints after it (default without waypoints: rest for an arm at its rest posture, else root)",
    )
    grip: Literal["open", "closed"] | None = param(None, doc="the fingers after the in-place part")

    def __post_init__(self):
        if self.home and (self.waypoints or self.follow is not None):
            raise ValueError("home goes alone: give waypoints (with or without follow), home or follow")

    @property
    def moves(self) -> bool:
        return bool(self.waypoints) or self.home


@dataclass(frozen=True)
class Hands:
    left: HandCommand | None = None
    right: HandCommand | None = None

    def of(self, side: str) -> HandCommand:
        return getattr(self, side) or HandCommand()


@dataclass(frozen=True)
class Walk:
    """The walk of a request; the walk skill's parameters extend it (``skills.walk.WalkParams``)."""

    target: tuple[float, ...] | None = param(None, length=2, frame="xy", doc="pelvis destination [x, y] in O, m")
    turn_deg: int = param(0, range=(-180, 180), doc="heading change by the end of the walk, degrees; left +")
    footsteps: tuple[Footstep, ...] = param((), length=(0, 12), doc="explicit placements instead of target")

    def __post_init__(self):
        if (self.target is None) == (not self.footsteps):
            raise ValueError("give exactly one of target or footsteps")
        if self.footsteps and self.turn_deg:
            raise ValueError("turn_deg goes with target; with footsteps give each step's root_yaw_deg")


@dataclass(frozen=True)
class Request:
    height: float | None = param(None, range=HEIGHT_RANGE, doc='pelvis height, m, or "auto"')
    torso: Torso | None = None
    body_duration_s: float | None = param(None, range=(0.6, 8.0), doc="time of the height and torso change, s")
    hands: Hands | None = None
    walk: Walk | None = None
    wait_s: float | None = param(None, range=(0.2, 10.0), doc="stand still this long at the end, s")
    eye: Eye | None = None  # the head camera's setting for the observation after this request
    auto: bool = False  # height "auto": set by parse_request only

    @property
    def hand_commands(self) -> dict[str, HandCommand]:
        hands = self.hands or Hands()
        return {side: hands.of(side) for side in SIDES}

    @property
    def grips(self) -> dict[str, str]:
        return {side: hand.grip for side, hand in self.hand_commands.items() if hand.grip is not None}

    @property
    def in_place(self) -> bool:
        """Whether the request has an in-place part: a height or torso to reach, or a hand to move."""
        return (
            self.height is not None
            or self.auto
            or self.torso is not None
            or any(hand.moves for hand in self.hand_commands.values())
        )

    @property
    def moves(self) -> bool:
        return self.in_place or self.walk is not None or self.wait_s is not None


def parse_request(data) -> Request:
    """A request from JSON, with the rules that need no robot state (docs/REFERENCE.md 4.1)."""
    if not isinstance(data, dict):
        raise invalid("the request must be an object")
    data = dict(data)
    if "auto" in data:
        raise invalid('request: unknown key auto; give height "auto" instead')
    auto = data.get("height") == AUTO
    if auto:
        del data["height"]
    if isinstance(data.get("eye"), dict) and set(data["eye"]) != {"yaw_deg", "pitch_deg"}:
        raise invalid("eye: give both yaw_deg and pitch_deg")
    request = replace(parse(Request, data, "request"), auto=auto)
    if not request.moves and not request.grips and request.eye is None:
        follows = any(hand.follow for hand in request.hand_commands.values())
        raise invalid(
            "nothing moves: follow only says how a hand moves while the body moves; give height, torso, waypoints, "
            "home, walk or wait_s"
            if follows
            else "nothing to do: give height, torso, hands, walk, wait_s or eye"
        )
    if request.grips and request.walk is not None:
        raise invalid(
            "grip happens after the in-place part and needs the robot standing still; walk in the next execute"
        )
    if auto and request.torso is not None:
        raise invalid('height "auto" keeps the torso in its standard posture; give a height instead to lean')
    if auto and request.body_duration_s is not None:
        raise invalid('height "auto" moves the body on the waypoints\' clock; leave out body_duration_s')
    if request.body_duration_s is not None and request.height is None and request.torso is None:
        raise invalid("body_duration_s times a height or torso change; give height or torso with it")
    if auto and not any(hand.waypoints for hand in request.hand_commands.values()):
        raise invalid('height "auto" chooses a pelvis height for hand waypoints; give waypoints with it')
    return request
