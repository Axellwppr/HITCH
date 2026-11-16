"""The head camera's setting on its two-axis mount: yaw (left positive) and pitch (down positive), both degrees and
relative to the torso. The geometry of the mount is in ``wbc.robot.eye``.

The model may use only the ``STEP_DEG`` grid inside ``YAW_RANGE`` and ``PITCH_RANGE`` (narrower than the mount's ±48°
and 0-63°).
"""

from __future__ import annotations

from dataclasses import dataclass

from wbc.core.params import param

YAW_RANGE = (-45, 45)
PITCH_RANGE = (0, 60)
STEP_DEG = 15
DEFAULT = (0, 45)  # yaw, pitch down


@dataclass(frozen=True)
class Eye:
    yaw_deg: int = param(DEFAULT[0], range=YAW_RANGE, doc=f"camera turn left of the torso, multiples of {STEP_DEG}")
    pitch_deg: int = param(
        DEFAULT[1], range=PITCH_RANGE, doc=f"camera tilt down from the torso, multiples of {STEP_DEG}"
    )

    def __post_init__(self):
        if self.yaw_deg % STEP_DEG or self.pitch_deg % STEP_DEG:
            raise ValueError(f"eye yaw_deg and pitch_deg are multiples of {STEP_DEG}")

    def to_json(self) -> dict:
        return {"yaw_deg": int(self.yaw_deg), "pitch_deg": int(self.pitch_deg)}

    @classmethod
    def from_json(cls, value: dict | None) -> Eye:
        return cls() if value is None else cls(int(value["yaw_deg"]), int(value["pitch_deg"]))
