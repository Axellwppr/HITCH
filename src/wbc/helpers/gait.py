"""Height-conditioned G1 gait parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wbc.core.types import STANDING_HEIGHT

MIN_HEIGHT = 0.40
# Tallest pelvis height: the knees nearly straight (0.13 rad). Walking uses a lower range (WALK_HEIGHT).
MAX_HEIGHT = 0.79
WALK_HEIGHT = (0.58, 0.78)  # the public walker range
SEARCH_HEIGHT = (0.58, 0.77)  # a hand request's own pelvis-height search (planner.arms)
HEIGHT_TOLERANCE_M = 0.001  # a height this close to the walking range counts as inside it
# Walking pace, interpolated in pelvis height: 1 is the fast gait, 0 the conservative one.
# Stride, swing and transfer time all move together between the two.
PACE = ((0.58, 0.3), (0.70, 1.0))
CONSERVATIVE = (1.0, 0.56, 0.18)  # forward advance factor, swing s, weight transfer s
FAST = (1.8, 0.40, 0.12)
BACKWARD_RATIO = 0.7  # a backward step is this fraction of a forward one


def nominal_height(height: float) -> float:
    """The height a solve aims for: ``height``, snapped to the walking range when within ``HEIGHT_TOLERANCE_M`` of
    its boundary. A reference that ends at a boundary carries a small IK residual; snapping the next solve's goal
    keeps residuals from accumulating across repeated calls. The reference pose itself is not touched."""
    low, high = WALK_HEIGHT
    if low - HEIGHT_TOLERANCE_M <= height <= high + HEIGHT_TOLERANCE_M:
        return float(np.clip(height, low, high))
    return height


def balance_pose(height: float) -> tuple[float, float]:
    """(pelvis pitch, waist pitch) in rad of the standard posture at a pelvis height, standing and walking: the torso
    upright down to 0.62 m, then pitched forward from vertical as ``STANCE`` gives, the hips taking the first
    ``STANCE_HIP_DEG`` and the waist the rest."""
    pitch = np.radians(np.interp(height, *zip(*STANCE, strict=True)))
    pelvis = min(pitch, np.radians(STANCE_HIP_DEG))
    return float(pelvis), float(pitch - pelvis)


# The standard posture's torso pitch from vertical: (pelvis height m, degrees). Lower than 0.62 m an upright torso
# runs the ankles into their dorsiflexion limit (-50 degrees) with the feet flat; this is about the least pitch that
# keeps them 2-5 degrees inside it.
STANCE = ((0.40, 37.0), (0.45, 29.0), (0.50, 20.0), (0.55, 12.0), (0.60, 4.0), (0.62, 0.0))
STANCE_HIP_DEG = 20.0
STAND_COM = ((0.60, 0.015), (0.70, 0.025))  # (pelvis height m, CoM ahead of the feet's centre m) standing


def stand_balance(height: float) -> float:
    """How far (m) the CoM stands ahead of the feet's centre on both planted feet at a pelvis height: less than the
    walk's ``forward_balance`` when crouched, which would push the ankles past their dorsiflexion limit."""
    return float(np.interp(height, *zip(*STAND_COM, strict=True)))


@dataclass(frozen=True)
class GaitProfile:
    crouched: bool
    forward_advance: float  # m per step at this height
    lateral_advance: float
    swing_pitch: float  # rad foot pitch during swing
    support_inset: float  # m, ZMP inset toward the stance foot centre
    roll_gain: float  # pelvis roll per metre of lateral CoM offset
    forward_balance: float  # m, CoM ahead of the foot centre
    swing_s: float  # s per swing
    transfer_s: float  # s of weight transfer before each swing

    def step_count(self, delta) -> float:
        """Steps needed for a displacement [forward, left] in the walk's starting heading."""
        along = self.forward_advance * (1.0 if delta[0] >= 0 else BACKWARD_RATIO)
        if not self.crouched:
            return float(np.hypot(delta[0] / along, delta[1] / self.forward_advance))
        return max(abs(delta[0]) / along, abs(delta[1]) / self.lateral_advance)


def gait_profile(height: float) -> GaitProfile:
    height = float(np.clip(height, MIN_HEIGHT, MAX_HEIGHT))
    pace = float(np.interp(height, *zip(*PACE, strict=True)))
    stride, swing_s, transfer_s = (c + pace * (f - c) for c, f in zip(CONSERVATIVE, FAST, strict=True))
    return GaitProfile(
        crouched=height < STANDING_HEIGHT,
        forward_advance=stride
        * float(
            np.interp(
                height,
                [0.35, 0.40, 0.45, 0.50, 0.60, 0.70],
                [0.045, 0.05, 0.08, 0.09, 0.12, 0.14],
            )
        ),
        lateral_advance=float(np.interp(height, [0.35, 0.50, 0.60, 0.70], [0.075, 0.09, 0.12, 0.14])),
        swing_pitch=float(np.interp(height, [0.35, 0.50, 0.65, 0.70], [0.50, 0.35, 0.12, 0.0])),
        support_inset=float(np.interp(height, [0.40, 0.60, 0.70], [0.010, 0.012, 0.012])),
        roll_gain=float(np.interp(height, [0.35, 0.40, 0.50, 0.60], [3.0, 2.5, 1.2, 0.0])),
        # A forward CoM shift reduces forward drift at 0.50/0.60 m; at 0.40 m
        # it would exhaust ankle travel, so the deep value is 0.025.
        forward_balance=float(
            np.interp(
                height,
                [0.40, 0.45, 0.50, 0.60, 0.70],
                [0.025, 0.025, 0.045, 0.045, 0.025],
            )
        ),
        swing_s=swing_s,
        transfer_s=transfer_s,
    )
