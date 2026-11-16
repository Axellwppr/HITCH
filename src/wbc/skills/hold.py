"""Stand still for a time: the wait of a request (``wait_s``)."""

from __future__ import annotations

from dataclasses import dataclass

from wbc.core.params import param
from wbc.core.types import MotionDraft, StartState, phases_from_labels
from wbc.helpers.interp import frames_for
from wbc.helpers.interp import hold as repeat


@dataclass(frozen=True)
class HoldParams:
    duration_s: float = param(0.2, range=(0.02, 30.0))


def hold(start: StartState, p: HoldParams) -> MotionDraft:
    """Keep the current pose for a while."""
    frames = repeat(start.pose, frames_for(p.duration_s) + 1)
    labels = ["start"] + ["hold"] * (len(frames) - 1)
    return MotionDraft(
        frames,
        phases_from_labels(labels),
        start.posture,
        support=None,
        diagnostics={"ik_residual": 0.0},
    )
