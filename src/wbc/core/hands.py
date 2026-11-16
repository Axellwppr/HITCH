"""The robot's hands as the agent is told about them, and which hand this process has (``WBC_HAND``).

The hand models (meshes, contact shapes, palm points, finger poses) live in ``wbc.robot.hands`` (``HANDS``); this
module holds only what the agent layer may read, for the same names: ``PROMPTS`` has every hand of ``HANDS``, and
``GRIPPING`` those with a ``grip``.
"""

from __future__ import annotations

import os

DEFAULT_HAND = "rubber"
RUBBER = (
    "Both wrists carry fixed rubber hands with curled fingers; they have no finger control. A palm position is the "
    "centre of the palm face. The curled fingertips reach about 3.5 cm out from the palm face, toward the side it "
    "faces, 7-9 cm ahead of the palm position along the fingers: holding an object between the palms, they are what "
    "presses it."
)
DEX3 = (
    "Both wrists carry Unitree Dex3-1 hands, held open unless a hand command closes them (grip). A palm position is "
    "the centre of the palm face. Two parallel fingers, one 3 cm to each side of the palm position across the hand "
    "(above and below it when the palms face each other), reach straight from 2 cm to 13 cm ahead of it along the "
    "fingers. The thumb stands straight out of the palm face, 10 cm long, 1-4 cm behind the palm position. Closing "
    "curls the two fingers back toward the palm and swings the thumb toward them, into a fist around whatever lies in "
    "front of the palm face between the thumb and the fingers. To hold an object between the palms, keep the thumbs in "
    "front of its near face, so the two fingers of each hand lie along its sides."
)
PROMPTS = {"rubber": RUBBER, "dex3": DEX3}
GRIPPING = frozenset({"dex3"})  # hands whose fingers a hand command can close (grip)
# The stand-still after a finger command, s: Dex3 fingers close in about 0.5 s, open thumb first in
# 0.7 s. The session holds still at least this long; the agent's execute adds the request's wait_s to it.
GRIP_S = 1.0


def hand_name() -> str:
    """This process's hand: ``WBC_HAND``, else the rubber hand."""
    name = os.environ.get("WBC_HAND", DEFAULT_HAND)
    if name not in PROMPTS:
        raise ValueError(f"WBC_HAND={name!r}: known hands are {sorted(PROMPTS)}")
    return name


def hand_prompt(name: str | None = None) -> str:
    """The system prompt's description of the hands (this process's, unless named)."""
    return PROMPTS[name or hand_name()]


def hand_grips(name: str | None = None) -> bool:
    """Whether the hand (this process's, unless named) has finger control: the hands skill's grip."""
    return (name or hand_name()) in GRIPPING
