"""Data passed between planner, skills and session."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from wbc.core.layout import DT, POSE_SIZE
from wbc.core.limits import LimitProfile


class Posture(str, Enum):
    STANDING = "standing"
    LOW = "low"


class RunState(str, Enum):
    """Where one appended plan is (``wbc.session.runs``); results give the value."""

    SUBMITTING = "submitting"
    UNKNOWN = "unknown"  # the append's outcome is not known yet
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    REFUSED = "refused"  # nothing was appended
    LOST = "lost"  # the controller restarted or faulted before the run finished


RUN_ENDED = frozenset({RunState.DONE, RunState.REFUSED, RunState.LOST})  # a run in these states changes no more


STANDING_HEIGHT = 0.70  # pelvis height separating STANDING from LOW


def posture_for_height(height: float) -> Posture:
    return Posture.STANDING if height >= STANDING_HEIGHT - 1e-6 else Posture.LOW


@dataclass(frozen=True)
class StartState:
    """The state a skill starts from (the planner hands skills their own step frame: origin, heading 0)."""

    pose: np.ndarray
    posture: Posture
    torso: tuple[int, int] = (0, 0)  # lean and turn (degrees) a hand request set beyond the standard posture
    # each hand's intent: the palm pose its requests last asked for, {"position": [3], "rotation": [3][3]} in the
    # plan frame; a hand absent has none (its reference pose is what it intends: docs/REFERENCE.md 3.3)
    hands: dict | None = None

    def __post_init__(self):
        pose = np.asarray(self.pose, float)
        if pose.shape != (POSE_SIZE,) or not np.isfinite(pose).all():
            raise ValueError("start pose must be 36 finite values")
        object.__setattr__(self, "pose", pose)

    @property
    def height(self) -> float:
        return float(self.pose[2])

    def to_dict(self) -> dict:
        return {
            "pose": self.pose.tolist(),
            "posture": self.posture.value,
            "torso": list(self.torso),
            "hand_intents": self.hands,
        }

    @classmethod
    def from_dict(cls, value: dict) -> StartState:
        """The state ``to_dict`` wrote; keys it does not know are ignored."""
        return cls(
            np.asarray(value["pose"]),
            Posture(value["posture"]),
            tuple(value.get("torso") or (0, 0)),
            value.get("hand_intents"),
        )


@dataclass(frozen=True)
class MotionPhase:
    name: str
    start: int
    stop: int  # exclusive


def phases_from_labels(labels) -> tuple[MotionPhase, ...]:
    """Collapse a per-frame label sequence into contiguous phases."""
    labels = [str(v) for v in labels]
    phases = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            phases.append(MotionPhase(labels[start], start, i))
            start = i
    return tuple(phases)


def labels_from_phases(phases, count: int) -> np.ndarray:
    labels = np.full(count, "", dtype=object)
    for phase in phases:
        labels[phase.start : phase.stop] = phase.name
    return labels.astype(str)


@dataclass
class MotionDraft:
    """What a motion skill or a planner part returns, in the frame its start state is in.

    ``frames[0]`` must equal the start pose. ``support[k, side]`` marks feet
    that must not slide; ``None`` means the skill makes no support claim.
    """

    frames: np.ndarray
    phases: tuple[MotionPhase, ...]
    end_posture: Posture
    support: np.ndarray | None = None
    keyframes: tuple[int, ...] = ()
    diagnostics: dict = field(default_factory=dict)
    limit_stages: tuple[tuple[int, LimitProfile], ...] = ()  # internal frame offsets
    end_torso: tuple[int, int] | None = None  # the torso lean and turn at the end; None leaves the start's

    def __post_init__(self):
        frames = np.asarray(self.frames, float)
        if frames.ndim != 2 or frames.shape[1] != POSE_SIZE or len(frames) < 2:
            raise ValueError("a motion needs at least two 36-value frames")
        self.frames = frames
        if self.support is not None:
            support = np.asarray(self.support, bool)
            if support.shape != (len(frames), 2):
                raise ValueError("support must be [frames, 2]")
            self.support = support
        if not self.phases or self.phases[-1].stop != len(frames):
            raise ValueError("phases must cover every frame")

    def __len__(self):
        return len(self.frames)

    @property
    def duration(self) -> float:
        return (len(self.frames) - 1) * DT

    def labels(self) -> np.ndarray:
        return labels_from_phases(self.phases, len(self.frames))

    def held(self, extra: int) -> MotionDraft:
        """This motion with its last pose held for ``extra`` more frames (labelled hold, feet planted)."""
        if extra <= 0:
            return self
        labels = [*self.labels(), *["hold"] * extra]
        support = None if self.support is None else np.vstack([self.support, np.ones((extra, 2), bool)])
        return replace(
            self,
            frames=np.vstack([self.frames, np.repeat(self.frames[-1:], extra, axis=0)]),
            phases=phases_from_labels(labels),
            support=support,
        )
