"""The walk skill: a request's walk (``core.request.Walk``, integer degrees in O) plus the timing the harness sets
itself, converted once to the walker's goal in radians (``walking.footsteps.WalkGoal``); body generation and bounded
stance recovery live in ``walking/``."""

from dataclasses import dataclass

import numpy as np

from wbc.core.errors import ErrorCode, WbcError
from wbc.core.frames import SE2
from wbc.core.params import param, parse, reframe, to_json
from wbc.core.request import Walk
from wbc.core.types import MotionDraft, StartState
from wbc.skills.walking import plan_body
from wbc.skills.walking.footsteps import WalkGoal


@dataclass(frozen=True)
class WalkParams(Walk):
    """``target`` and ``footsteps`` are in O (the planner maps them into the step's frame); ``turn_deg`` is relative to
    the step's starting heading."""

    swing_s: float | None = param(None, range=(0.4, 1.8), doc="seconds per swing")
    swing_height_m: float = param(0.045, range=(0.03, 0.28))
    recover: bool | None = param(
        None, doc="collect feet once on a stance failure; default on for target, off for explicit footsteps"
    )

    def body_parameters(self) -> dict:
        """Convert the public command once; reframing may already have added fractional degrees."""
        parameters = to_json(self)
        parameters["turn"] = float(np.radians(parameters.pop("turn_deg")))
        for step in parameters["footsteps"]:
            for name in ("yaw", "root_yaw"):
                value = step.pop(name + "_deg")
                step[name] = None if value is None else float(np.radians(value))
        return parameters


def restate_attempts(attempts, to_plan: SE2) -> None:
    """The walker's recovery attempts (an error's ``attempts``, built in the step's frame S) with their requests in O
    and in public units, in place, using the same parameter metadata as the forward conversion."""
    for attempt in attempts:
        if attempt.get("parameters"):
            try:
                attempt["parameters"] = to_json(reframe(parse(WalkGoal, attempt["parameters"]), to_plan))
            except WbcError:  # an internal retry may exceed the agent-facing ranges; keep it as built, in S
                attempt["parameters_frame"] = "S"
            attempt["parameters"] = describe_attempt(attempt["parameters"])


def describe_attempt(parameters: dict) -> dict:
    """Expose an internal recovery request in public units, after mapping it back to O."""
    parameters = {**parameters, "footsteps": [dict(s) for s in parameters.get("footsteps", ())]}
    if "turn" in parameters:
        parameters["turn_deg"] = round(float(np.degrees(parameters.pop("turn"))))
    for step in parameters["footsteps"]:
        for name in ("yaw", "root_yaw"):
            if name in step:
                value = step.pop(name)
                step[name + "_deg"] = None if value is None else round(float(np.degrees(value)))
    return parameters


def walk(start: StartState, p: WalkParams) -> MotionDraft:
    """Walk at current height (0.58..0.78 m) to a root target or explicit foot placements, the torso in its standard
    posture (the request planner says so first, with the agent's remedy: planner.request._check_walk)."""
    if tuple(start.torso) != (0, 0):
        lean, turn = start.torso
        raise WbcError(
            ErrorCode.PRECONDITION,
            f"walk: the torso is leaned {lean}° and turned {turn}°; return it first (torso pitch_deg 0, yaw_deg 0)",
        )
    return plan_body(start, **p.body_parameters())
