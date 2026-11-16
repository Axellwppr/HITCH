"""The model's tools: three, each a fixed sequence of the session host's operations (docs/REFERENCE.md).

- ``execute``: ``plan`` the structured request, and if it is ready, ``execute``
  it, wait until it has been played, close or open the fingers it asks for
  (the session's ``grip``, with its stand-still and the request's ``wait_s``),
  and ``observe`` the result. A refused plan stops there: nothing moved, and
  the last observation still holds. A plan refused only for being too fast is
  planned once more with slower default timing before giving up. A plan
  refused because the latest observation is stale observes again, so the
  model can plan in a fresh frame O.
- ``perceive``: the session's ``perceive`` of the given normalized image points on the latest
  observation, always with the marked image.
- ``finish``: end the episode.

This layer only chooses what to call and in which order. The text the model
reads comes from ``render``; the session results it was made from are kept as
the result's ``data`` (each operation with its arguments and result or error),
so the episode can be rendered again. Every failure, including bad arguments,
comes back as text for the model to read; nothing a model sends can stop the
loop.
"""

from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from wbc.agent import render
from wbc.agent.conversation import Image, ToolCall
from wbc.agent.prompts import tool_descriptions
from wbc.core.errors import STALE, Cause, WbcError, cause_of
from wbc.core.hands import GRIP_S, hand_grips
from wbc.core.limits import SPEED_CHECKS
from wbc.core.request import parse_request
from wbc.core.types import RUN_ENDED
from wbc.session import client

POLL_S = 120.0  # one HTTP long poll while waiting for a run
WAIT_MARGIN_S = 60.0  # a run may take this much longer than its plan before execute stops waiting
# Wall seconds per simulated second to allow for a motion: a simulator slower than real time needs more.
WAIT_SCALE = float(os.environ.get("WBC_WAIT_SCALE", "1"))
NOT_SENT = ("PLAN_STALE", "PLAN_REJECTED", "SESSION_STATE", "PRECONDITION", "INVALID_REQUEST")
TERMINAL = frozenset(state.value for state in RUN_ENDED)  # a run's states as results give them
# Too fast: the walker/hand reasons that slower timing can fix (the plan checks: limits.SPEED_CHECKS).
SPEED_REASONS = {"SPEED_LIMIT", "HAND_PATH_RATE_LIMIT"}


def _object(properties: dict, required=()) -> dict:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


def _field(kind, description: str, **more) -> dict:
    return {"type": kind, "description": description, **more}


def schemas() -> dict[str, dict]:
    """Parameter schemas of the three tools; the full rules are in the system prompt (prompts/system.md)."""
    vector = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    waypoint = _object(
        {
            "position": _field(
                "array", "palm [x, y, z] in O, m; z above the ground", items={"type": "number"}, minItems=3, maxItems=3
            ),
            "duration_s": _field("number", "time from the previous waypoint (or the start), 0.4-8 s"),
            "orientation": _object(
                {
                    "palm": _field("string", "preset in O", enum=["inward", "down", "up", "forward"]),
                    "normal": {**vector, "description": "palm normal in O, with fingers"},
                    "fingers": {**vector, "description": "finger direction in O, perpendicular to normal"},
                }
            ),
        },
        ["position", "duration_s"],
    )
    hand = _object(
        {
            "waypoints": _field("array", "1-12 palm waypoints in O, in order", items=waypoint, minItems=1, maxItems=12),
            "home": _field("boolean", "return the arm to its rest posture"),
            "follow": _field(
                "string", "how a hand without waypoints moves with the body", enum=["root", "world", "body"]
            ),
            "grip": _field("string", "fingers after the in-place part", enum=["open", "closed"]),
        }
    )
    point = {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}, "minItems": 2, "maxItems": 2}
    footstep = _object(
        {
            "side": _field("string", "the stepping foot", enum=["left", "right"]),
            "xy": _field("array", "ankle [x, y] in O, m", items={"type": "number"}, minItems=2, maxItems=2),
            "yaw_deg": _field("integer", "foot heading in O, degrees"),
            "root_yaw_deg": _field("integer", "pelvis heading in O reached during this step, degrees"),
        },
        ["side", "xy"],
    )
    return {
        "execute": _object(
            {
                "height": _field(["number", "string"], 'pelvis height 0.30-0.79 m, or "auto"'),
                "torso": _object(
                    {
                        "pitch_deg": _field(
                            "integer", "lean beyond this height's standard posture, 0-40, multiples of 5"
                        ),
                        "yaw_deg": _field("integer", "turn at the waist, -45-45, multiples of 5, left +"),
                    },
                    ["pitch_deg", "yaw_deg"],
                ),
                "body_duration_s": _field("number", "time of the height and torso change, 0.6-8 s"),
                "hands": _object({"left": hand, "right": hand}),
                "walk": _object(
                    {
                        "target": _field(
                            "array",
                            "pelvis destination [x, y] in O, m",
                            items={"type": "number"},
                            minItems=2,
                            maxItems=2,
                        ),
                        "turn_deg": _field("integer", "heading change by the end, -180-180, left +"),
                        "footsteps": _field("array", "1-12 explicit steps instead of target", items=footstep),
                    }
                ),
                "wait_s": _field("number", "stand still at the end, 0.2-10 s"),
                "eye": _object(
                    {
                        "yaw_deg": _field("integer", "camera turn left of the torso, -45-45, multiples of 15"),
                        "pitch_deg": _field("integer", "camera tilt down from the torso, 0-60, multiples of 15"),
                    },
                    ["yaw_deg", "pitch_deg"],
                ),
            }
        ),
        "perceive": _object(
            {
                "points": _field(
                    "array",
                    "[x, y] in rgb.png as integers normalized to 0–1000 on each axis; "
                    "top-left [0,0], bottom-right [1000,1000], x right, y down",
                    items=point,
                    minItems=1,
                    maxItems=100,
                ),
            },
            ["points"],
        ),
        "finish": _object(
            {
                "success": _field("boolean", "true only if your latest observations show the task done"),
                "summary": _field("string", "what you did, what you saw at the end, anything uncertain"),
            },
            ["success", "summary"],
        ),
    }


@dataclass
class Result:
    text: str
    images: tuple[Image, ...] = ()
    data: dict | None = None  # the session operations behind the text
    finished: dict | None = None  # set by finish
    moved: bool = False  # a motion was sent (or the frame was renewed): the latest observation, and its O, are new


class Tools:
    def __init__(self, session: str, images: Path):
        self.session = session  # the session's URL
        self.images = images  # where fetched images are kept
        self.seen: str | None = None  # the latest observation the model has been shown
        self.fingers: bool = hand_grips()  # finger control: this process's hand, then the latest observation's
        self.walked = False  # a walk was played since the latest idle observation
        descriptions = tool_descriptions()
        self.specs: list[dict] = [  # name, description, parameters
            {"name": name, "description": descriptions[name], "parameters": schema}
            for name, schema in schemas().items()
        ]

    def run(self, call: ToolCall) -> Result:
        try:
            args = json.loads(call.arguments or "{}", parse_constant=_no_constant, parse_float=_finite)
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as error:
            return _refused("INVALID_ARGUMENTS", f"the arguments are not a JSON object: {error}")
        offered = [spec["name"] for spec in self.specs]
        operation = getattr(self, f"_{call.name}", None)
        if operation is None or call.name not in offered:
            usable = ", ".join(offered[:-1]) + " or " + offered[-1]  # "execute, perceive or finish"
            return _refused("UNKNOWN_TOOL", f"there is no tool {call.name!r}; use {usable}")
        try:
            inspect.signature(operation).bind(**args)
        except TypeError as error:  # unexpected or missing arguments
            return _refused("INVALID_ARGUMENTS", f"{call.name}: {error}")
        return operation(**args)

    def skip(self, call: ToolCall, why: str) -> Result:
        """A call the loop does not run, with the reason as its result."""
        return _refused("NOT_RUN", why, {"tool": call.name, "calls": []})

    def observe(self) -> Result:
        """The current observation, as the first message shows it."""
        record = {"tool": "observe", "calls": []}
        observation = self._step(record, "observe", since=self.seen)
        if observation is None:
            return Result(render.failure(record["calls"][-1]["error"], "Observing"), data=record)
        self._observed(observation)
        images = self._fetch(observation)  # first: a failed download is part of the observation's text
        return Result(render.observation(observation), images, record)

    # ------------------------------------------------------------- the tools
    def _execute(
        self, height=None, torso=None, body_duration_s=None, hands=None, walk=None, wait_s=None, eye=None
    ) -> Result:
        record = {"tool": "execute", "calls": []}
        given = dict(
            height=height, torso=torso, body_duration_s=body_duration_s, hands=hands, walk=walk, wait_s=wait_s, eye=eye
        )
        request = {key: value for key, value in given.items() if value is not None}
        try:
            parsed = parse_request(request)
        except WbcError as error:
            return _refused("INVALID_ARGUMENTS", error.message, record)
        grips = parsed.grips
        if grips and not self.fingers:
            return _refused("INVALID_ARGUMENTS", "these hands have no finger control: leave out grip", record)
        hold_s = GRIP_S + (parsed.wait_s or 0.0)
        look = request.pop("eye", None)  # the camera turns once the rest is done, for the observation after it
        if not grips and not parsed.moves:  # the camera alone
            return self._observe_into(record, {}, look)
        if grips:  # the stand-still after the fingers takes the wait (docs/REFERENCE.md 3.1)
            request.pop("wait_s", None)
            if not parsed.in_place and parsed.walk is None:
                return self._grip(record, {}, grips, hold_s, look)
        return self._plan_and_send(record, request, parsed, hold_s, look)

    def _plan_and_send(self, record: dict, request: dict, parsed, hold_s: float, look: dict | None) -> Result:
        """Plan ``request`` and send a ready plan to be executed; a plan that is not ready ends the call here."""
        view, plan = self._plan_timed(record, request, parsed)
        if plan is None and _stale(view):
            view["stale"] = True
            if cause_of(view["plan_error"]) == Cause.MOVING:
                self._step(record, "wait", timeout=POLL_S)  # the latest run, to its end
            return self._observe_into(record, view)
        if plan is None or plan["state"] != "ready":
            return Result(render.execute(view), data=record)
        run = self._step(record, "execute", plan=plan["plan_id"])
        return self._finish_motion(record, view, plan, run, parsed, hold_s, look)

    def _finish_motion(
        self, record: dict, view: dict, plan: dict, run: dict | None, parsed, hold_s: float, look: dict | None
    ) -> Result:
        """After ``execute`` (``run``, None when it failed): wait for the run, move the fingers, observe."""
        grips = parsed.grips
        executed = _maybe_executed(plan, run, record)
        self.walked |= parsed.walk is not None and executed
        if run is None:
            view["execute_error"] = record["calls"][-1]["error"]
            # A plan gone stale since planning means the robot's reference moved: observe it (docs/REFERENCE.md 4.4)
            if view["execute_error"].get("code") in NOT_SENT and view["execute_error"]["code"] != "PLAN_STALE":
                return Result(render.execute(view), data=record)
        else:
            started = time.monotonic()
            deadline = started + plan["duration_s"] * WAIT_SCALE + WAIT_MARGIN_S
            if look is not None:  # the head turns while the body moves; the observation settles it (unrendered)
                self._step(record, "aim", eye=look)
            run = self._wait(record, run, deadline)
            view["run"], view["waited_s"] = run, time.monotonic() - started
            if run["state"] == "refused":
                return Result(render.execute(view), data=record)
            if grips and run["state"] == "done":
                return self._grip(record, view, grips, hold_s, look)
        if grips:
            view["grip_skipped"] = {"commanded": grips, "wait_s": parsed.wait_s}
        return self._observe_into(record, view, look if executed else None)  # a refused request turns nothing

    def _grip(self, record: dict, view: dict, grips: dict, hold_s: float, look: dict | None = None) -> Result:
        """Fingers after the in-place part (or alone): the session's ``grip``, its stand-still, then observe."""
        result = self._step(record, "grip", fingers=grips, hold_s=hold_s)
        if result is None:
            view["grip_error"] = record["calls"][-1]["error"]
        else:
            run, deadline = result, time.monotonic() + hold_s * WAIT_SCALE + WAIT_MARGIN_S
            if look is not None:
                self._step(record, "aim", eye=look)
            run = self._wait(record, run, deadline, to_deadline=False)
            view["grip"] = {
                "fingers": result["fingers"],
                "commanded": grips,
                "state": run["state"],
                "hold_s": hold_s,
                "max_effector_error_m": run.get("max_effector_error_m"),
            }
        return self._observe_into(record, view, look)

    def _wait(self, record: dict, run: dict, deadline: float, *, to_deadline: bool = True) -> dict:
        """Long-poll the session's ``wait`` on ``run`` until it ends, a poll fails or ``deadline`` (``time.monotonic``)
        passes; the run as last seen. Each poll waits POLL_S at most, and with ``to_deadline`` no longer than the time
        left (a grip's polls are not cut at its deadline)."""
        while run["state"] not in TERMINAL and time.monotonic() < deadline:
            timeout = min(POLL_S, max(0.0, deadline - time.monotonic())) if to_deadline else POLL_S
            polled = self._step(record, "wait", run=run["run_id"], timeout=timeout)
            if polled is None:
                break
            run = polled
        return run

    def _observe_into(self, record: dict, view: dict, look: dict | None = None) -> Result:
        """Turn the camera (``look``, when the request gave an eye), then observe."""
        if look is not None:
            turned = self._step(record, "look", eye=look)
            view["look"] = turned if turned is not None else {"error": record["calls"][-1]["error"]}
        observation = self._step(record, "observe", since=self.seen)
        images = ()
        if observation is None:
            view["observe_error"] = record["calls"][-1]["error"]
        else:
            view["walked"] = self.walked  # since the observation the model saw, numbers say nothing (render._since)
            self._observed(observation)
            if not observation.get("queued_s"):
                self.walked = False
            view["observation"], images = observation, self._fetch(observation)
        return Result(render.execute(view), images, record, moved=True)

    def _perceive(self, points: list) -> Result:
        record = {"tool": "perceive", "calls": []}
        result = self._step(record, "perceive", points=points, show=True)
        if result is None:
            return Result(render.failure(record["calls"][-1]["error"], "Perceiving"), data=record)
        images = self._fetch(result)
        return Result(render.perceived(result), images, record)

    def _finish(self, success: bool, summary: str) -> Result:
        claimed = success.strip().lower() == "true" if isinstance(success, str) else bool(success)
        finished = {"success": claimed, "summary": str(summary)}
        return Result("The episode is finished.", data={"tool": "finish", "calls": []}, finished=finished)

    # ---------------------------------------------------------------- helpers
    def _plan_timed(self, record: dict, request: dict, parsed) -> tuple[dict, dict | None]:
        """Plan ``request``, and once more slower when only its speed was refused and it has a body change or a
        walk to slow; the view for ``render.execute`` and the plan to execute (None after an error)."""
        view = {}
        plan = self._plan(record, view, request, slower=False)
        if _too_fast(view) and (parsed.height is not None or parsed.torso is not None or parsed.walk is not None):
            view = {"first": view}  # only the body change and the walk have timing to slow
            record["slowed"] = True
            plan = self._plan(record, view, request, slower=True)
        return view, plan

    def _plan(self, record: dict, view: dict, request: dict, *, slower: bool):
        """One ``plan`` call; its verdict goes into ``view`` for ``render.execute``."""
        plan = self._step(record, "plan", request=request, slower=slower or None)
        if plan is None:
            view["plan_error"] = record["calls"][-1]["error"]
        else:
            view["plan"] = plan
        return plan

    def _observed(self, observation: dict) -> None:
        self.seen = observation.get("observation_id")
        hands = (observation.get("body") or {}).get("hands") or {}
        if hands:  # the session adds each hand's finger state for hands that have fingers
            self.fingers = all("fingers" in hand for hand in hands.values())

    def _step(self, record: dict, op: str, **args):
        """One session operation, recorded; its result, or None after recording its error."""
        args = {k: v for k, v in args.items() if v is not None}
        try:
            result = client.call(self.session, op, **args)
        except WbcError as error:
            record["calls"].append({"op": op, "args": args, "error": error.to_dict()})
            return None
        record["calls"].append({"op": op, "args": args, "result": result})
        return result

    def _fetch(self, result: dict) -> tuple[Image, ...]:
        """Each image a result lists, fetched and saved under ``images``."""
        images = []
        for label in result.get("images", ()):
            path = self.images / "_".join(Path(label).parts[1:])  # observations/obs-1/rgb.png -> obs-1_rgb.png
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                path.write_bytes(client.file(self.session, label))
            except WbcError as error:  # recorded as a camera failure of this result (the loop's fault check)
                result.setdefault("vision_error", f"{label}: {error.code.value}: {error}")
                continue
            images.append(Image(path, label))
        return tuple(images)


def _no_constant(name: str):
    raise ValueError(f"{name} is not a number")


def _finite(text: str) -> float:
    value = float(text)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{text} is not a finite number")
    return value


def _too_fast(view: dict) -> bool:
    """Whether a refused plan failed only on speed, which slower timing can fix."""
    if view.get("plan_error"):
        return view["plan_error"].get("reason") in SPEED_REASONS
    plan = view.get("plan") or {}
    names = {f["name"] for f in plan.get("failures", ())}
    return plan.get("state") == "rejected" and bool(names) and names <= SPEED_CHECKS


def _maybe_executed(plan: dict | None, run: dict | None, record: dict) -> bool:
    """Whether a planned motion was, or may have been, appended: its run was not refused, or its ``execute`` failed
    with an error other than a refusal before appending (which leaves the outcome uncertain)."""
    if run is not None:
        return run["state"] != "refused"
    return plan is not None and plan["state"] == "ready" and record["calls"][-1]["error"].get("code") not in NOT_SENT


def _stale(view: dict) -> bool:
    """Whether the plan was refused because the latest observation no longer describes the robot."""
    error = view.get("plan_error") or {}
    return error.get("code") == "PRECONDITION" and cause_of(error) in STALE


def _refused(code: str, message: str, record: dict | None = None) -> Result:
    error = {"code": code, "message": message}
    return Result(render.failure(error, "The call"), data={**(record or {}), "refused": error})
