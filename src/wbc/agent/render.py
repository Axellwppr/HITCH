"""The session's JSON results, written as the short natural-language reports the model reads.

Every function here is pure: it takes what the session host returned (the
records ``tools.py`` keeps for each tool call) and returns text. The wording is
fixed, so episodes read alike and can be rendered again from their records. Scene positions use centimetres;
planning residuals retain millimetre precision where needed;
coordinates are always in the frame O of the observation they came from.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np

from wbc.core.errors import Cause, cause_of
from wbc.core.layout import FPS  # frame indices in plan findings
from wbc.core.limits import CHECK_REASONS, SLOWER

EDGE_SPREAD_M = 0.03  # depth range around a located pixel above which it is probably an edge
UNITS = ("m", "deg", "m/s", "deg/s", "s", "kg", "N")

FRAME_O = "origin on the ground under the pelvis, x forward, y left, z up, metres"
INTENT_SAME_M, INTENT_SAME_DEG = 0.005, 3.0  # a plan this close to what was requested is said to be as requested
TURN_SAME_DEG = 1.0  # a heading change below this (IMU noise, feet planted) is not carried into the conversion
# Walk parameters the harness sets itself: recovery attempts show the rest.
_HIDDEN_PARAMS = {"walk": ("swing_s", "swing_height_m", "recover")}
IMAGES = {"rgb.png": "the head camera"}
# error fields written above the details, or of no use to the model (frame bookkeeping, evidence kind)
SHOWN = {
    "code",
    "message",
    "reason",
    "stage",
    "time_s",
    "frame",
    "suggestion",
    "attempts",
    "stance",
    "step_index",
    "skill",
    "plan_frame_offset",
    "frame_scope",
    "evidence",
    # rate limits: in the message
    "height_adjustment",
    "waypoints",
    "window_s",
    "required_duration_s",
    # written above
    "part",
    "hands",
    "body_height_m",
    # for the tools (wbc.core.errors.Cause), and said in the text
    "cause",
    "fingers_commanded",
}


# ------------------------------------------------------------------- numbers
def num(value, digits: int = 3) -> str:
    text = f"{float(value):.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def xyz(values, digits: int = 2) -> str:
    return "(" + ", ".join(num(v, digits) for v in values) + ")"


def _value(value) -> str:
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return num(value) if isinstance(value, float) else str(value)
    if isinstance(value, list) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        return xyz(value, 3) if value else "[]"
    if isinstance(value, str):
        return value
    return json.dumps(_rounded(value), ensure_ascii=False)


def _rounded(value):
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------- observation
def observation(obs: dict, *, walked: bool = False, deviation: dict | None = None) -> str:
    """Body state and the attached images of one ``observe`` result (docs/REFERENCE.md 5.1): where this O lies in the
    one the model saw before (not after a walk: ``walked``), and the body measured and as the controller's reference.
    ``deviation`` is each effector's largest distance from its reference during the motion just played."""
    body = obs["body"]
    reference = body.get("reference") or {}
    lines = [f"Observation {obs['observation_id']}. Its frame O: {FRAME_O}."]
    if obs.get("since"):
        lines.append(_since(obs["since"], walked))
    if obs.get("queued_s"):
        lines.append(
            f"- The robot is still moving: {num(obs['queued_s'], 1)} s of motion remain; planned values are "
            "where the reference is now."
        )
    lines.append(_body(body, reference))
    for side, foot in body["feet"].items():
        planned = (reference.get("feet") or {}).get(side)
        line = f"- {side.capitalize()} foot: ankle {xyz(foot['ankle_xyz'])}, yaw {num(foot['yaw_deg'], 0)}°"
        if foot["sole_height_m"] > 0.01:
            line += f", sole {num(foot['sole_height_m'], 2)} m above the ground"
        if planned:
            line += f"; planned {xyz(planned['ankle_xyz'])}, yaw {num(planned['yaw_deg'], 0)}°"
        lines.append(line + ".")
    for side, hand in body["hands"].items():
        lines.append(_hand(side, hand, (reference.get("hands") or {}).get(side), (deviation or {}).get(f"{side}_hand")))
    if obs.get("vision_error"):
        lines.append(f"- Camera: failed ({obs['vision_error']}).")
    elif obs.get("eye"):
        lines.append(_eye(obs["eye"]))
    names = [label.rsplit("/", 1)[-1] for label in obs.get("images", ())]
    if names:
        lines.append("- Images: " + "; ".join(f"{name} ({IMAGES.get(name, 'image')})" for name in names) + ".")
    return "\n".join(lines)


def _eye(eye: dict) -> str:
    """Where the camera looked: its mount setting (relative to the torso) and its optical axis in O."""
    actual = f"its view {num(eye['down_deg'], 0)}° below the horizontal, heading {num(eye['heading_deg'], 0)}° in O"
    if "yaw_deg" not in eye:
        return f"- Camera: fixed; {actual}."
    turn = (
        "straight ahead"
        if not eye["yaw_deg"]
        else (f"{abs(eye['yaw_deg'])}° {'left' if eye['yaw_deg'] > 0 else 'right'}")
    )
    return f"- Camera (eye): yaw {eye['yaw_deg']}° ({turn}), pitch {eye['pitch_deg']}° down from the torso; {actual}."


def _since(since: dict, walked: bool) -> str:
    """Where this O lies in the previous one, and how to carry a point over (nothing after a walk or a restart)."""
    before = since["observation_id"]
    if since.get("epoch_changed"):
        return (
            f"The controller restarted since {before}: its reference world was reset, so coordinates from "
            f"{before} cannot be carried over; measure again."
        )
    if walked:
        return (
            f"The robot walked since {before}: O moved with it, by an amount nobody measured (an obstacle or a "
            f"slip can leave the robot short of the walk's target). Coordinates from {before} no longer apply; "
            "measure again."
        )
    (x, y), turn = since["origin_xy"], since["turn_deg"]
    if abs(x) < 0.005 and abs(y) < 0.005 and abs(turn) < TURN_SAME_DEG:
        return (
            f"O has not moved since {before}, as planned with both feet planted: coordinates from {before} still "
            "apply, unless something pushed the robot or a foot slipped."
        )
    text = (
        f"Since {before}, O moved with the pelvis, as planned with both feet planted (a push or a slipping foot "
        f"would move it further, unmeasured): its origin is at {xyz([x, y])} in {before}'s O, turned "
        f"{num(turn, 0)}°. A point (x, y, z) of {before} is at "
    )
    if abs(turn) < TURN_SAME_DEG:
        return text + f"({_offset('x', x)}, {_offset('y', y)}, z) here."
    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    dx, dy = _offset("x", x), _offset("y", y)
    return text + (f"({num(c, 3)}·({dx}) + {num(s, 3)}·({dy}), {num(-s, 3)}·({dx}) + {num(c, 3)}·({dy}), z) here.")


def _offset(name: str, value: float) -> str:
    """``name`` minus ``value``, as a reader writes it: x - 0.05, y + 0.02."""
    if abs(value) < 0.005:
        return name
    return f"{name} {'-' if value > 0 else '+'} {num(abs(value), 2)}"


def _body(body: dict, reference: dict) -> str:
    pelvis = body["pelvis"]
    planned = (reference.get("pelvis") or {}).get("xyz")
    text = (
        f"- Body: {body['posture']}; pelvis {num(pelvis['xyz'][2], 2)} m high"
        + (f" (planned {num(planned[2], 2)})" if planned else "")
        + f", roll {num(pelvis['roll_deg'], 0)}°, pitch {num(pelvis['pitch_deg'], 0)}°."
    )
    torso, measured = body.get("torso"), body.get("torso_measured")
    if torso:
        text += (
            f" Torso: leaned {torso['lean_deg']}° and turned {torso['turn_deg']}° beyond the standard posture "
            "of this height as planned"
        )
        text += f" (measured {measured['lean_deg']}° and {measured['turn_deg']}°)" if measured else ""
    elif measured:
        text += (
            f" Torso: leaned {measured['lean_deg']}° and turned {measured['turn_deg']}° beyond the standard "
            "posture of this height (measured)"
        )
    if measured:
        text += f"; {measured['pitch_deg']}° from vertical in all."
    elif torso:
        text += "."
    return text


def _hand(side: str, hand: dict, planned: dict | None, deviation: float | None) -> str:
    """One palm: position and directions measured, and as planned, how far apart, and its largest distance from its
    reference during the motion just played; then the fingers."""
    text = f"- {side.capitalize()} hand: palm {xyz(hand['palm_xyz'])}"
    if hand.get("normal") is not None:
        text += f", normal {xyz(hand['normal'])}, fingers {xyz(hand['finger_direction'])}"
    if planned:
        text += f"; planned {xyz(planned['palm_xyz'])}"
        gap = f"{num(100 * math.dist(hand['palm_xyz'], planned['palm_xyz']), 1)} cm"
        if planned.get("normal") is not None and hand.get("normal") is not None:
            same = planned["normal"] == hand["normal"] and planned["finger_direction"] == hand["finger_direction"]
            text += (
                ", same directions"
                if same
                else f", normal {xyz(planned['normal'])}, fingers {xyz(planned['finger_direction'])}"
            )
            gap += f" and {num(_angle(side, hand, planned), 0)}°"
        text += f"; {gap} apart"
    if deviation is not None:
        text += f"; at most {num(100 * deviation, 1)} cm from its reference during the motion"
    intent = (planned or {}).get("intent")
    if intent is not None:  # what the requests asked of this hand, and how far the plan got from it
        if intent["from_reference_m"] < INTENT_SAME_M and intent["from_reference_deg"] < INTENT_SAME_DEG:
            text += "; planned as requested"
        else:
            text += (
                f"; requested {xyz(intent['palm_xyz'])}, normal {xyz(intent['normal'])}, fingers "
                f"{xyz(intent['finger_direction'])}: the plan is {num(100 * intent['from_reference_m'], 1)} cm "
                f"and {num(intent['from_reference_deg'], 0)}° from it"
            )
    text += "."
    if hand.get("fingers") in ("open", "closed"):
        text += f" Fingers {hand['fingers']}."
    return text


def _angle(side: str, a: dict, b: dict) -> float:
    """Degrees between two palm orientations given as normal and finger directions."""

    def frame(hand):
        x = np.asarray(hand["finger_direction"], float)
        y = np.asarray(hand["normal"], float) * (-1 if side == "left" else 1)
        x, y = x / max(np.linalg.norm(x), 1e-9), y / max(np.linalg.norm(y), 1e-9)
        return np.column_stack([x, y, np.cross(x, y)])

    relative = frame(a).T @ frame(b)
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0))))


# ---------------------------------------------------------------------- plan
def plan_ready(plan: dict) -> str:
    """A plan that passed its checks: how long, what it does in place, and what the planner changed."""
    notes, parts = [], []
    for entry in plan.get("body", ()):
        kind = entry.get("kind")
        if kind == "in_place":
            parts.append(f"{num(entry['duration_s'], 2)} s in place")
            change = _in_place(entry)
            if change:
                notes.append(change)
        elif kind == "walk":
            parts.append(f"then walking from {num(entry['start_s'], 2)} s" if parts else "walking")
            if entry.get("recovered"):
                decision = (entry.get("recovery") or {}).get("decision", "a recovery")
                notes.append(
                    f"The walk needed a stance correction first ({decision}); it is part of the executed motion."
                )
        elif kind == "hand_timing":
            changed = [s for s in entry["segments"] if s["duration_s"] > s["requested_duration_s"] + 1e-8]
            if changed:
                detail = "; ".join(
                    f"{num(s['requested_duration_s'], 2)} -> {num(s['duration_s'], 2)} s (ends at {num(s['end_time_s'], 2)} s)"
                    for s in changed
                )
                notes.append(f"To keep within the speed limits the planner slowed these intervals: {detail}.")
        elif kind == "hand_height":
            heights = "; ".join(f"{num(w['height_m'], 2)} m at {num(w['time_s'], 2)} s" for w in entry["waypoints"])
            notes.append(f"Height auto: the pelvis follows the hand path with both feet planted: {heights}.")
    head = f"The plan passed its checks: {num(plan['duration_s'], 2)} s" + (f" ({', '.join(parts)})" if parts else "")
    lines = [head + "."] + notes + [f"Warning: {_finding(w)}" for w in plan.get("warnings", ())]
    return "\n".join(lines)


def _in_place(entry: dict) -> str:
    """The height and torso change of the in-place part, if any."""
    (h0, h1), (t0, t1) = entry["height_m"], entry["torso_deg"]
    changes = []
    if abs(h1 - h0) > 1e-4:
        changes.append(f"pelvis {num(h0, 2)} -> {num(h1, 2)} m")
    if t0[0] != t1[0]:
        changes.append(f"lean {t0[0]} -> {t1[0]}°")
    if t0[1] != t1[1]:
        changes.append(f"turn {t0[1]} -> {t1[1]}°")
    if not changes:
        return ""
    return f"Body: {', '.join(changes)} over {num(entry['body_s'], 2)} s from the start, both feet planted."


def plan_rejected(plan: dict) -> str:
    lines = ["The plan was rejected by its checks:"]
    lines += [f"- {_finding(f)}" for f in plan.get("failures", ())]
    if plan.get("warnings"):
        lines.append("Warnings:")
        lines += [f"- {_finding(w)}" for w in plan["warnings"]]
    return "\n".join(lines)


def _finding(finding: dict, clock: str = "") -> str:
    """One failed or warning check: what, how much against which limit, and when (``clock``: since what)."""
    name, detail = finding["name"], finding.get("detail", "")
    if name in CHECK_REASONS:
        name = f"{CHECK_REASONS[name]} ({name})"
    value, limit = finding.get("value"), finding.get("limit")
    if value is not None and detail.split(" ", 1)[0] in UNITS:
        text = f"{name}: {num(value)} {detail}"
        extra = [f"limit {num(limit)}"] if limit is not None else []
    elif value is None and detail:  # a note rather than a measurement, e.g. a recovery note
        text = detail[0].upper() + detail[1:]
        extra = []
    else:
        text = f"{name}: {detail}" if detail else name
        extra = [f"value {num(value)}"] if value is not None else []
        extra += [f"limit {num(limit)}"] if limit is not None else []
    if finding.get("parts"):
        text += f"; {_contact(finding['parts'])}"
    if finding.get("frame") is not None:
        extra.append(f"worst at {num(finding['frame'] / FPS, 2)} s{clock}")
    return text + (f" ({', '.join(extra)})" if extra else "")


def _contact(parts) -> str:
    """The robot parts touching at the worst frame."""
    return " and the ".join(["the " + _part(parts[0])] + [_part(p) for p in parts[1:]]) + " touch"


def _part(geom: str) -> str:
    """A robot collision geom as words: left_hand_palm_collision -> left hand."""
    name = geom.removesuffix("_collision")
    name = re.sub(r"foot\d+$", "foot", name)
    name = name.replace("hand_palm", "hand")
    return name.replace("_", " ")


def failure(error: dict, what: str) -> str:
    """A refused call (``what``: e.g. "Planning"), with every detail it carried (the arms' raw residuals only when
    no explanation of the failing hands is given)."""
    head = [error.get("code", "ERROR")]
    reason = error.get("reason")
    if reason == "COLLISION" and error.get("stage") == "hand_rest":
        reason = "HAND_REST"
    if reason:
        head.append(reason)
    where = []
    if error.get("part"):
        where.append(f"the {error['part']}")
    if error.get("skill"):
        where.append(f"the {error['skill']}")
    if error.get("stage"):
        where.append(f"stage {error['stage']}")
    if error.get("time_s") is not None:
        where.append(f"{num(error['time_s'], 2)} s into the plan")
    if error.get("body_height_m") is not None:
        where.append(f"pelvis {num(error['body_height_m'], 2)} m high then")
    text = f"{what} failed ({', '.join(head)}{'; ' + ', '.join(where) if where else ''}): {error.get('message', '')}"
    for side, hand in (error.get("hands") or {}).items():
        text += "\n" + _failing_hand(side, hand)
    suggestion = error.get("suggestion") or {}
    shift = suggestion.get("translation_O_m")
    if shift:
        text += (
            f"\nSuggestion: try moving the body by about {xyz(shift[:2])} m in XY of this observation's O frame. "
            "This passed a reach check with the body pose held fixed; confirm the route in the image, "
            "then observe and re-plan after moving."
        )
    if error.get("attempts"):
        text += "\nAttempts:\n" + "\n".join(_attempt(n, a) for n, a in enumerate(error["attempts"], 1))
    if error.get("stance"):
        text += "\nStance: " + _stance(error["stance"])
    if error.get("height_adjustment"):
        attempt = error["height_adjustment"]["segments"][-1]
        candidates = "; ".join(
            f"{num(c['height_m'], 2)} m: "
            + ("passed" if c["passed"] else c["error"].get("reason", c["error"].get("code", "rejected")))
            for c in attempt["candidates"]
        )
        text += (
            f"\nHeight retries over {num(attempt['start_frame'] / FPS, 2)}–"
            f"{num(attempt['end_frame'] / FPS, 2)} s: {candidates}."
        )
        if error["height_adjustment"].get("foot_policy") == "fixed":
            text += "\nAll height trials kept both feet at their starting placements."
    shown = SHOWN | ({"arms"} if error.get("hands") else set())
    details = _details({k: v for k, v in error.items() if k not in shown})
    return text + ("\nDetails:\n" + "\n".join(details) if details else "")


def _failing_hand(side: str, hand: dict) -> str:
    """What a hand that could not be planned was doing, and whether the target or the body put it out of reach."""
    text = f"{side.capitalize()} hand ({hand['doing']})"
    if "target_O_m" not in hand:
        return text + f": the body carried its shoulder {xyz(hand['shoulder_moved_m'])} m by then."
    start, now, reach = hand["distance_at_start_m"], hand["distance_now_m"], hand["reach_m"]
    text += (
        f": target {xyz(hand['target_O_m'])} in O, {num(start, 2)} m from the shoulder at the start and "
        f"{num(now, 2)} m at that moment (the arm reaches {num(reach, 2)} m); the body had carried the shoulder "
        f"{xyz(hand['shoulder_moved_m'])} m."
    )
    if start > reach:
        text += " The target is out of reach even from where the shoulder starts."
    elif now > reach:
        text += " The body's motion carried the shoulder out of reach of the target."
    if "position_error_m" in hand:
        text += (
            f" The closest the arm came: {num(1000 * hand['position_error_m'], 1)} mm from the palm position (5 mm "
            f"allowed), {num(hand['rotation_error_deg'], 1)}° from its orientation "
            f"({num(hand['rotation_tolerance_deg'], 1)}° allowed), {num(100 * hand['minimum_body_clearance_m'], 1)} cm "
            "from the body."
        )
    if hand.get("joints_at_limit"):
        text += f" Joints at the end of their range: {', '.join(j.replace('_', ' ') for j in hand['joints_at_limit'])}."
    return text


def _stance(stance: dict) -> str:
    """The feet before the step: one line of measures and flags."""
    measures = [f"{k.removesuffix('_m')} {num(v)} m" for k, v in stance.items() if k.endswith("_m")]
    flags = [k if v else f"not {k}" for k, v in stance.items() if isinstance(v, bool)]
    rest = [f"{k} {_value(v)}" for k, v in stance.items() if not k.endswith("_m") and not isinstance(v, bool)]
    return ", ".join(measures + flags + rest)


def _attempt(number: int, attempt: dict) -> str:
    """One walker attempt (the request, a stance correction, a retry): what it tried and why it did not pass."""
    kind = attempt.get("kind", "?")
    head = f"{number}. {attempt.get('stage', '?')} ({kind})"
    hidden = _HIDDEN_PARAMS.get(kind, ())  # the harness's own values are not the agent's to change
    shown = {k: v for k, v in (attempt.get("parameters") or {}).items() if k not in hidden and v not in (None, [], {})}
    if shown:
        head += f" with {_value(shown)}"
    if attempt.get("passed"):
        return head + ": passed"
    if attempt.get("error"):
        error = attempt["error"]
        text = head + f": {error.get('reason') or error.get('code', 'error')}, {error.get('message', '')}"
        extra = [f"{k} {_value(v)}" for k, v in error.items() if k not in ("code", "reason", "message")]
        return text + (f" ({', '.join(extra)})" if extra else "")
    findings = ((attempt.get("checks") or {}).get("checks")) or ()
    return head + ": " + ("; ".join(_finding(f, " into this attempt") for f in findings) or "did not pass")


def _details(value: dict, prefix: str = "", depth: int = 0) -> list[str]:
    """Nested fields as ``- path: value`` lines; deep structures as compact JSON."""
    lines = []
    for key, item in value.items():
        path = f"{prefix}{key}"
        if isinstance(item, dict) and item and depth < 2:
            lines += _details(item, path + ".", depth + 1)
        elif isinstance(item, list) and item and all(isinstance(v, dict) for v in item) and depth < 2:
            for index, entry in enumerate(item, 1):
                lines += _details(entry, f"{path}[{index}].", depth + 1)
        elif item not in (None, "", [], {}):
            lines.append(f"- {path}: {_value(item)}")
    return lines


# ----------------------------------------------------------------- execution
def run(run_record: dict, waited_s: float | None = None) -> str:
    state = run_record["state"]
    if state == "done":
        text = "It was executed to the end."
    elif state == "refused":
        text = f"It was not executed: {run_record.get('detail', 'refused')}. The robot has not moved."
    elif state == "lost":
        text = (
            f"Execution was lost ({run_record.get('detail', 'the controller restarted or faulted')}); "
            "check the observation."
        )
    else:
        wait = f" after waiting {num(waited_s, 0)} s" if waited_s is not None else ""
        text = f"The motion is still {state}{wait}; the observation below was taken before it ended."
    if run_record.get("detail") and state in ("done", "queued", "running"):
        text += f"\nNote: {run_record['detail']}."
    return text


def execute(record: dict) -> str:
    """The whole ``execute`` tool result (docs/REFERENCE.md 5): plan verdict, execution, fingers, then the new
    observation. When the first plan was refused for being too fast, ``record`` is the slower plan's view, with the
    first one under ``first``."""
    parts = []
    if record.get("first"):
        parts.append(_slowed(record["first"]))
    if "plan" not in record and "plan_error" not in record:  # fingers and/or the camera only
        if record.get("grip") or record.get("grip_error"):
            parts.append(_grip(record))
    elif record.get("stale"):  # the plan was refused for a stale observation, and the robot observed again
        cause = (
            "the robot's motion reference changed after the last observation was taken"
            if cause_of(record.get("plan_error") or {}) == Cause.REFERENCE_CHANGED
            else "the robot was still finishing its previous motion when the last observation was taken"
        )
        parts.append(
            f"Planning failed (OBSERVATION_STALE): {cause}. Nothing was planned or executed for this call."
            + (
                " The robot has been observed again: plan in the new frame O below."
                if record.get("observation")
                else ""
            )
        )
    elif record.get("plan_error"):
        parts += [failure(record["plan_error"], "Planning"), _unchanged()]
    elif record["plan"]["state"] != "ready":
        parts += [plan_rejected(record["plan"]), _unchanged()]
    else:
        parts.append(plan_ready(record["plan"]))
        if record.get("execute_error"):
            parts.append(failure(record["execute_error"], "Execution"))
        if record.get("run"):
            parts.append(run(record["run"], record.get("waited_s")))
        if record.get("grip") or record.get("grip_error"):
            parts.append(_grip(record))
        elif record.get("grip_skipped"):
            skipped = record["grip_skipped"]
            hands = " and ".join(f"the {side} hand's" for side in skipped["commanded"])
            parts.append(
                f"The motion did not finish, so {hands} fingers were not moved"
                + (" and the wait was not made." if skipped.get("wait_s") else ".")
            )
        refused = record.get("run", {}).get("state") == "refused" or record.get("execute_error")
        if refused and not record.get("observation") and not record.get("observe_error"):
            parts.append(_unchanged())
    if record.get("look"):
        parts.append(_look(record["look"]))
    if record.get("observation"):
        parts.append(
            observation(record["observation"], walked=bool(record.get("walked")), deviation=_deviation(record))
        )
    elif record.get("observe_error"):
        parts.append(failure(record["observe_error"], "Observing"))
    return "\n\n".join(p for p in parts if p)


def _deviation(record: dict) -> dict | None:
    """Each effector's largest distance from its reference over the motion and the fingers' stand-still."""
    rows = [r.get("max_effector_error_m") for r in (record.get("run"), record.get("grip")) if r]
    rows = [r for r in rows if r]
    if not rows:
        return None
    return {name: max(row.get(name, 0.0) for row in rows) for row in rows for name in row}


def _grip(record: dict) -> str:
    """What a finger command did."""
    if record.get("grip_error"):
        error = record["grip_error"]
        text = failure(error, "Moving the fingers")
        if error.get("fingers_commanded"):
            text += "\nThe fingers were commanded, but the stand-still that lets them move was not played."
        if record.get("run"):
            text += "\nThe motion before it was executed."
        return text
    grip = record["grip"]
    done = " and ".join(
        f"the {side} hand's fingers {'closed' if state == 'closed' else 'opened'}"
        for side, state in grip["commanded"].items()
    )
    if grip["state"] != "done":
        return (
            f"The fingers were commanded ({done}), but the stand-still that lets them move did not finish "
            f"({grip['state']})."
        )
    return f"Then {done} (the robot stood still {num(grip.get('hold_s', 1.0), 1)} s while they moved)."


def _look(look: dict) -> str:
    """The camera turned for the observation below (the request's eye)."""
    if "error" in look:
        return failure(look["error"], "Turning the camera")
    return f"Then the camera turned to yaw {look['yaw_deg']}°, pitch {look['pitch_deg']}° down from the torso."


def _slowed(first: dict) -> str:
    """Why the plan was made again, and how much slower."""
    why = failure(first["plan_error"], "Planning") if first.get("plan_error") else plan_rejected(first["plan"])
    return (
        f"The first plan was too fast:\n{why}\n\nIt was planned once more with the height and torso change and the "
        f"walk's swing {SLOWER:g} times slower. What follows is the result of the slower plan."
    )


def _unchanged() -> str:
    return "Nothing was executed and the robot has not moved; the last observation and your measurements in it are still current."


# ---------------------------------------------------------------- perception
def perceived(result: dict) -> str:
    points = result["points"]
    valid = [p for p in points if p.get("valid")]
    text = f"Located {len(points)} point{'s' if len(points) != 1 else ''} on {result['observation_id']}: "
    text += f"{len(valid)} with depth, {len(points) - len(valid)} without."
    if result.get("images"):
        text += " Check the marked image: if a point is not on what you meant, perceive it again."
    lines = [text]
    for number, point in enumerate(points, 1):
        u, v = point["image_point"]
        label = f"image point ({num(u, 0)}, {num(v, 0)})"
        if not point.get("valid"):
            lines.append(f"{number}. {label}: {point.get('reason', 'no depth')}")
            continue
        line = f"{number}. {label} → {xyz(point['xyz'])}"
        spread = point.get("patch_depth_spread_m") or 0.0
        if spread > EDGE_SPREAD_M:
            line += (
                f"; nearby depth varies by {num(spread * 100, 1)} cm. "
                "Check that the marker lies on the intended surface; perspective or an edge can cause this spread"
            )
        lines.append(line)
    lines.append(f"3D positions are in frame O of {result['observation_id']}; they stay valid until the robot moves.")
    return "\n".join(lines)
