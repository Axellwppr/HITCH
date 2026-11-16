"""What each hand does over the in-place body (docs/REFERENCE.md 3.3): its path, compiled once from the request, and
the explanation of a hand that could not be planned.

A path gives the arm solver (``helpers.arm_path``) a target for every frame of the body: a palm pose in the pelvis
heading frame (``PalmPlan``), or arm joint angles (``JointPlan``). Waypoints live in the plan's observation frame O,
which stays fixed for the whole plan, and start with the body: the palm moves in O whatever the body does. A hand
that also has ``follow`` moves with the body first (held in the pelvis heading frame, in O, or on its joint angles)
while the body changes, and takes its waypoints from where that leaves it. A waypoint without an orientation keeps the
one before it; the first keeps the hand's intent (``StartState.hands``: what its requests last asked for, not what the
arm reached). The palm point is the ``*_palm`` site (``wbc.robot.hands``): 0.08 m along the fingers from the wrist-yaw
frame on the rubber hand, 0.09 m on the Dex3.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core.errors import WbcError
from wbc.core.layout import DT, SIDES, arm
from wbc.core.request import HandCommand
from wbc.core.types import MotionDraft
from wbc.helpers.arm_ik import FOLLOW_ROTATION_TOL, POSTURE_WEIGHT, ROTATION_TOL
from wbc.helpers.interp import BLEND_S, HOLD_S, frames_for, smooth
from wbc.robot.arm import NEAR_REST, ArmGeometry, arm_is_hanging, arm_is_home, nominal_arm
from wbc.robot.kinematics import Kinematics, body_rotation


@dataclass
class JointPlan:
    """Arm joint angles blending from ``start`` to ``end`` over ``seconds``; ``kind`` rest lets the solver move the arm
    off them to clear the body (an arm at its rest posture)."""

    start: np.ndarray
    end: np.ndarray
    seconds: float
    kind: str = "joints"

    def at(self, k, pose):
        u = smooth(min(1.0, k * DT / self.seconds))
        return self.start + u * (self.end - self.start)


class Piece(NamedTuple):
    """One interval of a palm path. ``frame`` "O": palm positions and rotation matrices in O moving from p0, r0 to
    p1, r1; "root": p0, r0 held in the pelvis heading frame; "joints": the palm the arm joint angles p0 put on the
    body (arm index p1)."""

    first: int  # first frame
    count: int  # frames
    p0: object
    p1: object
    r0: object
    r1: object
    frame: str


@dataclass
class PalmPlan:
    kind = "pose"
    pieces: list[Piece]
    rotation_tolerance: float = ROTATION_TOL  # reaching allows 60 degrees; holding a pose, far less
    moves: bool = True  # its pieces are intervals of a path (waypoints); False: one held pose, no intervals
    follow_until: int = 0  # frames of a ``follow`` before the waypoints, held to the following tolerance

    def tolerance(self, k) -> float:
        return FOLLOW_ROTATION_TOL if k < self.follow_until else self.rotation_tolerance

    def at(self, k, pose):
        index = next((i for i, s in enumerate(self.pieces) if k <= s.first + s.count), len(self.pieces) - 1)
        first, n, p0, p1, r0, r1, frame = self.pieces[index]
        if frame == "root":
            return p0, r0
        if frame == "joints":
            return _palm_on_joints(pose, p1, p0)
        heading = body_rotation(pose)
        u = smooth(np.clip((k - first) / n, 0.0, 1.0))
        position, rotation = (1 - u) * p0 + u * p1, _slerp(r0, r1, u)
        return heading.T @ (position - pose[:3]), heading.T @ rotation

    @property
    def waypoints(self) -> list:
        """The pieces of the waypoints (after a ``follow``)."""
        return [piece for piece in self.pieces if piece.first >= self.follow_until and self.moves]


@cache
def _kinematics() -> Kinematics:
    return Kinematics()


def _palm_on_joints(pose, index, joints):
    """The palm (pelvis heading frame) that arm joint angles put on the body ``pose``."""
    held = np.array(pose, float)
    held[arm(SIDES[index])] = joints
    return _kinematics().palm_in_body(held, index)


def _slerp(r0, r1, u) -> np.ndarray:
    """The rotation matrix a fraction ``u`` of the way from ``r0`` to ``r1``."""
    if u <= 0.0 or np.array_equal(r0, r1):
        return np.asarray(r0)
    if u >= 1.0:
        return np.asarray(r1)
    start = Rotation.from_matrix(r0)
    return (start * Rotation.from_rotvec(u * (start.inv() * Rotation.from_matrix(r1)).as_rotvec())).as_matrix()


@dataclass
class ArmPlan:
    """The body reference (held long enough for every path) and each arm's path over it."""

    body: MotionDraft
    paths: dict
    posture_weight: float = POSTURE_WEIGHT


def compile_hands(
    body: MotionDraft, commands: dict[str, HandCommand], body_end: int | None = None, intents: dict | None = None
) -> ArmPlan:
    """Each hand's path over ``body`` from its first pose (docs/REFERENCE.md 3.3): waypoints (after a ``follow`` while
    the body changes, up to frame ``body_end``), home, or how it follows the body; a hand given none of these rests if
    its arm is at its rest posture, keeps its joint angles if it still hangs as at the start, and otherwise follows
    root. ``intents`` (StartState.hands): the orientation a first waypoint without one keeps."""
    pose = body.frames[0]
    heading = body_rotation(pose)
    paths, needed = {}, 0
    for index, side in enumerate(SIDES):
        command = commands[side]
        joints = pose[arm(side)].copy()
        in_body, rotation_in_body = _kinematics().palm_in_body(pose, index)  # the pelvis heading frame
        position, rotation = pose[:3] + heading @ in_body, heading @ rotation_in_body  # O
        resting = arm_is_home(pose, side, NEAR_REST)  # at its rest posture, or making room from there
        if command.home or (not command.waypoints and command.follow is None and resting):
            paths[side] = JointPlan(joints, nominal_arm(side), BLEND_S, "rest")
            if np.max(np.abs(joints - nominal_arm(side))) > 1e-6:
                needed = max(needed, frames_for(BLEND_S))
        elif command.waypoints:
            pieces, frame = [], 0
            if command.follow is not None and body_end:  # with the body first, then the waypoints from there
                end = body.frames[body_end]
                if command.follow == "root":
                    pieces.append(Piece(0, body_end, in_body, in_body, rotation_in_body, rotation_in_body, "root"))
                    there, turn = in_body, rotation_in_body
                elif command.follow == "world":
                    pieces.append(Piece(0, body_end, position, position, rotation, rotation, "O"))
                    there, turn = None, None
                else:
                    pieces.append(Piece(0, body_end, joints, index, None, None, "joints"))
                    there, turn = _palm_on_joints(end, index, joints)
                if there is not None:
                    position, rotation = end[:3] + body_rotation(end) @ there, body_rotation(end) @ turn
                frame = body_end
            follow_until = frame
            intent = (intents or {}).get(side)
            goal_rotation = rotation if intent is None else np.asarray(intent["rotation"], float)
            previous, previous_rotation = position, rotation
            for waypoint in command.waypoints:
                if waypoint.orientation is not None:
                    goal_rotation = waypoint.orientation.matrix(side)
                goal, count = np.asarray(waypoint.position, float), frames_for(waypoint.duration_s)
                pieces.append(Piece(frame, count, previous, goal, previous_rotation, goal_rotation, "O"))
                frame += count
                previous, previous_rotation = goal, goal_rotation
            paths[side] = PalmPlan(pieces, follow_until=follow_until)
            needed = max(needed, frame)
        elif command.follow is None and arm_is_hanging(pose, side):  # hanging as at the start: it swings along
            paths[side] = JointPlan(joints, joints, BLEND_S, "joints")
        elif (command.follow or "root") == "root":
            piece = Piece(0, 1, in_body, in_body, rotation_in_body, rotation_in_body, "root")
            paths[side] = PalmPlan([piece], rotation_tolerance=FOLLOW_ROTATION_TOL, moves=False)
        elif command.follow == "world":
            piece = Piece(0, 1, position, position, rotation, rotation, "O")
            paths[side] = PalmPlan([piece], rotation_tolerance=FOLLOW_ROTATION_TOL, moves=False)
        else:  # body: the joint angles stay
            paths[side] = JointPlan(joints, joints, BLEND_S, "joints")
    return ArmPlan(body.held(needed + 1 + frames_for(HOLD_S) - len(body.frames)), paths)


def end_intents(plan: ArmPlan, motion: MotionDraft, intents: dict | None) -> dict:
    """Each hand's intent where the in-place ``motion`` (planned from ``plan``) ends, in the plan frame: its last
    waypoint as asked; a hand following root carries its intent with the pelvis, one following world keeps it; a hand
    on joint angles (home, rest, body, hanging) has none (its reference pose is its intent)."""
    start, end = motion.frames[0], motion.frames[-1]
    out = {}
    for side, path in plan.paths.items():
        intent = (intents or {}).get(side)
        if path.kind != "pose":
            continue
        if path.moves:
            last = path.pieces[-1]
            out[side] = {"position": np.asarray(last.p1, float).tolist(), "rotation": np.asarray(last.r1).tolist()}
        elif intent is not None and path.pieces[0].frame == "root":
            carry = body_rotation(end) @ body_rotation(start).T
            position = end[:3] + carry @ (np.asarray(intent["position"], float) - start[:3])
            out[side] = {
                "position": position.tolist(),
                "rotation": (carry @ np.asarray(intent["rotation"], float)).tolist(),
            }
        elif intent is not None:
            out[side] = intent
    return out


def explain(error: WbcError, plan: ArmPlan, commands: dict[str, HandCommand]) -> None:
    """Add to a hand failure what each failing hand was doing, where its target was against its shoulder at the
    start and at that moment, how far the body had carried the shoulder, and what stopped the arm."""
    frame = error.details.get("frame")
    # after retiming (stage hand_timing) frames are of the stretched clock, and a height search fails on a trial body:
    # either way the planned body is not the one that failed
    if frame is None or frame >= len(plan.body.frames) or error.details.get("stage") == "hand_timing":
        return
    if "height_adjustment" in error.details:
        return
    pose, pose0 = plan.body.frames[int(frame)], plan.body.frames[0]
    arms = error.details.get("arms") or {}
    if error.details.get("reason") == "HAND_REACH_LIMIT" and arms:
        sides = [s for s, row in arms.items() if row.get("distance_m", 0.0) > row.get("absolute_radius_m", np.inf)]
    elif arms:
        sides = list(arms)
    elif error.details.get("side") in SIDES:
        sides = [error.details["side"]]
    else:
        sides = [s for s in SIDES if plan.paths[s].kind == "pose"]
    explained = {}
    for side in sides:
        command, path = commands[side], plan.paths[side]
        if command.waypoints and frame < path.follow_until:
            doing = f"follow {command.follow} while the body changes, before its waypoints"
        elif command.waypoints:
            pieces = path.waypoints
            number = next((i for i, piece in enumerate(pieces, 1) if frame <= piece.first + piece.count), None)
            doing = (
                f"waypoint {number} of {len(pieces)}"
                if number is not None
                else f"holding its last waypoint ({len(pieces)})"
            )
        elif path.kind == "rest":
            moving = np.max(np.abs(path.start - path.end)) > 1e-6 and frame * DT < path.seconds
            doing = "moving to its rest posture" if moving else "at its rest posture"
        elif path.kind == "joints" and command.follow is None:
            doing = "hanging as at the start (keeping its joint angles)"
        else:
            doing = f"follow {command.follow or 'root'}"
        entry = {"doing": doing}
        geometry = ArmGeometry(side, Kinematics())
        moved = _shoulder(geometry, pose) - _shoulder(geometry, pose0)
        entry["shoulder_moved_m"] = [round(float(v), 3) for v in moved]
        if path.kind == "pose":
            target_body, _ = path.at(int(frame), pose)
            target = pose[:3] + body_rotation(pose) @ target_body  # O
            at_start = body_rotation(pose0).T @ (target - pose0[:3])
            entry["target_O_m"] = [round(float(v), 3) for v in target]
            entry["distance_at_start_m"] = round(geometry.reach(pose0, at_start)["distance_m"], 3)
            entry["distance_now_m"] = round(geometry.reach(pose, target_body)["distance_m"], 3)
            entry["reach_m"] = round(geometry.max_reach, 3)
        for key in (
            "position_error_m",
            "rotation_error_deg",
            "rotation_tolerance_deg",
            "minimum_body_clearance_m",
            "joints_at_limit",
        ):
            if key in arms.get(side, {}):
                value = arms[side][key]
                entry[key] = value if isinstance(value, list) else round(float(value), 4)
        explained[side] = entry
    error.details["hands"] = explained
    error.details.setdefault("body_height_m", round(float(pose[2]), 3))
    error.details.setdefault("time_s", round(float(frame) * DT, 2))


def _shoulder(geometry: ArmGeometry, pose) -> np.ndarray:
    """The shoulder joint in O."""
    return pose[:3] + body_rotation(pose) @ geometry.shoulder_in_body(pose)
