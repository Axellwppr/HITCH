"""Flat-ground walking: footstep schedule -> LIPM CoM preview -> per-frame leg IK.

Root height is a bounded geometric preference. Upper-body joints hold their
start values (the arms are joint-locked during a walk).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import dist

import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation

from wbc.core.errors import invalid
from wbc.core.frames import heading, wrap
from wbc.core.layout import DT, ROOT_QUAT, SIDES, UPPER
from wbc.core.types import MotionDraft, StartState, phases_from_labels
from wbc.helpers.gait import MIN_HEIGHT, PACE, balance_pose, gait_profile, nominal_height
from wbc.helpers.ik import ComXY, JointTarget, RootAttitude, RootHeight, TaskIK, Worst, feet_tasks
from wbc.helpers.interp import HOLD_S, lipm, slerp, smooth
from wbc.helpers.support import ANKLE_HEIGHT, toe_lift

from .footsteps import STANCE_WIDTH_M, WalkGoal, automatic, turning
from .stance import Stance

SETTLE_S = 0.36  # active re-centring after the last step
PATH_TOLERANCE_M = 1e-3  # a straight walk's footstep path exceeds its root distance by round-off
MAX_PATH_M = 2.0
MAX_DURATION_S = 20.0
FORWARD_ADVANCE_GAIN = 0.35  # longer standing strides; cadence preserves swing acceleration below
TURN_TIME_SCALE = 0.60  # short-arc standing turns


@dataclass
class Timeline:
    """Per-frame targets accumulated while walking through the step schedule."""

    feet: np.ndarray
    rotations: np.ndarray
    zmp: np.ndarray
    yaw: float = 0.0
    samples: list = field(default_factory=list)
    zmps: list = field(default_factory=list)
    support: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    yaws: list = field(default_factory=list)

    def phase(self, seconds, label, zmp_goal, *, swing=None, yaw_goal=None):
        n = max(2, round(seconds / DT))
        before, before_r, yaw0 = self.feet.copy(), self.rotations.copy(), self.yaw
        rotate = None
        if swing is not None:
            rotate = slerp(
                Rotation.from_matrix(before_r[swing.side]).as_quat(scalar_first=True),
                Rotation.from_matrix(swing.rotation).as_quat(scalar_first=True),
            )
        for k in range(n):
            t = k / n
            u = smooth(t)
            feet, rotations = before.copy(), before_r.copy()
            contact = np.ones(2, bool)
            if swing is not None:
                feet[swing.side], rotations[swing.side] = swing.at(t, u, before[swing.side], rotate)
                contact[swing.side] = False
            self.samples.append((feet, rotations))
            self.zmps.append((1 - u) * self.zmp + u * np.asarray(zmp_goal))
            self.support.append(contact)
            self.labels.append(label)
            self.yaws.append(yaw0 if yaw_goal is None else (1 - u) * yaw0 + u * yaw_goal)
        self.zmp = np.asarray(zmp_goal, float)
        if swing is not None:
            self.feet[swing.side] = swing.destination
            self.rotations[swing.side] = swing.rotation
        if yaw_goal is not None:
            self.yaw = yaw_goal


@dataclass
class Swing:
    side: int
    destination: np.ndarray
    rotation: np.ndarray
    height: float
    pitch: float
    toe_compensation: bool

    def at(self, t, u, origin, rotate):
        lift = 64 * t**3 * (1 - t) ** 3
        position = (1 - u) * origin + u * self.destination
        position[2] += self.height * lift
        pitch = self.pitch * lift
        if self.toe_compensation:
            position[2] += toe_lift(pitch)
        rotation = (
            Rotation.from_quat(rotate(u), scalar_first=True).as_matrix() @ Rotation.from_euler("y", pitch).as_matrix()
        )
        return position, rotation


class LegSolver:
    """Feet and CoM dominate a weak nominal-height preference; upper joints fixed."""

    def __init__(self, reference):
        self.knee_reference = reference[[10, 16]].copy()
        self.ik = TaskIK(joints=range(12), root="xyz_tilt")  # the walker's own checks include its balance arms
        self.kin = self.ik.kin
        # Triangle inequality along the actual root-to-foot chain. This is a
        # loose mechanical ceiling, not a tuned gait height or reach promise.
        self.leg_reach = 0.0
        body = self.kin.feet[0]
        while body != self.kin.pelvis:
            self.leg_reach += float(np.linalg.norm(self.kin.model.body_pos[body]))
            body = int(self.kin.model.body_parentid[body])
        self.height_bounds = (MIN_HEIGHT, ANKLE_HEIGHT + self.leg_reach)

    def solve(self, pose, feet, rotations, com_xy, height, attitude=None):
        tasks = feet_tasks(feet, rotations) + [
            ComXY("com", 1.0, com_xy),
            RootHeight("height", 0.01, height),
            # Preserve the input knee flexion as a weak posture preference.
            # Half a leg length converts angular deviation to a length scale.
            # This avoids selecting the straight-knee singularity merely to
            # maximize height; the preference uses the input pose alone.
            JointTarget(
                "knee_posture",
                0.01 * self.leg_reach / 2,
                np.array([3, 9]),
                self.knee_reference,
            ),
            RootAttitude(
                "attitude",
                1.0,
                pose[ROOT_QUAT].copy() if attitude is None else attitude,
                (0.02, 0.02, 1.0),
            ),
        ]
        # Height has 1e-4 the squared task cost. Feet/CoM remain independently
        # checked; height gives way when a split stance exhausts leg reach.
        result = self.ik.solve(
            pose,
            tasks,
            height=height,
            joint_step=0.06,
            root_step=0.06,
            height_bounds=(self.height_bounds[0], min(height, self.height_bounds[1])),
        )
        return result.pose, result.worst(("feet", "com"))


def step_motion(start: StartState, p: WalkGoal):
    # Bound a root request before automatic() constructs its footstep list.
    # The later path check also catches detours and explicit multi-step paths.
    if p.target is not None:
        distance = dist(p.target, start.pose[:2])
        if distance > MAX_PATH_M:
            raise invalid(
                "root destination exceeds the per-request walking range",
                reason="REQUEST_LIMIT",
                limit="root_distance_m",
                requested=distance,
                maximum=MAX_PATH_M,
            )
    height = nominal_height(start.height)
    gait = gait_profile(height)
    base_advance = gait.forward_advance
    in_place = (
        p.target is not None
        and dist(p.target, start.pose[:2]) < 1e-6
        and abs(p.turn) > 1e-6
        and not gait.crouched
        and Stance.read(start).resting
    )
    forward = 0.0
    # Combined translation/yaw uses the standard placement spacing. Its
    # stance reach depends on both heading change and translation per landing.
    if p.target is not None and abs(p.turn) <= 1e-6:
        delta = np.asarray(p.target) - start.pose[:2]
        forward = max(0.0, delta[0]) / max(float(np.linalg.norm(delta)), 1e-9)
        upright = float(np.interp(height, [PACE[0][0], PACE[-1][0]], [0.0, 1.0]))
        gait = replace(gait, forward_advance=base_advance * (1 + FORWARD_ADVANCE_GAIN * upright * forward**2))
    if in_place:
        gait = replace(gait, swing_s=gait.swing_s * TURN_TIME_SCALE, transfer_s=gait.transfer_s * TURN_TIME_SCALE)
    swing_s, transfer_s, inset, width = (
        p.swing_s or gait.swing_s,
        gait.transfer_s,
        gait.support_inset,
        STANCE_WIDTH_M,
    )
    solver = LegSolver(reference=start.pose)
    kin = solver.kin
    kin.forward(start.pose)
    timeline = Timeline(
        kin.foot_positions(),
        kin.foot_rotations(),
        kin.com()[:2].copy(),
        heading(start.pose[ROOT_QUAT]),
    )
    com_height = float(kin.com()[2])
    origin = timeline.feet[:, :2].mean(axis=0)
    placement = turning if in_place else automatic
    steps = list(p.footsteps) or placement(start, np.asarray(p.target), p.turn, gait, width, solver, origin)
    pitch = gait.swing_pitch
    max_swing = 2 * solver.leg_reach
    toe = gait.crouched
    # Preflight the complete schedule before allocating per-frame targets/IK.
    path = 0.0
    last_mid = origin.copy()
    placed = timeline.feet[:, :2].copy()
    forward_travel = 0.0
    for number, step in enumerate(steps):
        side = SIDES.index(step.side)
        distance = dist(step.xy, placed[side])
        if distance > max_swing:
            raise invalid(
                f"footstep {number} travels more than {max_swing} m",
                reason="TARGET_NOT_REACHED",
                step_index=number,
                side=step.side,
                distance_m=distance,
                mechanical_ceiling_m=max_swing,
            )
        forward_travel = max(forward_travel, step.xy[0] - placed[side, 0])
        placed[side] = step.xy
        mid = placed.mean(axis=0)
        path += float(np.linalg.norm(mid - last_mid))
        last_mid = mid
    if path > MAX_PATH_M + PATH_TOLERANCE_M:
        raise invalid(
            "walk path exceeds the per-request walking range",
            reason="REQUEST_LIMIT",
            limit="path_m",
            requested=path,
            maximum=MAX_PATH_M,
        )
    if p.swing_s is None and forward > 0:
        # Quintic swing acceleration scales as distance / time**2. Longer
        # forward swings keep that scale; short walks retain their usual cadence.
        swing_s *= max(1.0, np.sqrt(forward_travel / (2 * base_advance)))
    duration = DT * (
        len(steps) * (max(2, round(transfer_s / DT)) + max(2, round(swing_s / DT)))
        + max(2, round(SETTLE_S / DT))
        + max(2, round(HOLD_S / DT))
    )
    if duration > MAX_DURATION_S:
        raise invalid(
            "walk exceeds the per-request duration; split it into shorter walks",
            reason="REQUEST_LIMIT",
            limit="duration_s",
            requested=duration,
            maximum=MAX_DURATION_S,
        )
    # Each foot's yaw relative to the pelvis when it last landed: a footstep without yaw keeps it.
    angle_to_pelvis = [Rotation.from_matrix(r).as_euler("xyz")[2] - timeline.yaw for r in timeline.rotations]
    for step in steps:
        side = SIDES.index(step.side)
        stance = 1 - side
        xy = np.asarray(step.xy, float)
        root_goal = timeline.yaw if step.root_yaw is None else timeline.yaw + float(wrap(step.root_yaw - timeline.yaw))
        foot_yaw = root_goal + angle_to_pelvis[side] if step.yaw is None else step.yaw
        angle_to_pelvis[side] = foot_yaw - root_goal
        stance_yaw = Rotation.from_matrix(timeline.rotations[stance]).as_euler("xyz")[2]
        offset = Rotation.from_euler("z", stance_yaw).apply([gait.forward_balance, (2 * stance - 1) * inset, 0])[:2]
        zmp = timeline.feet[stance, :2] + offset
        swing = Swing(
            side,
            np.r_[xy, ANKLE_HEIGHT],
            Rotation.from_euler("z", foot_yaw).as_matrix(),
            p.swing_height_m,
            pitch,
            toe,
        )
        timeline.phase(transfer_s, "transfer", zmp)
        timeline.phase(
            swing_s,
            f"swing_{step.side}",
            zmp,
            swing=swing,
            yaw_goal=root_goal,
        )
    centre = (
        timeline.feet[:, :2].mean(axis=0)
        + Rotation.from_euler("z", timeline.yaw).apply([gait.forward_balance, 0, 0])[:2]
    )
    timeline.phase(SETTLE_S, "settle", centre)
    timeline.phase(HOLD_S, "hold", timeline.zmp)
    com_xy = preview_com(timeline, timeline.zmps[0], com_height, transfer_s)
    frames, residual, com_path = _solve_frames(start, timeline, com_xy, gait, solver, transfer_s)
    support = np.asarray(timeline.support)
    diagnostics = {
        **residual.diagnostics(),
        "path_m": path,
        "com_lateral_range_m": float(np.ptp(com_path[:, 1])),
        "root_height_min_m": float(frames[:, 2].min()),
        "root_height_max_m": float(frames[:, 2].max()),
        "root_height_nominal_m": height,
        "gait_mode": "turn" if in_place else "walk",
        "swing_s": round(swing_s / DT) * DT,
        "transfer_s": round(transfer_s / DT) * DT,
        "footsteps": [dict(side=s.side, xy=list(s.xy), yaw=s.yaw, root_yaw=s.root_yaw) for s in steps],
    }
    if p.target is not None:
        diagnostics["target_error_m"] = float(np.linalg.norm(frames[-1, :2] - np.asarray(p.target)))
    return MotionDraft(
        frames,
        phases_from_labels(timeline.labels),
        start.posture,
        support=support,
        diagnostics=diagnostics,
    )


def preview_com(timeline: Timeline, initial_com, com_height, transfer_s):
    """LIPM CoM preview with smooth (zero velocity/acceleration) entry and exit."""
    zmp = np.asarray(timeline.zmps)
    xy = lipm(zmp, com_height, DT)
    first = min(round(transfer_s / DT), len(xy) - 2)
    velocity = (xy[first + 1] - xy[first - 1]) / (2 * DT)
    accel = (xy[first + 1] - 2 * xy[first] + xy[first - 1]) / DT**2
    entry = BPoly.from_derivatives(
        [0, first * DT],
        [[initial_com, np.zeros(2), np.zeros(2)], [xy[first], velocity, accel]],
    )
    xy[: first + 1] = entry(np.arange(first + 1) * DT)
    settle = timeline.labels.index("settle")
    hold = timeline.labels.index("hold")
    velocity = (xy[settle] - xy[settle - 2]) / (2 * DT)
    accel = (xy[settle] - 2 * xy[settle - 1] + xy[settle - 2]) / DT**2
    finish = BPoly.from_derivatives(
        [0, (hold - settle) * DT],
        [[xy[settle - 1], velocity, accel], [zmp[-1], np.zeros(2), np.zeros(2)]],
    )
    xy[settle:hold] = finish(np.arange(1, hold - settle + 1) * DT)
    xy[hold:] = zmp[-1]
    return xy


def _solve_frames(start, timeline, com_xy, gait, solver, transfer_s):
    height = nominal_height(start.height)
    count = len(timeline.samples)
    heights = np.full(count, height)
    neutral = slerp(
        start.pose[ROOT_QUAT],
        Rotation.from_euler("y", balance_pose(height)[0]).as_quat(scalar_first=True),
    )
    pose = start.pose.copy()
    frames, coms = [start.pose.copy()], []
    solver.kin.forward(start.pose)
    coms.append(solver.kin.com())
    residual = Worst()
    for i in range(1, count):
        feet, rotations = timeline.samples[i]
        yaw = timeline.yaws[i]
        lateral = Rotation.from_euler("z", -yaw).apply(np.r_[com_xy[i] - feet[:, :2].mean(axis=0), 0.0])[1]
        roll = -np.clip(lateral * gait.roll_gain, -0.3, 0.3) * smooth(i * DT / transfer_s)
        pose[UPPER] = start.pose[UPPER]
        attitude = (
            Rotation.from_euler("z", yaw)
            * Rotation.from_quat(neutral(smooth(i * DT / 0.8)), scalar_first=True)
            * Rotation.from_euler("x", roll)
        ).as_quat(scalar_first=True)
        pose, error = solver.solve(pose, feet, rotations, com_xy[i], heights[i], attitude)
        residual.add(error, i)
        frames.append(pose.copy())
        coms.append(solver.kin.com())
    return np.asarray(frames), residual, np.asarray(coms)
