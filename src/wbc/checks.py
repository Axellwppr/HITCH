"""Every pass/fail decision on a planned motion, in one place.

Each check reads the motion (plan frame) and thresholds from ``core.limits``.
Blocking checks decide readiness; non-blocking ones are reported as warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core import limits as L
from wbc.core.layout import DT, LEGS, ROOT_POS, ROOT_QUAT, UPPER
from wbc.core.types import MotionDraft
from wbc.helpers.support import hull_margin, support_hull
from wbc.robot.kinematics import Kinematics
from wbc.robot.model import joint_range, lowest_point, robot_model


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool | None  # None: not applicable
    value: float | None = None
    limit: float | None = None
    frame: int | None = None
    blocking: bool = True
    detail: str = ""
    parts: tuple[str, ...] = ()  # for contacts: the two things touching at the worst frame

    def to_dict(self) -> dict:
        return {k: (list(v) if k == "parts" else v) for k, v in self.__dict__.items() if v not in (None, "", ())}


@dataclass
class CheckReport:
    results: list[CheckResult]
    collisions: dict[int, list[str]] = field(default_factory=dict)  # frame -> geom names

    @property
    def passed(self) -> bool:
        return all(r.passed is not False for r in self.results if r.blocking)

    @property
    def failures(self) -> list[str]:
        return [r.name for r in self.results if r.blocking and r.passed is False]

    def findings(self, blocking: bool) -> list[dict]:
        """Checks that did not pass: blocking ones reject the plan, the others are warnings."""
        out = []
        for r in self.results:
            if r.passed is False and r.blocking == blocking:
                out.append({k: v for k, v in r.to_dict().items() if k not in ("passed", "blocking")})
        return out

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "checks": [r.to_dict() for r in self.results],
            "collision_frames": {str(k): v for k, v in sorted(self.collisions.items())[:50]},
        }


def _worst(values, limit, name, blocking=True, detail="") -> CheckResult:
    values = np.asarray(values, float)
    if values.size == 0:
        return CheckResult(name, None, blocking=blocking)
    frame = int(np.argmax(values))
    value = float(values[frame])
    return CheckResult(name, value <= limit, value, limit, frame, blocking, detail)


def check_finite(frames) -> CheckResult:
    finite = bool(np.isfinite(frames).all())
    norms = np.abs(np.linalg.norm(frames[:, ROOT_QUAT], axis=1) - 1) if finite else np.array([np.inf])
    worst = float(norms.max()) if finite else None
    return CheckResult("finite", finite and worst < L.QUATERNION_NORM, worst)


def check_joint_limits(frames, model) -> CheckResult:
    limits = joint_range(model)
    excess = np.maximum(limits[:, 0] - frames[:, 7:], frames[:, 7:] - limits[:, 1]).max(axis=1)
    return _worst(
        np.degrees(np.maximum(excess, 0.0)),
        float(np.degrees(L.JOINT_LIMIT_TOLERANCE)),
        "joint_limits",
        detail="deg beyond range",
    )


def _ratio_worst(values, limits, name, detail) -> CheckResult:
    """Worst value relative to a per-sample limit; infinite limits are not checked."""
    values, limits = np.asarray(values, float), np.asarray(limits, float)
    checked = np.isfinite(limits)
    if not checked.any():
        return CheckResult(name, None)
    ratio = np.where(checked, values / np.where(checked, limits, 1.0), -np.inf)
    frame = int(np.argmax(ratio))
    return CheckResult(name, bool(ratio[frame] <= 1.0), float(values[frame]), float(limits[frame]), frame, True, detail)


@dataclass(frozen=True)
class Limits:
    """Limit profile of every frame: each planned step keeps its own profile."""

    starts: tuple[int, ...]  # first frame of each step
    profiles: tuple[L.LimitProfile, ...]

    def at(self, frames) -> list[L.LimitProfile]:
        index = np.searchsorted(self.starts, np.asarray(frames), side="right") - 1
        return [self.profiles[max(0, i)] for i in np.atleast_1d(index)]

    def per_transition(self, count: int, field: str) -> np.ndarray:
        """Limit for each move k -> k+1 of ``count`` frames: that of the step the move belongs to."""
        return self.per_frame(count, field)[:-1]

    def per_frame(self, count: int, field: str) -> np.ndarray:
        return np.array([getattr(p, field) for p in self.at(np.arange(count))], float)

    @property
    def names(self) -> str:
        return "+".join(dict.fromkeys(p.name for p in self.profiles))


def check_speed(frames, limits: Limits) -> list[CheckResult]:
    speed = np.abs(np.diff(frames, axis=0)) / DT  # move k -> k+1, judged by the step frame k belongs to
    count = len(frames)
    return [
        _ratio_worst(
            np.degrees(speed[:, LEGS].max(axis=1)),
            np.degrees(limits.per_transition(count, "leg_speed")),
            "leg_speed",
            "deg/s",
        ),
        _ratio_worst(
            np.degrees(speed[:, UPPER].max(axis=1)),
            np.degrees(limits.per_transition(count, "upper_speed")),
            "upper_speed",
            "deg/s",
        ),
    ]


def check_root(frames, limits: Limits) -> list[CheckResult]:
    """Pelvis linear and angular speed: a reference must not jump."""
    rotations = Rotation.from_quat(frames[:, ROOT_QUAT], scalar_first=True)
    linear = np.linalg.norm(np.diff(frames[:, ROOT_POS], axis=0), axis=1) / DT
    angular = (rotations[:-1].inv() * rotations[1:]).magnitude() / DT
    count = len(frames)
    return [
        _ratio_worst(linear, limits.per_transition(count, "root_speed"), "root_speed", "m/s"),
        _ratio_worst(np.degrees(angular), np.degrees(limits.per_transition(count, "root_turn")), "root_turn", "deg/s"),
    ]


def check_ik(diagnostics: dict) -> CheckResult:
    if "ik_residual" not in diagnostics:
        return CheckResult("ik_residual", None)
    value = float(diagnostics["ik_residual"])
    worst = diagnostics.get("ik_worst", {})
    return CheckResult(
        "ik_residual", value <= L.IK_RESIDUAL, value, L.IK_RESIDUAL, worst.get("frame"), detail=worst.get("what", "")
    )


def check_geometry(
    motion: MotionDraft, limits: Limits, *, fixed_support=None, fixed_until=None
) -> tuple[list[CheckResult], dict]:
    """One FK pass: stance slip, self collision, ground penetration and balance; with ``fixed_support``, both feet
    stay at its placement up to frame ``fixed_until`` (default: every frame)."""
    kin = Kinematics()
    m, d = kin.model, kin.data
    robot = set(kin.collisions)
    names = {g: m.geom(g).name for g in robot}
    support = motion.support
    slip_limits = limits.per_transition(len(motion.frames), "stance_slip")
    feet_prev = None
    pinned = []
    if fixed_support is not None:
        kin.forward(fixed_support)
        fixed_positions, fixed_rotations = kin.foot_positions(), kin.foot_rotations()
    slip, self_pen, ground, balance = [], [], [], []
    self_pairs = []  # per frame: the deepest pair, or ()
    collisions: dict[int, set] = {}
    for k, pose in enumerate(motion.frames):
        kin.forward(pose)
        feet = kin.foot_positions()
        if fixed_support is not None and (fixed_until is None or k <= fixed_until):
            pinned.append(
                (
                    np.linalg.norm(feet - fixed_positions, axis=1).max(),
                    Rotation.from_matrix(kin.foot_rotations() @ fixed_rotations.transpose(0, 2, 1)).magnitude().max(),
                )
            )
        if support is not None and feet_prev is not None:
            planted = support[k] & support[k - 1]
            slip.append(float((np.linalg.norm(feet - feet_prev, axis=1) / DT)[planted].max()) if planted.any() else 0.0)
        feet_prev = feet
        worst_self, self_pair = 0.0, ()
        for contact in d.contact:
            a, b = int(contact.geom[0]), int(contact.geom[1])
            if a in robot and b in robot and -contact.dist > L.SELF_PENETRATION:
                if -float(contact.dist) > worst_self:
                    worst_self, self_pair = -float(contact.dist), (names[a], names[b])
                collisions.setdefault(k, set()).update((names[a], names[b]))
        self_pen.append(worst_self)
        self_pairs.append(self_pair)
        lowest = min(lowest_point(m, d, g) for g in robot)
        ground.append(max(0.0, -lowest))
        if support is not None and support[k].all():
            balance.append(-hull_margin(support_hull(m, d), kin.com()[:2]))
    results = [
        _ratio_worst(slip, slip_limits, "stance_slip", "m/s of a supporting foot")
        if support is not None
        else CheckResult("stance_slip", None),
        _touching(_worst(self_pen, L.SELF_PENETRATION, "self_collision", detail="m penetration"), self_pairs),
        _worst(ground, L.GROUND_PENETRATION, "ground_penetration", detail="m below ground"),
        _worst(balance, -L.BALANCE_MARGIN, "balance", blocking=False, detail="m CoM outside the support hull")
        if balance
        else CheckResult("balance", None, blocking=False),
    ]
    if pinned:
        values = np.asarray(pinned)
        results += [
            _worst(values[:, 0], L.FIXED_FOOT_POSITION, "fixed_feet_position", detail="m a planted foot moved"),
            _worst(
                np.degrees(values[:, 1]),
                float(np.degrees(L.FIXED_FOOT_ROTATION)),
                "fixed_feet_rotation",
                detail="deg a planted foot turned",
            ),
        ]
    return results, {k: sorted(v) for k, v in collisions.items()}


def _touching(result: CheckResult, pairs: list) -> CheckResult:
    """Name the robot parts touching at the worst frame of a contact check."""
    if result.frame is None or not result.value:
        return result
    return replace(result, parts=tuple(pairs[result.frame]))


def run_checks(motion: MotionDraft, limits: Limits, *, fixed_support=None, fixed_until=None) -> CheckReport:
    frames = motion.frames
    results = [check_finite(frames)]
    if results[0].passed is False:
        return CheckReport(results)
    results += [check_joint_limits(frames, robot_model()), *check_speed(frames, limits), *check_root(frames, limits)]
    results.append(check_ik(motion.diagnostics))
    geometry, collisions = check_geometry(motion, limits, fixed_support=fixed_support, fixed_until=fixed_until)
    return CheckReport(results + geometry, collisions)
