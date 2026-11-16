"""Weighted task-space IK, shared by every generator; each step is a mink QP.

A solve combines task terms (feet, CoM, palms, joint targets, root attitude)
into one least-squares objective per iteration, under joint limits, a step
bound and self-collision avoidance (``L.SELF_COLLISION_MARGIN`` between robot
parts). Root height may be a bounded preference when Z is enabled; root degrees
of freedom are chosen per solve.

Every result breaks its residual down by task part ("left palm rotation", ...),
so a skill can report where and when its motion fell short (``Worst``).
"""

from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core import limits as L
from wbc.core.layout import SIDES
from wbc.helpers.ik_core import advance
from wbc.robot.kinematics import Kinematics
from wbc.robot.model import descendants, self_collision_pairs

# Columns of the floating base in qvel: x, y, z, roll, pitch, yaw.
ROOT_DOFS = {
    "none": (),
    "x": (0,),
    "xy": (0, 1),
    "xy_tilt": (0, 1, 3, 4, 5),
    "xyz": (0, 1, 2),
    "xyz_tilt": (0, 1, 2, 3, 4, 5),
}
COLLISION_DETECTION_M = 0.05  # pairs closer than this constrain the step
# Each linearised step lets a pair at the margin creep slightly closer; a small negative relaxation
# makes such pairs back off by 1 mm per step instead of drifting into each other.
COLLISION_RELAXATION = -0.001


@dataclass(frozen=True)
class Part:
    """One part of a task's residual: ``value`` is what the IK threshold judges, ``error`` its physical size."""

    task: str
    what: str
    value: float
    error: float
    unit: str

    @property
    def description(self) -> str:
        error, unit = (np.degrees(self.error), "deg") if self.unit == "rad" else (self.error, self.unit)
        return f"{self.what} off by {error:.3f} {unit}".rstrip()


@dataclass
class Task:
    """One residual block. Subclasses implement ``evaluate`` (target - current, and its Jacobian)."""

    name: str
    weight: float = 1.0

    def evaluate(self, ik: TaskIK) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def parts(self, residual: np.ndarray) -> list[Part]:
        value = float(np.max(np.abs(residual)))
        return [Part(self.name, self.name, value, value, "")]

    def _split(self, residual, what: str, rotation_scale: float = 1.0) -> list[Part]:
        position, rotation = residual[:3], residual[3:]
        return [
            Part(
                self.name,
                f"{what} position",
                float(np.max(np.abs(position))),
                float(np.linalg.norm(position)),
                "m",
            ),
            Part(
                self.name,
                f"{what} rotation",
                float(np.max(np.abs(rotation))),
                float(np.linalg.norm(rotation)) / rotation_scale,
                "rad",
            ),
        ]


@dataclass
class FootPose(Task):
    side: int = 0
    position: np.ndarray | None = None
    rotation: np.ndarray | None = None

    def evaluate(self, ik):
        m, d = ik.kin.model, ik.kin.data
        body = ik.kin.feet[self.side]
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jp, jr, body)
        r_pos = self.position - d.xpos[body]
        r_rot = Rotation.from_matrix(self.rotation @ d.xmat[body].reshape(3, 3).T).as_rotvec()
        return np.r_[r_pos, r_rot], np.vstack([jp, jr])

    def parts(self, residual):
        return self._split(residual, f"{SIDES[self.side]} foot")


@dataclass
class ComXY(Task):
    target: np.ndarray | None = None
    axes: tuple[int, ...] = (0, 1)

    def evaluate(self, ik):
        m, d = ik.kin.model, ik.kin.data
        jac = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jac, ik.kin.pelvis)
        axes = list(self.axes)
        target = np.asarray(self.target, float)[: len(axes)]
        return target - d.subtree_com[ik.kin.pelvis, axes], jac[axes]

    def parts(self, residual):
        return [
            Part(
                self.name,
                "CoM",
                float(np.max(np.abs(residual))),
                float(np.linalg.norm(residual)),
                "m",
            )
        ]


@dataclass
class JointTarget(Task):
    """Pull joints (canonical indices 0..28) toward values."""

    joints: np.ndarray | None = None
    values: np.ndarray | None = None

    def evaluate(self, ik):
        m, d = ik.kin.model, ik.kin.data
        joints = np.asarray(self.joints, int)
        jac = np.zeros((len(joints), m.nv))
        jac[np.arange(len(joints)), 6 + joints] = 1.0
        return np.asarray(self.values, float) - d.qpos[7 + joints], jac

    def parts(self, residual):
        value = float(np.max(np.abs(residual)))
        return [Part(self.name, f"{self.name} joints", value, value, "rad")]


@dataclass
class RootAttitude(Task):
    quaternion: np.ndarray | None = None
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def evaluate(self, ik):
        m, d = ik.kin.model, ik.kin.data
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jp, jr, ik.kin.pelvis)
        error = (
            Rotation.from_quat(self.quaternion, scalar_first=True)
            * Rotation.from_quat(d.qpos[3:7], scalar_first=True).inv()
        ).as_rotvec()
        w = np.asarray(self.axis_weights)
        return w * error, w[:, None] * jr

    def parts(self, residual):
        value = float(np.max(np.abs(residual)))
        return [Part(self.name, "pelvis attitude", value, value, "rad")]


@dataclass
class RootHeight(Task):
    target: float = 0.0

    def evaluate(self, ik):
        jac = np.zeros((1, ik.kin.model.nv))
        jac[0, 2] = 1
        return np.array([self.target - ik.kin.data.qpos[2]]), jac


class HeightLimit(mink.Limit):
    """Exact linear bound on the floating base Z, which joint limits omit."""

    def __init__(self, nv, lower, upper):
        self.lower, self.upper = lower, upper
        self.projection = np.zeros((2, nv))
        self.projection[:, 2] = [1, -1]

    def compute_qp_inequalities(self, configuration, dt):
        z = configuration.data.qpos[2]
        return mink.Constraint(self.projection, np.array([self.upper - z, z - self.lower]))


@dataclass
class IKResult:
    pose: np.ndarray
    parts: list[Part]

    def worst(self, tasks: tuple[str, ...]) -> Part | None:
        """The largest residual part among the named tasks."""
        chosen = [p for p in self.parts if p.task in tasks]
        return max(chosen, key=lambda p: p.value, default=None)


@dataclass
class Worst:
    """The largest IK residual over a motion, with where and when it happened."""

    value: float = 0.0
    what: str = ""
    frame: int = 0

    def add(self, part: Part | None, frame: int) -> None:
        if part is not None and part.value > self.value:
            self.value, self.frame = part.value, frame
            self.what = part.description

    def merge(self, diagnostics: dict, offset: int = 0) -> None:
        """Take a draft's worst (frames shifted by ``offset``) if it is larger."""
        value = float(diagnostics.get("ik_residual", 0.0))
        if value > self.value:
            worst = diagnostics.get("ik_worst", {})
            self.value, self.what, self.frame = (
                value,
                worst.get("what", ""),
                offset + int(worst.get("frame", 0)),
            )

    def diagnostics(self) -> dict:
        out: dict = {"ik_residual": self.value}
        if self.what:
            out["ik_worst"] = {"what": self.what, "frame": self.frame}
        return out


class _MinkTask(mink.Task):
    """One of our tasks as a mink least-squares term, evaluated once per iteration by the solve loop."""

    def __init__(self, task: Task):
        super().__init__(cost=np.ones(1))
        self.task = task
        self.residual = self.jacobian = None

    def compute_qp_residual(self, configuration):
        w = self.task.weight
        return w * self.jacobian, w * self.residual, 0.0

    def compute_error(self, configuration):
        return -self.residual

    def compute_jacobian(self, configuration):
        return self.jacobian


class TaskIK:
    def __init__(
        self,
        kin: Kinematics | None = None,
        *,
        joints=range(12),
        root: str = "xy",
        margin: float = 1e-5,
        collision_avoidance: bool = True,
        placeholder_arms: bool = False,
    ):
        """``placeholder_arms``: the arms in the poses solved are a stand-in for their mass (the in-place body is
        planned with nominal arms and the real ones are solved afterwards, making room for the body), so the arms are
        left out of collision avoidance: a stand-in forearm beside a crouching knee must not push the legs away. Only
        for callers whose own checks do not include those arms."""
        self.kin = kin or Kinematics()
        m = self.kin.model
        self.joints = np.asarray(list(joints), int)
        self.root_dofs = np.asarray(ROOT_DOFS[root], int)
        self.cols = np.r_[self.root_dofs, 6 + self.joints].astype(int)
        frozen = sorted(set(range(m.nv)) - set(self.cols.tolist()))
        self.frozen = [mink.DofFreezingTask(m, frozen)]
        self.configuration = mink.Configuration(m)
        self.configuration.data = self.kin.data  # tasks and limits read the same data
        self.joint_limits = mink.ConfigurationLimit(m, min_distance_from_limits=margin)
        if 2 in self.root_dofs:
            # Use the forward-knee branch; free Z must not pass through a straight
            # knee and switch to the model's small hyperextension allowance.
            for joint in ["left_knee_joint", "right_knee_joint"]:
                adr = m.jnt_qposadr[m.joint(joint).id]
                self.joint_limits.lower[adr] = max(margin, self.joint_limits.lower[adr])
        self.collisions = (
            _self_collision_limit(m, self.joints, _arm_bodies(m) if placeholder_arms else frozenset())
            if collision_avoidance
            else None
        )
        self._steps: dict[tuple[float, float], list] = {}  # step bounds, by (joint_step, root_step)

    def solve(
        self,
        start,
        tasks: list[Task],
        *,
        height: float,
        iterations: int = 8,
        joint_step: float = 0.08,
        root_step: float = 0.015,
        tolerance: float = 1e-4,
        damping: float = 1e-7,
        height_bounds: tuple[float, float] | None = None,
    ) -> IKResult:
        seed = np.asarray(start, float).copy()
        if 2 not in self.root_dofs:
            seed[2] = height
        objective = [_MinkTask(task) for task in tasks]
        limits = [self.joint_limits, *self._step_limits(joint_step, root_step)]
        if self.collisions is not None:
            limits.append(self.collisions)
        if height_bounds is not None:
            limits.append(HeightLimit(self.kin.model.nv, *height_bounds))

        def prepare():
            for term in objective:
                term.residual, term.jacobian = term.task.evaluate(self)

        def ready():
            for term in objective:
                task, error = term.task, term.residual
                if isinstance(task, FootPose):
                    if max(np.linalg.norm(error[:3]), np.linalg.norm(error[3:])) > tolerance:
                        return False
                elif (
                    isinstance(task, ComXY)
                    and np.linalg.norm(error) > tolerance
                    or isinstance(task, RootAttitude)
                    and abs(error[2]) > tolerance
                ):
                    return False
            return True

        outcome = advance(
            self.configuration,
            seed,
            objective,
            limits,
            self.frozen,
            prepare=prepare,
            ready=ready,
            iterations=iterations,
            damping=damping,
        )
        parts = [
            Part(p.task, p.what, p.value / term.task.weight, p.error, p.unit)
            for term in objective
            for p in term.task.parts(term.residual)
        ]
        return IKResult(outcome.pose[: len(start)], parts)

    def _step_limits(self, joint_step: float, root_step: float) -> list:
        """Bounds on one iteration's step (velocity limits with dt = 1)."""
        key = (joint_step, root_step)
        if key not in self._steps:
            m = self.kin.model
            names = [m.joint(1 + j).name for j in self.joints]
            self._steps[key] = [
                mink.VelocityLimit(m, dict.fromkeys(names, joint_step)),
                mink.FreeJointVelocityLimit(m, max_linear_velocity=root_step, max_angular_velocity=root_step),
            ]
        return self._steps[key]


def _self_collision_limit(model, joints, ignored=frozenset()) -> mink.CollisionAvoidanceLimit | None:
    """Keep robot parts apart, over the pairs whose distance the free joints can change (none on ``ignored``
    bodies)."""
    moving = _moved_bodies(model, set((1 + np.asarray(joints)).tolist()))  # joint ids; 0 is the free joint
    pairs = [
        ([a], [b])
        for a, b in self_collision_pairs(model)
        if model.geom_bodyid[a] not in ignored
        and model.geom_bodyid[b] not in ignored
        and (
            (model.geom_bodyid[a] in moving) != (model.geom_bodyid[b] in moving) or _chain_differs(model, a, b, moving)
        )
    ]
    if not pairs:
        return None
    return mink.CollisionAvoidanceLimit(
        model,
        pairs,
        minimum_distance_from_collisions=L.SELF_COLLISION_MARGIN,
        collision_detection_distance=COLLISION_DETECTION_M,
        bound_relaxation=COLLISION_RELAXATION,
    )


def _arm_bodies(model) -> frozenset[int]:
    """The bodies of both arms: everything below the shoulder pitch links, those included."""
    return frozenset().union(*(descendants(model, model.body(f"{side}_shoulder_pitch_link").id) for side in SIDES))


def _moved_bodies(model, joint_ids: set[int]) -> dict[int, frozenset[int]]:
    """Bodies below at least one of ``joint_ids``, each with the set of those joints above it."""
    moving = {}
    for body in range(model.nbody):
        above, b = set(), body
        while b > 0:
            start, count = model.body_jntadr[b], model.body_jntnum[b]
            above.update(j for j in range(start, start + count) if j in joint_ids)
            b = model.body_parentid[b]
        if above:
            moving[body] = frozenset(above)
    return moving


def _chain_differs(model, a, b, moving) -> bool:
    """Both geoms move, but not by exactly the same joints (so their distance can change)."""
    body_a, body_b = model.geom_bodyid[a], model.geom_bodyid[b]
    return body_a in moving and body_b in moving and moving[body_a] != moving[body_b]


def feet_tasks(positions, rotations, weight: float = 1.0) -> list[Task]:
    return [FootPose("feet", weight, side, positions[side], rotations[side]) for side in (0, 1)]
