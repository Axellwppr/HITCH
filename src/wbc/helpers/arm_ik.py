"""Independent seven-DOF Mink arms: position first, orientation within its budget.

The body is prescribed per sample. A weak previous-frame preference resolves
redundancy. Carrying retains its actual orientation; free reaching tolerates up
to 60 degrees while continuously improving orientation inside a 5 mm position
budget. Rest has no palm target and can make room for the knees.
"""

from dataclasses import dataclass

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core import limits as L
from wbc.helpers.ik_core import JointBox, advance, clearance
from wbc.robot.arm import ArmGeometry, nominal_arm
from wbc.robot.kinematics import Kinematics, body_rotation

POSITION_TOL = 0.005
ROTATION_TOL = np.pi / 3
FOLLOW_ROTATION_TOL = np.radians(5.0)  # a hand following the body while it leans or changes height (docs 3.3)
POSTURE_WEIGHT = 0.01  # m/rad; weak previous-frame preference


@dataclass
class ArmResult:
    """One arm's solution at one frame and how far it is from its goal."""

    joints: np.ndarray
    position_error: np.ndarray  # target minus achieved in pelvis heading frame
    rotation_error: float
    clearance: float
    iterations: int
    termination: str
    rotation_tolerance: float = ROTATION_TOL

    @property
    def feasible(self) -> bool:
        """Whether the palm is within its position and rotation tolerances and the arm clear of the body."""
        return bool(
            np.linalg.norm(self.position_error) <= POSITION_TOL
            and self.rotation_error <= self.rotation_tolerance
            and self.clearance >= L.SELF_COLLISION_MARGIN - 0.0002
        )

    def feedback(self) -> dict:
        """The errors in public units, for a failure's details."""
        return dict(
            position_error_body_m=self.position_error.tolist(),
            position_error_m=float(np.linalg.norm(self.position_error)),
            rotation_error_deg=float(np.degrees(self.rotation_error)),
            rotation_tolerance_deg=float(np.degrees(self.rotation_tolerance)),
            minimum_body_clearance_m=self.clearance,
            termination=self.termination,
            iterations=self.iterations,
        )


class Point(mink.Task):
    """World translation independent of the SE(3) logarithm's rotation."""

    def __init__(self, site, target):
        super().__init__(cost=np.ones(3))
        self.site, self.target = site, np.asarray(target)

    def compute_error(self, configuration):
        return configuration.data.site_xpos[self.site] - self.target

    def compute_jacobian(self, configuration):
        jac = np.zeros((3, configuration.model.nv))
        mujoco.mj_jacSite(configuration.model, configuration.data, jac, None, self.site)
        return jac


class CartesianBudget(mink.Limit):
    """A step that keeps a task's linearised error within ``radius`` (the palm position while orientation improves)."""

    def __init__(self, jacobian, error, radius):
        self.matrix = np.vstack([jacobian, -jacobian])
        self.bound = np.r_[radius - error, radius + error]

    def compute_qp_inequalities(self, configuration, dt):
        return mink.Constraint(self.matrix, self.bound)


class ArmIK:
    """One arm's seven joints on a given body pose; the rest of the robot stays as it is."""

    def __init__(self, side, *, posture_weight=POSTURE_WEIGHT):
        if not np.isfinite(posture_weight) or not 0 <= posture_weight <= 0.02:
            raise ValueError("posture_weight must be in [0, 0.02] m/rad")
        self.kin = Kinematics()
        self.geometry = g = ArmGeometry(side, self.kin)
        self.posture_weight = posture_weight
        self.nominal = nominal_arm(side)
        m = self.kin.model
        self.configuration = mink.Configuration(m)
        self.configuration.data = self.kin.data
        self.frozen = [mink.DofFreezingTask(m, sorted(set(range(m.nv)) - set(g.cols)))]
        self.collisions = mink.CollisionAvoidanceLimit(
            m,
            [([a], [b]) for a, b in g.pairs],
            minimum_distance_from_collisions=L.SELF_COLLISION_MARGIN,
            collision_detection_distance=0.05,
            bound_relaxation=-0.001,
        )

    def solve(self, pose, position, rotation, seed, *, bounds=None, rotation_tolerance=ROTATION_TOL) -> ArmResult:
        """The arm from ``seed`` toward a palm ``position`` and ``rotation`` (pelvis heading frame) on ``pose``."""
        return self._solve(pose, seed, (position, rotation), bounds, rotation_tolerance)

    def rest(self, pose, target, seed, *, bounds=None) -> ArmResult:
        """The arm from ``seed`` toward the joint angles ``target`` (no palm goal), clear of the body on ``pose``."""
        return self._solve(pose, seed, None, bounds, ROTATION_TOL, preferred=target)

    def _solve(self, pose, seed, goal, bounds, angle_limit, preferred=None):
        g, c, m = self.geometry, self.configuration, self.kin.model
        state = np.asarray(pose).copy()
        state[g.slice] = seed
        low, high = g.limits.T
        if bounds is not None:
            low, high = np.maximum(low, bounds[0]), np.minimum(high, bounds[1])
        box = JointBox(m.nv, g.slice, g.cols, low, high)
        limits = [box, self.collisions]
        heading = body_rotation(pose)
        target = target_rotation = None
        tasks = []
        cost = np.zeros(m.nv)
        cost[g.cols] = self.posture_weight if goal is not None else 1.0
        posture = mink.PostureTask(m, cost=cost)
        desired = state.copy()
        desired[g.slice] = seed if preferred is None else preferred
        posture.set_target(desired)
        tasks.append(posture)
        if goal is not None:
            position, rotation = goal
            target, target_rotation = pose[:3] + heading @ position, heading @ rotation
            point = Point(g.site, target)
            rotate = mink.FrameTask(m.site(g.site).name, "site", 0, 0.3 if angle_limit < ROTATION_TOL else 0.03)
            rotate.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(target_rotation), target))
            tasks[:0] = [point, rotate]

        result = None

        def prepare():
            nonlocal result
            error, angle = np.zeros(3), 0.0
            if goal is not None:
                error = heading.T @ (target - c.data.site_xpos[g.site])
                angle = float(
                    Rotation.from_matrix(target_rotation @ c.data.site_xmat[g.site].reshape(3, 3).T).magnitude()
                )
            result = ArmResult(
                c.data.qpos[g.slice].copy(), error, angle, clearance(c, self.collisions), 0, "", angle_limit
            )

        def budget():
            primary = mink.solve_ik(c, [point], 1.0, "daqp", damping=1e-7, limits=limits, constraints=self.frozen)
            jac, error = heading.T @ point.compute_jacobian(c), heading.T @ point.compute_error(c)
            radius = np.maximum(0.95 * POSITION_TOL / np.sqrt(3), abs(error + jac @ primary) + 1e-8)
            return [CartesianBudget(jac, error, radius)]

        outcome = advance(
            c,
            state,
            tasks,
            limits,
            self.frozen,
            prepare=prepare,
            ready=lambda: result.feasible,
            iterations=8 if bounds is not None else 32,
            secondary_limits=budget if goal is not None else None,
            coordinates=(g.slice, g.cols),
        )
        result.iterations, result.termination = outcome.iterations, outcome.status
        return result
