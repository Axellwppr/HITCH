"""Warm-started Mink iteration shared by the body and the arm solvers.

Adapters choose tasks, active coordinates and physical acceptance tolerances.
Soft preferences contribute to the QP without becoming convergence conditions.
"""

from dataclasses import dataclass

import mink
import mujoco
import numpy as np


@dataclass
class Outcome:
    pose: np.ndarray
    iterations: int
    status: str


def advance(
    configuration,
    start,
    tasks,
    limits,
    frozen,
    *,
    prepare,
    ready,
    iterations=8,
    secondary_limits=None,
    damping=1e-7,
    coordinates=None,
):
    """Advance one sample; all frame bounds remain fixed across iterations.

    ``secondary_limits`` may add a position budget established by a primary QP.
    ``coordinates`` allows independent arms to preserve the body's bits exactly.
    """
    configuration.update(start)
    prepare()
    count, status = 0, "iteration_limit"
    for _ in range(iterations):
        try:
            extra = secondary_limits() if secondary_limits is not None else []
            delta = mink.solve_ik(
                configuration, tasks, 1.0, "daqp", damping=damping, limits=[*limits, *extra], constraints=frozen
            )
        except mink.NoSolutionFound:
            status = "qp_infeasible"
            break
        if coordinates is None:
            configuration.integrate_inplace(delta, 1.0)
        else:
            qpos, dofs = coordinates
            pose = configuration.data.qpos.copy()
            pose[qpos] += delta[dofs]
            configuration.update(pose)
        count += 1
        prepare()
        if ready():
            status = "task_tolerance"
            break
        if np.max(abs(delta)) < 1e-8:
            status = "stationary"
            break
    return Outcome(configuration.data.qpos.copy(), count, status)


class JointBox(mink.Limit):
    """Physical/frame bounds plus a numerical radius for one linearization."""

    def __init__(self, nv, qpos, dofs, lower, upper, step=0.08):
        self.qpos, self.lower, self.upper, self.step = qpos, lower, upper, step
        projection = np.eye(nv)[dofs]
        self.matrix = np.vstack([projection, -projection])

    def compute_qp_inequalities(self, configuration, dt):
        q = configuration.data.qpos[self.qpos]
        return mink.Constraint(
            self.matrix, np.r_[np.minimum(self.upper - q, self.step), np.minimum(q - self.lower, self.step)]
        )


def clearance(configuration, collisions):
    """Exact narrow phase for the collision limit's conservative broad phase.

    The limit and this acceptance check share pairs, distances and current FK.
    The planner checks interactions between independently solved limbs afterwards.
    """
    if collisions is None:
        return 0.05
    return min(
        (
            float(
                mujoco.mj_geomDistance(
                    configuration.model, configuration.data, *collisions.geom_id_pairs[i], 0.05, None
                )
            )
            # mink's private broad phase (CollisionAvoidanceLimit._broadphase_survivors): the pairs the limit itself
            # constrains this step. Private API: mink is pinned (==1.3.0, pyproject.toml); check it on an upgrade.
            for i in collisions._broadphase_survivors(configuration.data)
        ),
        default=0.05,
    )
