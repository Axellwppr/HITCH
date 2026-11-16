"""Body reference checks, including the walker's millimetre accuracy gates."""

import numpy as np
from scipy.spatial.transform import Rotation

from wbc.checks import CheckResult, Limits, run_checks
from wbc.core import limits as L
from wbc.core.frames import SE2, wrap
from wbc.core.layout import SIDES, UPPER
from wbc.helpers.gait import gait_profile
from wbc.robot.kinematics import Kinematics
from wbc.robot.model import lowest_point


def profile(start):
    return L.LOW_GAIT if gait_profile(start.height).crouched else L.WALK


def check(start, draft, parameters):
    report = run_checks(draft, Limits((0,), (profile(start),)))
    if "finite" in report.failures:
        return report

    def add(name, value, limit, detail="", frame=None):
        report.results.append(
            CheckResult(name, bool(value <= limit), float(value), float(limit), detail=detail, frame=frame)
        )

    add("start_continuity", abs(draft.frames[0] - start.pose).max(), L.WALKER_UNCHANGED)
    add("upper_preserved", abs(draft.frames[:, UPPER] - start.pose[UPPER]).max(), L.WALKER_UNCHANGED)
    add("ik_accuracy", draft.diagnostics["ik_residual"], L.WALKER_IK_ACCURACY, detail="scaled foot/CoM residual")
    kin = Kinematics()
    feet, rotations, soles = [], [], []
    foot_geoms = [g for g in kin.collisions if kin.model.geom_bodyid[g] in kin.feet]
    for pose in draft.frames:
        kin.forward(pose)
        feet.append(kin.foot_positions())
        rotations.append(kin.foot_rotations())
        soles.append(min(lowest_point(kin.model, kin.data, g) for g in foot_geoms))
    feet, rotations = np.asarray(feet), np.asarray(rotations)
    add("sole_penetration", max(0, -min(soles)), L.WALKER_SOLE_PENETRATION)
    if parameters.get("footsteps"):
        swings = [p for p in draft.phases if p.name.startswith("swing_")]
        for phase, goal in zip(swings, parameters["footsteps"], strict=True):
            side = SIDES.index(goal["side"])
            add(
                "landing_position",
                np.linalg.norm(feet[phase.stop, side, :2] - goal["xy"]),
                L.WALKER_LANDING_POSITION,
                detail=f"{goal['side']} ankle XY error, m",
                frame=phase.stop,
            )
            if goal.get("yaw") is not None:
                error = Rotation.from_euler("z", goal["yaw"]).inv() * Rotation.from_matrix(rotations[phase.stop, side])
                add(
                    "landing_heading",
                    np.degrees(error.magnitude()),
                    np.degrees(L.WALKER_LANDING_HEADING),
                    detail=f"{goal['side']} orientation error, deg",
                    frame=phase.stop,
                )
    else:
        add(
            "root_target",
            np.linalg.norm(draft.frames[-1, :2] - parameters["target"]),
            L.WALKER_ROOT_TARGET,
            detail="root XY destination error, m",
        )
        goal_yaw = SE2.of_pose(start.pose).yaw + parameters.get("turn", 0.0)
        add(
            "root_heading",
            np.degrees(abs(wrap(SE2.of_pose(draft.frames[-1]).yaw - goal_yaw))),
            np.degrees(L.WALKER_ROOT_HEADING),
            detail="root yaw error, deg",
        )
    return report
