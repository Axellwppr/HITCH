"""Recovery policy, independent of generation and execution.

Only a stance-related failure admits collection. Recovery is a finite recipe,
not recursive planning: at most two foot orders, then one retry of the
unchanged goal. No timing/target/threshold search.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Recovery:
    steps: tuple[str, ...]
    decision: str

    def to_dict(self):
        return dict(steps=list(self.steps), decision=self.decision)


def choose(*, enabled, reason, stance):
    if not enabled:
        return Recovery((), "disabled")
    if reason not in {"STANCE_REQUIRED", "TARGET_NOT_REACHED", "SPEED_LIMIT"}:
        return Recovery((), "failure_not_repairable_by_collection")
    # Walking from an ordinary resting width has no stance problem to repair.
    if stance.resting:
        return Recovery((), "stance_already_suitable")
    return Recovery(("collect", "retry"), "adjust_stance")


# Ordered groups make combined failures deterministic. A check outside them
# counts as an invalid trajectory, which collecting the feet does not repair.
FAILURE_GROUPS = (
    ("INVALID_TRAJECTORY", {"finite", "start_continuity", "upper_preserved"}),
    ("COLLISION", {"self_collision"}),
    ("UPPER_BODY_LIMIT", {"upper_speed"}),
    (
        "TARGET_NOT_REACHED",
        {
            "ik_residual",
            "ik_accuracy",
            "landing_position",
            "landing_heading",
            "root_target",
            "root_heading",
            "joint_limits",
        },
    ),
    ("COLLISION", {"ground_penetration", "sole_penetration"}),
    ("SPEED_LIMIT", {"leg_speed", "root_speed", "root_turn", "stance_slip"}),
)


def failure_reason(attempt):
    if attempt.error is not None:
        return attempt.error["reason"]
    failures = set(attempt.report.failures)
    known = set().union(*(checks for _, checks in FAILURE_GROUPS))
    if failures - known:
        return "INVALID_TRAJECTORY"
    return next(
        (reason for reason, checks in FAILURE_GROUPS if failures & checks),
        "INVALID_TRAJECTORY",
    )
