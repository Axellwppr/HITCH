"""How far the torso may lean and turn at each pelvis height (docs/REFERENCE.md 3.2).

The planner checks every frame of a body change against ``torso_limits`` (``wbc.helpers.body.check_limits``), and the
system prompt renders the same table (``wbc.agent.prompts.torso_table``), so the two never drift apart. It lives in
``core`` because the agent layer reads it without the planner's IK and MuJoCo.
"""

# (lowest pelvis height m, largest lean, largest turn with a lean up to TURN_LEAN, largest turn beyond), degrees: one
# step inside what the planner plans from standing (running out first: the legs' room to keep both feet planted, the
# lower the more). Below 0.40 m the standard posture stays as at 0.40 m (helpers.gait.balance_pose clips to
# gait.MIN_HEIGHT).
TORSO_LIMITS = (
    (0.785, 35, 45, 35),
    (0.55, 60, 45, 40),
    (0.50, 55, 45, 40),
    (0.46, 45, 40, 20),
    (0.42, 25, 20, 10),
    (0.38, 20, 20, 15),
    (0.30, 10, 10, 10),
)
TURN_LEAN = 20


def torso_limits(height: float, lean: float = 0.0) -> tuple[int, int]:
    """(largest lean, largest turn with this lean) in degrees at a pelvis height; (0, 0) below the lowest row."""
    for low, lean_max, turn, turn_leaned in TORSO_LIMITS:
        if height >= low - 1e-6:
            return lean_max, turn if lean <= TURN_LEAN else turn_leaned
    return 0, 0
