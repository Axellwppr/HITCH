"""Planned double-support geometry (no contact sensing)."""

from __future__ import annotations

import re

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

FOOT_CAPSULE = re.compile(r"(left|right)_foot\d+_collision")
ANKLE_HEIGHT = 0.035  # ankle_roll_link origin above flat ground when the sole is flat
# Toe capsule geometry: lift needed to keep the toe above ground at a foot pitch.
TOE_REACH = 0.132
TOE_RADIUS_DROP = 0.025


def toe_lift(pitch: float) -> float:
    return TOE_REACH * np.sin(pitch) + TOE_RADIUS_DROP * (np.cos(pitch) - 1)


def support_hull(model: mujoco.MjModel, data: mujoco.MjData) -> ConvexHull:
    """Convex hull of both feet's capsule centrelines on the ground plane."""
    points = []
    for geom in range(model.ngeom):
        if FOOT_CAPSULE.fullmatch(model.geom(geom).name or ""):
            axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
            for sign in (-1, 1):
                points.append((data.geom_xpos[geom] + sign * model.geom_size[geom, 1] * axis)[:2])
    if len(points) < 3:
        raise ValueError("robot model lacks foot support capsules")
    return ConvexHull(np.array(points))


def hull_margin(hull: ConvexHull, point_xy) -> float:
    """Signed distance of a point inside (positive) the hull."""
    return float(-(hull.equations[:, :2] @ np.asarray(point_xy) + hull.equations[:, 2]).max())
