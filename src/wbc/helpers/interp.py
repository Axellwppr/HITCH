"""Time and interpolation primitives shared by generators."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.spatial.transform import Rotation, Slerp

from wbc.core.layout import DT

HOLD_S = 0.2  # stationary tail every generator ends with
BLEND_S = 1.2  # default duration of posture / hand blends


def frames_for(seconds: float) -> int:
    """Number of 50 Hz steps covering ``seconds`` (one rounding rule everywhere)."""
    return max(1, int(np.ceil(seconds / DT - 1e-9)))


def smooth(s):
    """Quintic ease with zero velocity and acceleration at both ends."""
    s = np.clip(s, 0.0, 1.0)
    return s**3 * (10 + s * (-15 + 6 * s))


def slerp(q0, q1):
    """Return u -> wxyz quaternion between two wxyz quaternions."""
    path = Slerp([0, 1], Rotation.from_quat(np.stack([q0, q1]), scalar_first=True))
    return lambda u: path(np.clip(u, 0, 1)).as_quat(scalar_first=True)


def hold(pose, count: int) -> np.ndarray:
    return np.repeat(np.asarray(pose, float)[None], count, axis=0)


def lipm(zmp, height: float, dt: float = DT):
    """Bounded solution of c'' = g/h (c - zmp) with fixed endpoints (CoM preview)."""
    a = height / 9.81 / dt**2
    n = len(zmp)
    bands = np.zeros((3, n))
    bands[0, 1:] = -a
    bands[1] = 1 + 2 * a
    bands[2, :-1] = -a
    bands[1, [0, -1]] = 1
    bands[0, 1] = 0
    bands[2, -2] = 0
    return solve_banded((1, 1), bands, zmp)
