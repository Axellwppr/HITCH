"""The body in numbers, as observations give it: pelvis, feet and palms in the observation frame O."""

from __future__ import annotations

from scipy.spatial.transform import Rotation

from wbc.core.layout import SIDES
from wbc.core.types import posture_for_height
from wbc.robot.kinematics import Kinematics
from wbc.robot.model import lowest_point


def _rounded(values, digits=3) -> list[float]:
    return [round(float(v), digits) + 0.0 for v in values]  # + 0.0: no -0.0


def body_description(pose, kin: Kinematics) -> dict:
    """Pelvis, both feet and both palms of ``pose`` (in O), in numbers. O's x axis is the pelvis heading."""
    data = kin.forward(pose)
    roll, pitch, _ = Rotation.from_quat(pose[3:7], scalar_first=True).as_euler("xyz", degrees=True)
    feet = {}
    for side, body in zip(SIDES, kin.feet, strict=True):
        soles = [lowest_point(kin.model, data, g) for g in kin.collisions if int(kin.model.geom_bodyid[g]) == body]
        feet[side] = {
            "ankle_xyz": _rounded(data.xpos[body]),
            "yaw_deg": round(float(Rotation.from_matrix(data.xmat[body].reshape(3, 3)).as_euler("xyz", True)[2])),
            "sole_height_m": _rounded([min(soles)])[0],
        }
    return {
        "posture": posture_for_height(float(pose[2])).value,
        "pelvis": {
            "xyz": _rounded(pose[:3]),
            "roll_deg": round(float(roll)),
            "pitch_deg": round(float(pitch)),
        },
        "feet": feet,
        "hands": {side: _palm(data, site, side) for side, site in zip(SIDES, kin.palms, strict=True)},
        "com_xy": _rounded(kin.com()[:2]),
    }


def _palm(data, site: int, side: str) -> dict:
    """A palm's position and directions as a hand request gives them (``orientation.normal`` and ``.fingers``):
    the palm normal and the finger direction (wbc.core.request.Orientation: the site's x axis is the fingers,
    its y axis the right palm's normal and the left palm's reverse normal). ``fingers`` stays free for the session's
    open/closed finger state."""
    frame = data.site_xmat[site].reshape(3, 3)
    return {
        "palm_xyz": _rounded(data.site_xpos[site]),
        "normal": _rounded(frame[:, 1] * (-1 if side == "left" else 1), 2),
        "finger_direction": _rounded(frame[:, 0], 2),
    }
