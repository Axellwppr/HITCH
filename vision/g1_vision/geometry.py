"""Transforms are destination_from_source; metres, angles in degrees.

The nominal chain is the head mount's design geometry, the same transforms the simulation uses.
Encoder zero is the horizontal pose, not the design's joint zero.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def transform(xyz=(0, 0, 0), axis=None, degrees=0):
    t = np.eye(4)
    t[:3, 3] = xyz
    if axis:
        t[:3, :3] = Rotation.from_euler(axis, degrees, degrees=True).as_matrix()
    return t


def rigid(value):
    t = np.asarray(value, float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1)):
        raise ValueError("expected a finite rigid 4x4 transform")
    return t


def nominal(yaw_deg, pitch_deg):
    mount = transform([0.0076864148932334, 0, 0.3823833123616], 'y', -0.7248449788842486)
    yaw = transform([0.0938, 0, 0.019158474645501], 'z', yaw_deg)
    pitch = transform([0.0245, 0, 0.0384], 'y', 5.724844978884216 + pitch_deg - 50)
    optical = transform([0.039067649660557, -0.0115, -0.02209708691208], 'y', 45)
    # Camera optical RDF -> mechanical FLU.
    optical[:3, :3] = optical[:3, :3] @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    return mount @ yaw @ pitch @ optical


class Mount:
    def __init__(self, calibration):
        self.base = rigid(calibration.get('base_correction', np.eye(4)))
        self.camera = rigid(calibration.get('camera_correction', np.eye(4)))

    def pose(self, eye):
        return self.base @ nominal(eye['yaw_deg'], eye['pitch_deg']) @ self.camera

    def servo(self, eye):
        """The servo angles at which this calibrated mount looks along the optical axis the nominal mount (the
        simulation's) has at the setting ``eye``. Two axes cannot also undo the image's roll about that axis nor the
        lens's offset."""
        if np.allclose(self.base[:3, :3], np.eye(3)) and np.allclose(self.camera[:3, :3], np.eye(3)):
            return {'yaw_deg': float(eye['yaw_deg']), 'pitch_deg': float(eye['pitch_deg'])}
        want = nominal(eye['yaw_deg'], eye['pitch_deg'])[:3, 2]
        fit = least_squares(lambda x: self.pose({'yaw_deg': x[0], 'pitch_deg': x[1]})[:3, 2] - want,
                            [eye['yaw_deg'], eye['pitch_deg']], xtol=1e-10)
        return {'yaw_deg': float(fit.x[0]), 'pitch_deg': float(fit.x[1])}
