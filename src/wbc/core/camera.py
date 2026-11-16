"""Pixel-to-ray distortion correction; images and depth stay in their original pixel grid.

Coefficient order: k1, k2, p1, p2, k3. RealSense inverse/modified Brown use the
librealsense 2.50 forward convention (radial scaling before tangential terms).
"""

import numpy as np


def distorted_rays(xy, distortion):
    """Forward projection in normalized coordinates, matching the RealSense SDK."""
    model = distortion["model"]
    if model not in ("brown_conrady", "inverse_brown_conrady", "modified_brown_conrady"):
        raise ValueError(f"unsupported camera distortion model: {model}")
    coefficients = np.asarray(distortion["coeffs"], float)
    if coefficients.shape != (5,) or not np.isfinite(coefficients).all():
        raise ValueError("camera distortion requires five finite coefficients")
    k1, k2, p1, p2, k3 = coefficients
    x, y = np.moveaxis(np.asarray(xy, float), -1, 0)
    r2 = x * x + y * y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    rx, ry = x * radial, y * radial
    if model != "brown_conrady":
        x, y = rx, ry
    return np.stack((rx + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), ry + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)), axis=-1)


def pixel_rays(pixels, intrinsics):
    """Return optical [x/z,y/z,1] rays for (...,2) original-image pixels.

    Missing distortion is a pinhole camera. Newton inversion checks its residual;
    unsupported models and failed inversions are explicit errors, never silent pinhole fallbacks.
    """
    focal = np.array([intrinsics["fx"], intrinsics["fy"]], float)
    center = np.array([intrinsics["cx"], intrinsics["cy"]], float)
    if not np.isfinite(np.r_[focal, center]).all() or np.any(focal <= 0):
        raise ValueError("invalid camera intrinsics")
    target = (np.asarray(pixels, float) - center) / focal
    if not np.isfinite(target).all():
        raise ValueError("nonfinite image coordinates")
    xy = target.copy()
    distortion = intrinsics.get("distortion")
    if distortion is not None:
        for _ in range(15):
            projected = distorted_rays(xy, distortion)
            error = projected - target
            if np.max(np.abs(error), initial=0) < 1e-10:
                break
            eps = 1e-6
            dx = (distorted_rays(xy + [eps, 0], distortion) - projected) / eps
            dy = (distorted_rays(xy + [0, eps], distortion) - projected) / eps
            jac = np.stack((dx, dy), axis=-1)
            xy -= np.linalg.solve(jac, error[..., None])[..., 0]
        if not np.isfinite(xy).all() or np.max(np.abs(distorted_rays(xy, distortion) - target), initial=0) > 1e-7:
            raise ValueError("camera distortion inversion did not converge")
    return np.concatenate((xy, np.ones((*xy.shape[:-1], 1))), axis=-1)
