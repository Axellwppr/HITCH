"""RGB-D geometry in the observation frame O, and the observation images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wbc.core.camera import pixel_rays
from wbc.core.errors import invalid
from wbc.robot.kinematics import Kinematics, link_transform

BACKGROUND = (18, 24, 34)
FOREGROUND = (218, 230, 243)


@dataclass(frozen=True)
class Camera:
    """Camera with optional lens distortion and pose in O (optical frame: x right, y down, z forward)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    pose: np.ndarray  # 4x4, O <- camera
    distortion: dict | None = None

    def to_dict(self) -> dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "pose": np.asarray(self.pose).tolist(),
            **({"distortion": self.distortion} if self.distortion is not None else {}),
        }

    @classmethod
    def from_dict(cls, value: dict) -> Camera:
        return cls(
            value["fx"],
            value["fy"],
            value["cx"],
            value["cy"],
            int(value["width"]),
            int(value["height"]),
            np.asarray(value["pose"], float),
            value.get("distortion"),
        )


def camera_in_observation(kin: Kinematics, pose_o, link: str, link_to_camera, intrinsics, shape) -> Camera:
    transform = link_transform(kin, pose_o, link) @ np.asarray(link_to_camera, float)
    height, width = shape
    return Camera(
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["cx"]),
        float(intrinsics["cy"]),
        int(width),
        int(height),
        transform,
        intrinsics.get("distortion"),
    )


def point_at(depth, camera: Camera, u: int, v: int) -> dict:
    """The point in O that the camera ray through pixel (u, v) meets at its axial depth."""
    h, w = depth.shape
    record = {"pixel": [u, v], "valid": False, "xyz": None}
    if not (0 <= u < w and 0 <= v < h):
        return {**record, "reason": "outside the image"}
    if not np.isfinite(depth[v, u]) or depth[v, u] <= 0:
        return {**record, "reason": "no depth at this pixel"}
    patch = depth[max(0, v - 2) : v + 3, max(0, u - 2) : u + 3]
    patch = patch[np.isfinite(patch)]
    spread = float(np.ptp(patch)) if patch.size else 0.0
    ray = pixel_rays([u, v], camera.to_dict())
    xyz = camera.pose[:3, :3] @ (ray * depth[v, u]) + camera.pose[:3, 3]
    return {  # millimetres: finer digits are noise
        **record,
        "valid": True,
        "xyz": [round(c, 3) + 0.0 for c in xyz.tolist()],
        "depth_m": round(float(depth[v, u]), 3),
        "ground_distance_m": round(float(np.linalg.norm(xyz[:2])), 3),
        "patch_depth_spread_m": round(spread, 3),
    }


def perceive(depth, camera: Camera, image_points) -> list[dict]:
    """3D points in the observation frame O for normalized image points ``[x, y]`` (each 0–1000, top-left [0, 0],
    x right, y down): one record per point, in order, with the pixel read (the nearest one) and the requested
    coordinates echoed as ``image_point``."""
    if not isinstance(image_points, (list, tuple)) or not 1 <= len(image_points) <= 100:
        raise invalid("points: give 1-100 image points [x, y]")
    for point in image_points:
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or not all(isinstance(c, (int, float)) and not isinstance(c, bool) and 0 <= c <= 1000 for c in point)
        ):
            raise invalid(f"points: each image point is [x, y], each 0-1000; got {point!r}")
    height, width = depth.shape
    located = []
    for x, y in image_points:
        record = point_at(depth, camera, round(x * (width - 1) / 1000), round(y * (height - 1) / 1000))
        located.append({**record, "image_point": [x, y]})
    return located


def font(size=16):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_points(rgb, points: list[dict], output, title: str) -> None:
    image = Image.fromarray(np.asarray(rgb)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for i, p in enumerate(points):
        u, v = p["pixel"]
        color = (105, 255, 180) if p["valid"] else (255, 125, 110)
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=color)
        draw.text(
            (u + 5, v - 11),
            str(i + 1),
            fill=color,
            stroke_width=2,
            stroke_fill=(10, 16, 25),
            font=font(13),
        )
    # The title sits in a strip above the image, so it never hides points near the top edge.
    canvas = Image.new("RGB", (image.width, image.height + 30), BACKGROUND)
    canvas.paste(image, (0, 30))
    ImageDraw.Draw(canvas).text((9, 5), title, fill=FOREGROUND, font=font(13))
    canvas.save(output)
