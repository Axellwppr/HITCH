"""Household rooms: furniture with horizontal support surfaces, the pose of a second
piece beside the first, and the builder of a layout (its ``objects``, ``furniture``, static ``fixtures``, walls and
materials). The scenes on it (``tabletop_pick``) draw the layout; scene geometry is simulator metadata, and a task
exposes semantic descriptions and coarse bearings.
"""

from __future__ import annotations

from itertools import product

import mujoco
import numpy as np

from wbc.robot.model import model_spec
from wbc_link_server.scenes import household_materials
from wbc_link_server.scenes.geometry import box

NAMES = {
    "table": "rectangular table",
    "round_table": "round table",
    "stool": "high stool",
    "counter": "long counter",
}
BOARD_THICKNESS = 0.045  # m: a furniture's boards when its layout names none
BOARD_RANGE = (0.02, 0.06)  # m: the boards of generated furniture (one thickness per piece)
SUPPORT_HEIGHT_RANGE = (0.40, 0.77)  # m, the height of support tops
LAYOUTS = ("parallel_left", "parallel_right", "corner_left", "corner_right", "opposed")
WALL = [0.75, 0.76, 0.78, 1]


def rotation(yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])


def _surface(role, index, height, half_xy, *, radius=None):
    result = {
        "geom": f"{role}_support_{index}",
        "shape": "disk" if radius is not None else "rect",
        "height": float(height),
        "half_xy": list(map(float, half_xy)),
    }
    if radius is not None:
        result["radius"] = float(radius)
    return result


def _furniture(rng, role, kind):
    depth = float(rng.uniform(0.62, 0.82))
    width = float(rng.uniform(0.78, 1.02))
    height = float(rng.uniform(*SUPPORT_HEIGHT_RANGE))
    if kind == "stool":
        depth, width = rng.uniform(0.54, 0.64, 2)
    elif kind == "counter":
        depth, width = float(rng.uniform(0.58, 0.72)), float(rng.uniform(1.15, 1.50))
    radius = float(rng.uniform(0.43, 0.52)) if kind == "round_table" else None
    half = [radius, radius] if radius else [float(depth / 2), float(width / 2)]
    surfaces = [_surface(role, 0, height, half, radius=radius)]
    return {
        "id": role,
        "kind": kind,
        "center": [0.0, 0.0],
        "yaw": 0.0,
        "half_xy": half,
        "height": height,
        "surfaces": surfaces,
    }


def _footprint_radius(furniture):
    return (
        furniture["half_xy"][0] if furniture["kind"] == "round_table" else float(np.linalg.norm(furniture["half_xy"]))
    )


def _destination_pose(rng, source, destination):
    """Construct a shared working aisle before adding bounded pose variation.

    Furniture's local -X is its front. Parallel fronts share an aisle; corner
    fronts face the inside of the L; opposed fronts face one another. Circle
    separation reserves room independently of furniture size and yaw jitter.
    """
    layout = str(rng.choice(LAYOUTS))
    side = 1 if layout.endswith("left") else -1
    # Side/corner layouts leave a 65--85 cm gap even between enclosing circles.
    # Facing fronts need a wider aisle for the robot to turn in.
    gap = float(rng.uniform(1.40, 1.60) if layout == "opposed" else rng.uniform(0.65, 0.85))
    distance = _footprint_radius(source) + _footprint_radius(destination) + gap
    if layout.startswith("parallel"):
        delta = np.array([rng.uniform(-0.15, 0.15), side * distance])
        facing = 0.0
    else:
        angle = (180 if layout == "opposed" else side * 135) + rng.uniform(-8, 8)
        delta = distance * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        facing = 180 if layout == "opposed" else side * 90
    destination["center"] = (np.asarray(source["center"]) + rotation(source["yaw"]) @ delta).tolist()
    destination["yaw"] = float(source["yaw"] + np.radians(facing + rng.uniform(-12, 12)))


def _coarse_heading(degrees):
    """The direction available in the instruction, with no sub-sector bearing."""
    heading = int(45 * np.floor((degrees + 22.5) / 45))
    heading = (heading + 180) % 360 - 180
    return 180 if heading == -180 else heading


DIRECTION_WORDS = {  # the instruction's eight bearings, each in three wordings
    0: ("roughly ahead", "in front of you", "straight ahead"),
    45: ("ahead and to your left", "ahead on your left", "in front of you, to the left"),
    90: ("roughly to your left", "on your left", "to your left"),
    135: ("behind and to your left", "behind you on the left", "back to your left"),
    180: ("roughly behind you", "behind you", "at your back"),
    -135: ("behind and to your right", "behind you on the right", "back to your right"),
    -90: ("roughly to your right", "on your right", "to your right"),
    -45: ("ahead and to your right", "ahead on your right", "in front of you, to the right"),
}


def _direction(degrees, variant=0):
    words = DIRECTION_WORDS[_coarse_heading(degrees)]
    return words[variant % len(words)]


def build(layout, xml_path=None):
    """The MuJoCo model of ``layout`` (written to ``xml_path`` too)."""
    spec = model_spec(layout["objects"])
    for obj in layout["objects"]:
        yaw = obj.get("yaw", 0.0)
        quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        spec.body(obj["name"]).quat = quat.tolist()
    for furniture in layout["furniture"]:
        role, kind, yaw = furniture["id"], furniture["kind"], furniture["yaw"]
        body = spec.worldbody.add_body(
            name=role, pos=[*furniture["center"], 0], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        )
        color = [1, 1, 1, 1]
        hx, hy = furniture["half_xy"]
        board = furniture.get("board", BOARD_THICKNESS)
        for surface in furniture["surfaces"]:
            center = [0, 0, surface["height"] - board / 2]
            if surface["shape"] == "disk":
                body.add_geom(
                    name=surface["geom"],
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=center,
                    size=[surface["radius"], board / 2, 0],
                    rgba=color,
                    friction=[0.6, 0.005, 0.0001],
                )
            else:
                box(body, surface["geom"], center, [hx, hy, board / 2], color)
        leg_height = furniture["height"] - board
        leg_half = [max(hx - 0.045, 0.10), max(hy - 0.045, 0.10)]
        if kind == "round_table":
            leg_half = [0.58 * hx, 0.58 * hy]
        for index, (ix, iy) in enumerate(product((-1, 1), repeat=2)):
            box(
                body,
                f"{role}_leg_{index}",
                [ix * leg_half[0], iy * leg_half[1], leg_height / 2],
                [0.025, 0.025, leg_height / 2],
                [0.27, 0.28, 0.29, 1],
            )
    for fixture in layout.get("fixtures", []):  # static assemblies (containers, mats): bodies of box/cylinder geoms
        yaw = fixture.get("yaw", 0.0)
        body = spec.worldbody.add_body(
            name=fixture["name"], pos=fixture["pos"], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        )
        for g in fixture["geoms"]:
            geom = body.add_geom(name=g["name"], pos=g["pos"], size=g["size"], rgba=g["rgba"])
            geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER if g["type"] == "cylinder" else mujoco.mjtGeom.mjGEOM_BOX
            if "yaw" in g:
                geom.quat = [np.cos(g["yaw"] / 2), 0, 0, np.sin(g["yaw"] / 2)]
            geom.friction = g.get("friction", [0.8, 0.005, 0.0001])
            if g.get("visual_only"):
                geom.contype = geom.conaffinity = 0
    walls = layout.get("walls")  # [[x_min, x_max], [y_min, y_max]] of the room; default 8.2 m square
    bounds = walls or [[-4.1, 4.1], [-4.1, 4.1]]
    for axis, sign in product((0, 1), (-1, 1)):
        center = [(bounds[0][0] + bounds[0][1]) / 2, (bounds[1][0] + bounds[1][1]) / 2, 1.4]
        half = [(bounds[0][1] - bounds[0][0]) / 2, (bounds[1][1] - bounds[1][0]) / 2, 1.4]
        center[axis], half[axis] = bounds[axis][0 if sign < 0 else 1], 0.05
        box(spec.worldbody, f"room_wall_{axis}_{sign}", center, half, WALL)
    household_materials.apply(spec, layout)
    model = spec.compile()
    if xml_path is not None:
        xml_path.write_text(spec.to_xml())
    return model
