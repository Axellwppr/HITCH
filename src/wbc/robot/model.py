"""The G1 MuJoCo model shared by FK, IK, checks and the simulator.

``build_model()`` compiles the robot once and hands out shallow copies.
Simulator scene objects are added through ``build_model(objects=...)``.
The hand on both wrists is this process's ``WBC_HAND`` (``wbc.robot.hands``).
"""

from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from functools import lru_cache

import mujoco
import numpy as np

from wbc.core.eye import Eye
from wbc.core.layout import JOINTS, SIDES
from wbc.paths import G1_ASSETS
from wbc.robot.eye import FOVY_DEG, mujoco_in_torso
from wbc.robot.hands import HANDS, Hand, current_hand

ROBOT_XML = G1_ASSETS / "g1.xml"
_nominal = json.loads((G1_ASSETS / "nominal.json").read_text())
NOMINAL_JOINTS = np.array(_nominal["joint_positions"])
# Fixed upper-body mass prior used to calibrate the body balance curves.
# Resting hands are independently solved against the resulting body frames.
BALANCE_ARM_JOINTS = np.array(_nominal["balance_arm_positions"])
NOMINAL_HEIGHT = 0.75

# Contact solver: elliptic friction cones, and a noslip pass so that static friction holds under a steady load
# (an object held in a still grip stays still).
CONTACT_SOLVER = {"cone": "elliptic", "impratio": "10", "tolerance": "1e-10", "noslip_iterations": "10"}


def nominal_pose(height: float = NOMINAL_HEIGHT) -> np.ndarray:
    return np.r_[0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0, NOMINAL_JOINTS]


def _add_object(world, obj: dict) -> None:
    body = ET.SubElement(world, "body", name=obj["name"], pos=" ".join(map(str, obj["pos"])))
    if "quat" in obj:  # w, x, y, z
        body.set("quat", " ".join(map(str, obj["quat"])))
    if obj.get("dynamic"):
        ET.SubElement(body, "freejoint", name=obj["name"] + "_free")
    if obj["type"] == "furniture":
        a, b = obj["half_xy"]
        h = obj["clearance"]
        thick = 0.06
        ET.SubElement(
            body,
            "geom",
            name=obj["name"] + "_top",
            type="box",
            size=f"{a} {b} {thick / 2}",
            pos=f"0 0 {h + thick / 2}",
            rgba=".5 .3 .17 1",
        )
        for i, (x, y) in enumerate(
            [(-a + 0.04, -b + 0.04), (-a + 0.04, b - 0.04), (a - 0.04, -b + 0.04), (a - 0.04, b - 0.04)]
        ):
            ET.SubElement(
                body,
                "geom",
                name=obj["name"] + f"_leg{i}",
                type="box",
                size=f".025 .025 {h / 2}",
                pos=f"{x} {y} {h / 2}",
                rgba=".4 .25 .14 1",
            )
        return
    attributes = dict(
        name=obj["name"] + "_geom",
        type=obj["type"],
        size=" ".join(map(str, obj["size"])),
        rgba=" ".join(map(str, obj.get("rgba", (0.95, 0.25, 0.12, 1)))),
        friction=" ".join(map(str, obj.get("friction", (0.6, 0.005, 0.0001)))),
        condim=str(obj.get("condim", 3)),
    )
    if obj.get("visual_only"):
        attributes.update(contype="0", conaffinity="0")
    if obj.get("dynamic"):
        attributes["mass"] = str(obj["mass"])
    if "solref" in obj:
        attributes["solref"] = " ".join(map(str, obj["solref"]))
    if "priority" in obj:
        attributes["priority"] = str(obj["priority"])
    ET.SubElement(body, "geom", **attributes)
    for tag in obj.get("visual_tags", []):
        ET.SubElement(
            body,
            "geom",
            name=obj["name"] + "_" + tag["name"],
            type="cylinder",
            size=".009 .0005",
            pos=" ".join(map(str, tag["pos"])),
            rgba=" ".join(map(str, tag["rgba"])),
            contype="0",
            conaffinity="0",
            mass="0",
        )


def _model_xml(objects=(), hand: Hand | None = None, simulate: bool = False) -> str:
    hand = hand or current_hand()
    root = ET.parse(ROBOT_XML).getroot()
    root.find("compiler").set("meshdir", str(ROBOT_XML.parent / "meshes"))
    option = root.find("option")
    option.set("timestep", ".005")
    for name, value in CONTACT_SOLVER.items():
        option.set(name, value)
    flags = option.find("flag")
    if flags is None:
        flags = ET.SubElement(option, "flag")
    flags.set("multiccd", "enable")
    actuators = root.find("actuator")
    for side in SIDES:
        for actuator in hand.mount(root, side, simulate):
            actuators.append(actuator)  # after the 29 body motors, which keep their canonical order
        root.find(f".//site[@name='{side}_palm']").set("pos", " ".join(map(str, hand.palm_position(side))))
    # the head camera at the mount's default setting (wbc.robot.eye; the vision service moves it per image)
    position, rotation = mujoco_in_torso(Eye())
    ET.SubElement(
        root.find(".//body[@name='torso_link']"),
        "camera",
        name="ego",
        pos=" ".join(f"{v:.6f}" for v in position),
        xyaxes=" ".join(f"{v:.9f}" for v in [*rotation[:, 0], *rotation[:, 1]]),
        fovy=f"{FOVY_DEG:.6f}",
    )
    world = root.find("worldbody")
    for obj in objects:
        _add_object(world, obj)
    return ET.tostring(root, encoding="unicode")


def model_spec(objects=(), *, hand: Hand | None = None, simulate: bool = True) -> mujoco.MjSpec:
    """Fresh specification for simulator scenery (finger joints held by their actuators); FK/IK keep the robot alone."""
    return mujoco.MjSpec.from_string(_model_xml(objects, hand, simulate))


def _compile(objects: tuple = (), hand: Hand | None = None, simulate: bool = False) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(_model_xml(objects, hand, simulate))
    hinges = [model.joint(j).name for j in range(1, model.njnt)]
    if tuple(n for n in hinges if "_hand_" not in n)[: len(JOINTS)] != JOINTS:
        raise ValueError("G1 model joint order differs from the canonical layout")
    if not simulate and hinges[: len(JOINTS)] != list(JOINTS):
        raise ValueError("a planning model has only the 29 body joints first")
    return model


@lru_cache(maxsize=len(HANDS))
def _robot_only(hand: str) -> mujoco.MjModel:
    return _compile(hand=HANDS[hand])


def robot_model() -> mujoco.MjModel:
    """The compiled robot model with this process's hand, shared: read it, never change it (``build_model`` gives a
    private copy). Its fingers are fixed, so qpos is the root and the 29 body joints."""
    return _robot_only(current_hand().name)


def build_model(objects=(), *, hand: Hand | None = None, simulate: bool = False) -> mujoco.MjModel:
    """A private robot model, optionally with extra scene objects (dicts, see _add_object). A simulator model
    (``simulate``) has the hand's finger joints too; find body joints with ``body_joint_addresses``."""
    hand = hand or current_hand()
    if not objects and not simulate:
        return copy.copy(_robot_only(hand.name))
    return _compile(tuple(objects), hand, simulate)


def body_joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """qpos and qvel addresses of the 29 body joints, in canonical order (finger joints may lie between them)."""
    joints = [model.joint(name).id for name in JOINTS]
    return model.jnt_qposadr[joints].copy(), model.jnt_dofadr[joints].copy()


def robot_extent(model: mujoco.MjModel) -> tuple[int, int]:
    """Lengths of the robot's leading block of qpos and qvel (root, body joints, fingers); scenery follows."""
    bodies = robot_bodies(model)
    joints = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in bodies]
    last = max(joints)
    width = {mujoco.mjtJoint.mjJNT_FREE: (7, 6), mujoco.mjtJoint.mjJNT_BALL: (4, 3)}.get(model.jnt_type[last], (1, 1))
    return int(model.jnt_qposadr[last]) + width[0], int(model.jnt_dofadr[last]) + width[1]


def hold_fingers(model: mujoco.MjModel, data: mujoco.MjData, hand: Hand | None = None) -> None:
    """Command the hand's finger actuators to its hold pose (no-op without finger actuators, as in planning models)."""
    for side in SIDES:
        set_fingers(model, data, side, closed=False, hand=hand)


def set_fingers(
    model: mujoco.MjModel, data: mujoco.MjData, side: str, closed: bool, hand: Hand | None = None, joints=None
) -> None:
    """Command one hand's finger actuators open (its hold pose) or closed (its grip); ``joints`` limits the command
    to some of them (see ``Hand.release``)."""
    hand = hand or current_hand()
    for joint in joints if joints is not None else {**hand.hold, **(hand.grip or {})}:
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_{joint.removesuffix('_joint')}")
        if actuator >= 0:
            data.ctrl[actuator] = hand.finger_target(side, joint, closed)


def set_robot_pose(model: mujoco.MjModel, data: mujoco.MjData, pose, hand: Hand | None = None) -> None:
    """Root and 29 body joints from a 36-vector; fingers at the hand's hold pose."""
    hand = hand or current_hand()
    qadr, _ = body_joint_addresses(model)
    pose = np.asarray(pose, float)
    data.qpos[:7] = pose[:7]
    data.qpos[qadr] = pose[7:]
    for side in SIDES:
        for joint in hand.hold:
            name = f"{side}_{joint}"
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
                data.qpos[model.joint(name).qposadr[0]] = hand.hold_target(side, joint)


def joint_range(model: mujoco.MjModel) -> np.ndarray:
    """[29, 2] position limits in canonical order."""
    return model.jnt_range[1 : 1 + len(JOINTS)].copy()


def descendants(model: mujoco.MjModel, root: int) -> set[int]:
    """The body ``root`` and every body below it (MuJoCo numbers a body after its parent)."""
    bodies = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    return bodies


def robot_bodies(model: mujoco.MjModel) -> set[int]:
    return descendants(model, model.body("pelvis").id)


def collision_geoms(model: mujoco.MjModel) -> list[int]:
    """Robot collision shapes (group 3), including torso mesh and hand shapes."""
    bodies = robot_bodies(model)
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies and model.geom_group[g] == 3]


def self_collision_pairs(model: mujoco.MjModel) -> list[tuple[int, int]]:
    """Robot geom pairs MuJoCo's own collision tests: not on one body or a parent and child, not excluded."""
    weld = model.body_weldid
    excluded = {(int(s) >> 16, int(s) & 0xFFFF) for s in model.exclude_signature}
    pairs = []
    geoms = collision_geoms(model)
    for i, a in enumerate(geoms):
        for b in geoms[i + 1 :]:
            body_a, body_b = int(weld[model.geom_bodyid[a]]), int(weld[model.geom_bodyid[b]])
            related = body_a == body_b or weld[model.body_parentid[body_a]] == body_b
            related = related or weld[model.body_parentid[body_b]] == body_a
            if not related and (min(body_a, body_b), max(body_a, body_b)) not in excluded:
                pairs.append((a, b))
    return pairs


def lowest_point(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> float:
    """World z of a collision shape's lowest point, including mesh support."""
    axis_z = data.geom_xmat[geom].reshape(3, 3)[2]
    size = model.geom_size[geom]
    kind = int(model.geom_type[geom])
    G = mujoco.mjtGeom
    if kind == G.mjGEOM_SPHERE:
        extent = size[0]
    elif kind == G.mjGEOM_CAPSULE:
        extent = size[0] + abs(axis_z[2]) * size[1]
    elif kind == G.mjGEOM_BOX:
        extent = np.abs(axis_z) @ size
    elif kind == G.mjGEOM_ELLIPSOID:
        extent = np.linalg.norm(axis_z * size)
    elif kind == G.mjGEOM_CYLINDER:
        extent = size[0] * np.linalg.norm(axis_z[:2]) + size[1] * abs(axis_z[2])
    elif kind == G.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        begin, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[begin : begin + count]
        return float(data.geom_xpos[geom, 2] + np.min(vertices @ axis_z))
    else:
        raise ValueError(f"unsupported collision primitive type {kind}")
    return float(data.geom_xpos[geom, 2] - extent)
