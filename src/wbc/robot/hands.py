"""The hands the G1 model can carry, one per wrist, and where the planner's palm point sits on each.

A hand is chosen per process with ``WBC_HAND`` (default ``rubber``); the simulator, planner, checks and agent of one
session must agree, so the launcher sets it for all of them. Every hand supplies:

- ``palm``: the planner's end-effector point, the centre of the palm face, in the left wrist-yaw frame (x along the
  fingers, the left palm facing -y, z up when the palm faces inward); the right hand mirrors y. IK, palm targets and
  the palm positions the agent reads all refer to this point (site ``{side}_palm``).
- ``mount(root, side, simulate)``: its visual, planning (group 3, no physics) and contact (group 4) shapes on the
  wrist-yaw link, and, in a simulator model, finger joints held by position actuators. Planning models keep fingers
  fixed at the hold pose, so the 29 body joints stay the model's only hinges.
- ``hold``: the finger targets the simulator commands throughout (empty for a hand without finger joints).

To add a hand, write its mount and register it in ``HANDS``, and give the agent its description in ``wbc.core.hands``
(``PROMPTS``, and ``GRIPPING`` if it has ``grip``).
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from wbc.core.hands import hand_name
from wbc.paths import G1_ASSETS

# Sliding, torsional (m) and rolling friction of the hands' contact surfaces. The torsional term is about a
# centimetre of contact radius, as for a palm-sized rubber pad.
HAND_FRICTION = (0.8, 0.01, 0.0001)
# Contact geometry of the rubber hand: capsules along the palm, and a flat pad where the curled fingertips of the real
# hand meet a face they press on (4.5 cm out from the wrist axis, 14.5-17 cm along the hand, 6.5 cm high). A flat
# pad gives an area contact on a flat face, as the rubber does; the wrist stays 1.3 cm clear of it.
RUBBER_CAPSULES = ("palm", "knuckle", "finger_root")
RUBBER_PAD = {"size": (0.0125, 0.010, 0.0325), "pos": (0.1575, 0.035, 0.0025)}  # pos y: toward the palm side
# Dex3-1 finger drives: stiffness 100 N m/rad, damping 10 N m s/rad, armature 0.1, all targets 0 (open). Torque stays
# within the hand's own limits (2.45 N m thumb base, 1.4 N m elsewhere; actuatorfrcrange in the Unitree description).
DEX3 = G1_ASSETS / "hands" / "dex3_1.xml"
DEX3_DRIVE = {"kp": 100.0, "kv": 10.0, "armature": 0.1}
DEX3_FRICTION = HAND_FRICTION  # the Dex3's palm and finger pads are rubber, as the rubber hand's
DEX3_SURFACE = {"friction": " ".join(map(str, DEX3_FRICTION)), "condim": "4"}


RELEASE_LAG_S = 0.3  # the fingers open this long after the thumb (Hand.release)


def _hold_angle(hold: dict[str, float], side: str, joint: str) -> float:
    """``hold``'s angle of a finger joint (without its side) on either hand: finger flexion is mirrored."""
    target = hold.get(joint, 0.0)
    return target if side == "left" or "thumb" in joint else -target


@dataclass(frozen=True)
class Hand:
    name: str
    palm: tuple[float, float, float]
    mount: Callable[[ET.Element, str, bool], list[ET.Element]]
    hold: dict[str, float] = field(default_factory=dict)  # left finger joint (without "left_") -> target (rad)
    # Added to the left arm's joints of the start pose (``rest``), mirrored on the right: room the hand needs
    # while the arms hang at the start.
    rest_offsets: dict[str, float] = field(default_factory=dict)
    grip: dict[str, float] | None = None  # closed fingers (left joint without "left_" -> rad); None: no finger control

    def hold_target(self, side: str, joint: str) -> float:
        """The held angle of a finger joint (without its side) on either hand: finger flexion is mirrored."""
        return _hold_angle(self.hold, side, joint)

    def finger_target(self, side: str, joint: str, closed: bool) -> float:
        """The commanded angle of a finger joint (without its side): held open, or closed (``grip``).

        The right hand mirrors every flexion of the left (its joint ranges are negated), except the thumb's base
        rotation (thumb_0, symmetric).
        """
        if not closed:
            return self.hold_target(side, joint)
        target = (self.grip or {}).get(joint, 0.0)
        return target if side == "left" or joint.startswith("hand_thumb_0") else -target

    def release(self) -> list[tuple[float, list[str]]]:
        """Opening a closed hand in stages, (delay s, joints): the thumb first, then the fingers RELEASE_LAG_S later.
        Opened at once, the two fingers let go first and the still-squeezing thumb pushes a tall object it holds
        from the side over."""
        joints = list({**self.hold, **(self.grip or {})})
        thumb = [j for j in joints if "thumb" in j]
        return [(0.0, thumb), (RELEASE_LAG_S, [j for j in joints if j not in thumb])]

    def palm_position(self, side: str) -> tuple[float, float, float]:
        x, y, z = self.palm
        return (x, y if side == "left" else -y, z)

    def rest(self, pose):
        """The simulator's start pose (root + 29 joints) for this hand."""
        from wbc.core.layout import JOINTS

        pose = pose.copy()
        for joint, offset in self.rest_offsets.items():
            for side, sign in (("left", 1), ("right", -1)):
                mirrored = sign if joint.endswith(("_roll", "_yaw")) else 1
                pose[7 + JOINTS.index(f"{side}_{joint}_joint")] += mirrored * offset
        return pose


def _wrist(root: ET.Element, side: str) -> ET.Element:
    return root.find(f".//body[@name='{side}_wrist_yaw_link']")


def _mount_rubber(root: ET.Element, side: str) -> list[ET.Element]:
    """The G1's fixed rubber hand: its mesh for planning, a fingertip pad for contact (see RUBBER_PAD)."""
    hand = _wrist(root, side)
    for part in RUBBER_CAPSULES:
        hand.remove(hand.find(f"geom[@name='{side}_hand_{part}_collision']"))
    visual = hand.find(f"geom[@mesh='{side}_rubber_hand']")
    ET.SubElement(
        hand,
        "geom",
        name=f"{side}_hand_collision",
        type="mesh",
        mesh=f"{side}_rubber_hand",
        pos=visual.get("pos"),
        quat=visual.get("quat", "1 0 0 0"),
        group="3",
        contype="0",
        conaffinity="0",
        density="0",
    )
    x, y, z = RUBBER_PAD["pos"]
    ET.SubElement(
        hand,
        "geom",
        name=f"{side}_hand_palm_collision",
        type="box",
        size=" ".join(map(str, RUBBER_PAD["size"])),
        pos=f"{x} {(-1 if side == 'left' else 1) * y} {z}",
        group="4",
        contype="1",
        conaffinity="1",
        density="0",
        friction=" ".join(map(str, HAND_FRICTION)),
        condim="4",
    )
    return []


# Flat pads on the Dex3's palm-side faces, as the rubber covering of its palm and fingers, in each link's frame for
# the left hand (the right mirrors y), 1 mm proud of the mesh: they give an area contact on a flat face.
DEX3_PADS = {  # link: (centre, half size)
    "palm": ((0.035, -0.0188, 0.0), (0.032, 0.0025, 0.030)),
    "index_0": ((0.032, -0.0132, 0.001), (0.012, 0.0025, 0.009)),
    "middle_0": ((0.032, -0.0132, -0.001), (0.012, 0.0025, 0.009)),
    "index_1": ((0.0225, -0.0073, 0.0), (0.0175, 0.0025, 0.010)),
    "middle_1": ((0.0225, -0.0073, 0.0), (0.0175, 0.0025, 0.010)),
}


def _pad(parent: ET.Element, side: str, part: str, offset=(0.0, 0.0, 0.0)) -> None:
    (x, y, z), size = DEX3_PADS[part]
    mirror = 1 if side == "left" else -1
    ET.SubElement(
        parent,
        "geom",
        name=f"{side}_hand_{part}_pad",
        type="box",
        size=" ".join(map(str, size)),
        pos=f"{x + offset[0]:.4f} {mirror * y + offset[1]:.4f} {z + offset[2]:.4f}",
        group="4",
        contype="1",
        conaffinity="1",
        density="0",
        **DEX3_SURFACE,
    )


def _shapes(parent: ET.Element, name: str, mesh: str, pos: str = "0 0 0") -> None:
    """A Dex3 link's visual mesh, its planning shape and its contact shape (MuJoCo collides the mesh's hull)."""
    ET.SubElement(parent, "geom", {"class": "visual"}, mesh=mesh, pos=pos, material="dex3")
    common = dict(type="mesh", mesh=mesh, pos=pos, density="0")
    ET.SubElement(parent, "geom", name=f"{name}_collision", group="3", contype="0", conaffinity="0", **common)
    ET.SubElement(
        parent,
        "geom",
        name=f"{name}_contact",
        group="4",
        contype="1",
        conaffinity="1",
        **DEX3_SURFACE,
        **common,
    )


def _mount_dex3(root: ET.Element, side: str, simulate: bool) -> list[ET.Element]:
    """Unitree Dex3-1 (three fingers): the Unitree description's links on the wrist-yaw link, open."""
    asset = root.find("asset")
    if asset.find("material[@name='dex3']") is None:
        ET.SubElement(asset, "material", name="dex3", rgba="0.25 0.25 0.27 1")
    for part in ("palm", "thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"):
        ET.SubElement(asset, "mesh", name=f"{side}_hand_{part}_link", file=f"{side}_hand_{part}_link.STL")
    wrist = _wrist(root, side)
    for geom in [
        g for g in wrist.findall("geom") if "rubber" in (g.get("mesh") or "") or "_hand_" in g.get("name", "")
    ]:
        wrist.remove(geom)
    source = ET.parse(DEX3).getroot().find(f"hand[@side='{side}']")
    wrist.remove(wrist.find("inertial"))
    wrist.insert(0, copy.deepcopy(source.find("inertial")))
    palm = source.find("palm")
    _shapes(wrist, f"{side}_hand_palm", palm.get("mesh"), palm.get("pos"))
    _pad(wrist, side, "palm", tuple(float(v) for v in palm.get("pos").split()))
    actuators = []
    for chain in source.findall("body"):
        chain = copy.deepcopy(chain)
        for body in chain.iter("body"):
            mesh = next(g.get("mesh") for g in body.findall("geom") if g.get("mesh"))
            for geom in body.findall("geom"):
                body.remove(geom)
            _shapes(body, body.get("name").removesuffix("_link"), mesh)
            part = body.get("name").removeprefix(f"{side}_hand_").removesuffix("_link")
            if part in DEX3_PADS:
                _pad(body, side, part)
            joint = body.find("joint")
            if not simulate:  # fixed at the held angle
                angle = _hold_angle(DEX3_HOLD, side, joint.get("name").removeprefix(f"{side}_"))
                axis = np.array([float(v) for v in joint.get("axis").split()])
                body.set("quat", " ".join(f"{v:.6f}" for v in (np.cos(angle / 2), *(np.sin(angle / 2) * axis))))
                body.remove(joint)
                continue
            joint.set("armature", str(DEX3_DRIVE["armature"]))
            limit = joint.attrib.pop("actuatorfrcrange")
            actuators.append(
                ET.Element(
                    "position",
                    name=joint.get("name").removesuffix("_joint"),
                    joint=joint.get("name"),
                    kp=str(DEX3_DRIVE["kp"]),
                    kv=str(DEX3_DRIVE["kv"]),
                    ctrlrange=joint.get("range"),
                    forcerange=limit,
                )
            )
        wrist.append(chain)
    if simulate:
        # The links' hulls overlap where they join (the palm plate covers the thumb's second link): no contact
        # within one hand. Contact with the rest of the robot, the other hand included, stays.
        contact = root.find("contact")
        links = [wrist.get("name"), *(b.get("name") for b in wrist.iter("body") if b is not wrist)]
        for i, a in enumerate(links):
            for b in links[i + 1 :]:
                ET.SubElement(contact, "exclude", body1=a, body2=b)
    return actuators


DEX3_FINGERS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
# Held pose (left hand; the right mirrors finger flexion): open, the thumb out and the two fingers straight.
DEX3_HOLD = {f"hand_{f}_joint": 0.0 for f in DEX3_FINGERS}
DEX3_REST = {"shoulder_roll": 0.05, "wrist_roll": 1.57}
# Closed (left hand): the fingers and the thumb curled to their limits; the actuators' torque limits set the squeeze.
DEX3_GRIP = {
    "hand_thumb_0_joint": 0.0,
    "hand_thumb_1_joint": 1.0,
    "hand_thumb_2_joint": 1.74,
    "hand_middle_0_joint": -1.57,
    "hand_middle_1_joint": -1.74,
    "hand_index_0_joint": -1.57,
    "hand_index_1_joint": -1.74,
}
HANDS = {
    # Centre of the palm face. Rubber: the flat palm, 1.0 cm out from the wrist axis, between the wrist (4.5 cm) and
    # the finger roots (11 cm). Dex3: the palm plate's face, 1.7 cm out, between the thumb base (ends 7.9 cm along)
    # and the finger roots (11.2 cm).
    "rubber": Hand("rubber", (0.080, -0.010, 0.005), lambda root, side, simulate: _mount_rubber(root, side)),
    # Its rest pose rolls the wrists 90 degrees (thumbs pointing back) with a little more shoulder roll, which keeps
    # the open hand clear of the legs, pelvis and torso while walking. The ready (home) pose holds the hands in front.
    "dex3": Hand(
        "dex3",
        (0.090, -0.017, 0.0),
        _mount_dex3,
        DEX3_HOLD,
        DEX3_REST,
        DEX3_GRIP,
    ),
}


def current_hand() -> Hand:
    """The hand this process models: ``WBC_HAND``, else the rubber hand."""
    return HANDS[hand_name()]
