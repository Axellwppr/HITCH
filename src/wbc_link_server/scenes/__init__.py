"""Simulator scenes (``sim_bridge.Bridge``), in one table. A scene kind has ``make_scene(seed)`` (the spec: its
``kind``, ``objects``, see wbc.robot.model._add_object, and the task) and ``build(spec, xml_path)``
(the MuJoCo model, also written to ``xml_path``). Here are ``flat``, the bare floor, and ``tabletop_pick``,
a household room with a single-hand pick-and-place task (``success`` checks its outcome). The settle before the first
step is the bridge's (``sim_bridge.initialize_scene``).

Scenes exist only inside the simulator; the harness never sees object state.
"""

from __future__ import annotations

import mujoco

from wbc.robot.model import build_model
from wbc_link_server.scenes import tabletop_pick


def flat(seed=0):
    return {"kind": "flat", "seed": seed, "objects": []}


def build_objects(spec, xml_path=None):
    """A spec's ``objects`` on the bare floor (``flat``)."""
    model = build_model(spec["objects"], simulate=True)
    if xml_path is not None:
        mujoco.mj_saveLastXML(str(xml_path), model)
    return model


SCENES = {"flat": flat, tabletop_pick.KIND: tabletop_pick.make_scene}  # name -> make_scene
BUILDERS = {"flat": build_objects, tabletop_pick.KIND: tabletop_pick.build}  # spec kind -> build


def scene(name: str, seed: int = 0) -> dict:
    if name not in SCENES:
        raise ValueError(f"unknown scene {name!r}; available: {sorted(SCENES)}")
    return SCENES[name](seed)


def build_scene(spec, *, xml_path=None):
    """The MuJoCo model of a scene spec, built by its kind's builder."""
    if spec["kind"] not in BUILDERS:
        raise ValueError(f"no scene builds kind {spec['kind']!r}; known: {sorted(BUILDERS)}")
    return BUILDERS[spec["kind"]](spec, xml_path)
