"""The success check of ``tabletop_pick``, on a simulator directory after its simulator has stopped (the
physics log is complete then):

  python -m wbc_link_server.scenes.success SIM_DIR      # the session's ``sim`` directory; prints the result as JSON

It replays the recorded states' kinematics, without contacts:
- pick: the named object ends at least LIFT above where it started;
- place: the object was at least LIFT above its start at some moment and ends over its destination (MARGIN around
  the outline): in a bin or bucket low enough to be below the rim if it stood upright (it may lie tilted), on a mat
  at its resting height;
in both cases the robot ends upright.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

LIFT = 0.03  # m above the start: lifted
MARGIN = 0.02  # m around the destination's outline, and around the height of an object resting on a mat
UPRIGHT = 0.45  # m: the pelvis of a standing or crouching robot is above this


def states(directory: Path):
    """The recorded simulator states (qpos per control frame), in order."""
    for path in sorted(directory.glob("physics_*.npz")):
        with np.load(path, allow_pickle=False) as chunk:
            yield from chunk["qpos"]


def over(facts: dict, xy) -> bool:
    """Whether ``xy`` (world) is over the destination's inner outline, MARGIN around it."""
    d = np.asarray(xy) - np.asarray(facts["centre"])
    c, s = np.cos(facts["yaw"]), np.sin(facts["yaw"])
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    if facts["type"] in ("bucket", "disk"):
        return bool(np.linalg.norm(local) <= facts["inner_radius" if facts["type"] == "bucket" else "radius"] + MARGIN)
    return bool(np.all(np.abs(local) <= np.asarray(facts["inner_half" if facts["type"] == "bin" else "half"]) + MARGIN))


def check(directory: Path) -> dict:
    """The result for the simulator directory ``directory``: ``success`` and what it was decided on."""
    spec = json.loads((directory / "scene_spec.json").read_text())
    task = spec.get("task_spec")
    if task is None or "destination" not in task:
        raise ValueError(f"scene {spec.get('kind')!r} has no task this check knows")
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    item = model.geom(task["item"] + "_geom").id
    start = lift = None
    for qpos in states(directory):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        centre = data.geom_xpos[item].copy()
        if start is None:
            start, lift = centre, 0.0
        lift = max(lift, float(centre[2] - start[2]))
    if start is None:
        raise ValueError(f"no recorded physics in {directory} (is the simulator still running?)")
    facts = task["destination"]
    resting = facts["base_z"] + facts["floor_z"] + task["item_height"] / 2  # its centre, resting in or on it
    result = {
        "goal": task["goal"],
        "instruction": spec["task"]["instruction"],
        "upright": bool(data.qpos[2] >= UPRIGHT),
        "maximum_lift_m": lift,
        "final_lift_m": float(centre[2] - start[2]),
    }
    if task["goal"] == "pick":
        done = result["final_lift_m"] >= LIFT
    else:
        result["over_destination"] = over(facts, centre[:2])
        result["height_above_resting_m"] = float(centre[2] - resting)
        if facts["type"] in ("bin", "bucket"):  # inside it, however it lies
            result["below_rim"] = bool(centre[2] - task["item_height"] / 2 <= facts["base_z"] + facts["rim_z"])
            low = result["below_rim"]
        else:
            low = abs(result["height_above_resting_m"]) <= MARGIN
        done = lift >= LIFT and result["over_destination"] and low
    return {"success": bool(done and result["upright"]), **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="simulator directory (scene_spec.json, scene.xml, physics_*.npz)")
    try:
        result = check(parser.parse_args().directory)
    except ValueError as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
