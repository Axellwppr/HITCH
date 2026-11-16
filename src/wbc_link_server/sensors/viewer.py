"""An interactive window on a running simulator, for a machine with a display.

  python -m wbc_link_server.sensors.viewer --directory SIM_DIR      (SIM_DIR: data/sessions/SES/runtime/sim)

It reads the simulator's frame ring, as the simulated camera does, and writes nothing: closing the window, or this
process failing, leaves the simulation as it is. A ``sim://...&viewer=1`` target starts it with the simulator.

The view follows the robot's pelvis and starts in front of the robot, a little to its left and above it, looking
down at it; the mouse turns and zooms it as in any MuJoCo viewer.
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path

import mujoco
import numpy as np

from wbc.core import logs
from wbc_link_server.sensors.frame_ring import FrameRing

log = logging.getLogger("wbc_link_server.sensors.viewer")

ROOT = "pelvis"  # the body the view follows
DISTANCE_M, ELEVATION_DEG, SIDE_DEG = 3.0, -20.0, 30.0  # the first view: how far, how steeply down, how far to the side


class SceneState:
    """The simulator's scene model, brought to the latest state the simulator has published."""

    def __init__(self, directory: Path):
        self.ring = FrameRing(directory / "camera_frames.npy")
        self.model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.frame = -1

    def update(self) -> bool:
        """Copy the latest published state into ``data``; False when there is none newer."""
        try:
            sample = self.ring.latest()
        except ValueError:  # no frame yet, or the one read was being written: the next call reads again
            return False
        frame = int(sample["frame"])
        if frame == self.frame:
            return False
        self.frame = frame
        self.data.qpos[:] = sample["qpos"]
        self.data.qvel[:] = sample["qvel"]
        self.data.time = float(sample["time"])
        mujoco.mj_forward(self.model, self.data)
        return True

    def follow(self, camera) -> None:
        """Make ``camera`` (the window's) track the robot's root, from in front of the robot as it stands now."""
        root = self.data.body(ROOT)
        forward = root.xmat.reshape(3, 3)[:, 0]
        heading = float(np.degrees(np.arctan2(forward[1], forward[0])))
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = root.id
        camera.distance = DISTANCE_M
        camera.elevation = ELEVATION_DEG
        # MuJoCo's azimuth is the direction the camera looks along: opposite the robot's heading from in front of it
        camera.azimuth = heading + 180.0 - SIDE_DEG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", type=Path, required=True, help="the simulator directory (runtime/sim)")
    parser.add_argument("--rate", type=float, default=30.0, help="window updates per second")
    args = parser.parse_args()
    logs.setup("viewer", logs.session_of(args.directory))
    from mujoco import viewer  # opens the windowing library: only where a window is wanted

    state = SceneState(args.directory)
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: stopping.append(True))
    with viewer.launch_passive(state.model, state.data, show_left_ui=False, show_right_ui=False) as window:
        log.info("viewer open on %s", args.directory)
        following = False
        while window.is_running() and not stopping:
            with window.lock():
                changed = state.update()
                if changed and not following:  # the first state says where the robot stands and faces
                    state.follow(window.cam)
                    following = True
            if changed:
                window.sync()
            time.sleep(1.0 / args.rate)


if __name__ == "__main__":
    main()
