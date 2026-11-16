"""Vision service: ``GET /vision`` returns the latest aligned RGB-D frame.

``SimCamera`` renders the robot's ego camera from the simulator frame ring; ``POST /vision`` with ``{"eye": {yaw_deg,
pitch_deg}}`` renders it from that setting of the head mount (wbc.robot.eye), which is not in the model. Real hardware
is served independently by vision/.

  python -m wbc_link_server.sensors.vision --directory SIM_DIR
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core import logs
from wbc.core.eye import Eye
from wbc.core.jsonhttp import JsonServer, wait_for_signal, write_json_atomic
from wbc.link.protocol import VisionFrame
from wbc.robot import eye as head
from wbc_link_server.sensors.frame_ring import FrameRing

log = logging.getLogger("wbc_link_server.sensors.vision")
OPTICAL_FROM_MUJOCO = np.diag([1.0, -1.0, -1.0])  # MuJoCo cameras look along -z with +y up


class SimCamera:
    timeout = 30.0  # seconds one capture may take

    def __init__(self, directory: Path, width: int = head.WIDTH, height: int = head.HEIGHT, max_depth: float = 5.0):
        self.ring = FrameRing(directory / "camera_frames.npy")
        self.model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = width, height
        self.model.vis.quality.offsamples = 0
        # MuJoCo stores clipping distances relative to scene extent. Keep the
        # ego camera's near plane at 5 mm, including in large rooms, so nearby
        # surfaces continue to occlude objects in both RGB and depth.
        self.model.vis.map.znear = 0.005 / self.model.stat.extent
        self.width, self.height, self.max_depth = width, height, max_depth
        self.directory = directory
        self.renderer = None
        self.source = "sim-" + uuid.uuid4().hex[:8]
        camera = self.model.camera("ego")
        self.camera_id = camera.id
        self.link = self.model.body(int(camera.bodyid[0])).name
        # the head mount moves the camera only where it is mounted: on the torso
        self.movable = self.link == "torso_link"
        self.fixed_pose = (camera.pos.copy(), camera.quat.copy())
        self.link_to_camera = self._link_to_camera()
        focal = height / (2 * np.tan(np.deg2rad(float(camera.fovy[0])) / 2))
        self.intrinsics = {"fx": focal, "fy": focal, "cx": (width - 1) / 2, "cy": (height - 1) / 2}

    def _link_to_camera(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = (
            Rotation.from_quat(self.model.cam_quat[self.camera_id], scalar_first=True).as_matrix() @ OPTICAL_FROM_MUJOCO
        )
        pose[:3, 3] = self.model.cam_pos[self.camera_id]
        return pose

    def _look(self, eye: dict | None) -> dict | None:
        """Place the ego camera for ``eye`` (None: where the model has it); the setting used, None when fixed."""
        if not self.movable:
            return None
        if eye is None:
            self.model.cam_pos[self.camera_id], self.model.cam_quat[self.camera_id] = self.fixed_pose
            self.link_to_camera = self._link_to_camera()
            return None
        setting = Eye.from_json(eye)
        position, rotation = head.mujoco_in_torso(setting)
        self.model.cam_pos[self.camera_id] = position
        self.model.cam_quat[self.camera_id] = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
        self.link_to_camera = self._link_to_camera()
        return setting.to_json()

    def capture(self, eye: dict | None = None) -> VisionFrame:
        m, d = self.model, self.data
        used = self._look(eye)
        for _attempt in range(5):
            try:
                sample = self.ring.latest()
                break
            except ValueError:
                time.sleep(0.002)
        else:
            raise RuntimeError("simulator frames unavailable")
        d.qpos[:] = sample["qpos"]
        d.qvel[:] = sample["qvel"]
        d.time = float(sample["time"])
        mujoco.mj_forward(m, d)
        if self.renderer is None:
            self.renderer = mujoco.Renderer(m, height=self.height, width=self.width)
        self.renderer.disable_segmentation_rendering()
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(d, camera="ego")
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().astype(np.float32)
        near = m.vis.map.znear * m.stat.extent
        depth[(depth <= near) | (depth >= self.max_depth) | ~np.isfinite(depth)] = np.nan
        frame = int(sample["frame"])
        return VisionFrame(
            rgb,
            depth,
            self.intrinsics,
            self.link,
            self.link_to_camera,
            self.source,
            frame,
            used,
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


class CaptureThread:
    """Runs every capture of ``provider`` on one thread, one at a time.

    HTTP requests arrive on many threads, but a renderer's GL context and a
    camera pipeline belong to one; concurrent requests queue up here.
    """

    def __init__(self, provider, timeout: float = 30.0):
        self.provider, self.timeout = provider, timeout
        self.requests: queue.Queue = queue.Queue()  # (reply, eye)
        self.thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while (item := self.requests.get()) is not None:
                reply, eye = item
                if not reply.set_running_or_notify_cancel():
                    continue  # its caller gave up waiting
                try:
                    reply.set_result(self.provider.capture(eye))
                except BaseException as error:
                    reply.set_exception(error)
        finally:
            self.provider.close()  # on the thread that owns its resources

    def capture(self, eye: dict | None = None) -> VisionFrame:
        reply: Future = Future()
        self.requests.put((reply, eye))
        try:
            return reply.result(self.timeout)
        except FutureTimeout:  # not the built-in TimeoutError before Python 3.11
            reply.cancel()
            raise

    def close(self) -> None:
        self.requests.put(None)
        self.thread.join(timeout=10)


def serve(provider, port: int, endpoint_file: Path | None, host: str = "127.0.0.1"):
    camera = CaptureThread(provider, timeout=provider.timeout)
    server = JsonServer(
        {
            ("GET", "/vision"): lambda: camera.capture().to_wire(),
            ("POST", "/vision"): lambda payload: camera.capture(payload.get("eye")).to_wire(),
            ("POST", "/aim"): lambda payload: {},  # a simulated camera is placed at the capture
        },
        host,
        port,
        quiet=("/vision", "/aim"),
    ).start()
    log.info("vision at %s", server.url)
    if endpoint_file:
        write_json_atomic(endpoint_file, {"vision": server.url})
    try:
        wait_for_signal()
    finally:
        server.close()
        camera.close()


def main() -> None:
    """Serve a ``SimCamera`` on the simulator directory given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", type=Path, required=True, help="simulator directory")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--endpoint-file", type=Path)
    args = parser.parse_args()
    logs.setup("vision", logs.session_of(args.directory))
    serve(SimCamera(args.directory), args.port, args.endpoint_file, args.host)


if __name__ == "__main__":
    main()
