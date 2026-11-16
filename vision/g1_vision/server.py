"""Standalone GET/POST /vision compatible with wbc.link.protocol.VisionFrame."""

import argparse
import base64
import io
import json
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
from .camera import Camera
from . import wire
from .geometry import Mount
from .head import Head


def pack(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


class BusyError(RuntimeError):
    pass


AIM_WAIT_S = 1.0  # a capture waits this long for the bus: an aim holds it for a few writes, a capture for seconds


class Vision:
    def __init__(self, camera, head, mount, calibration_path=None, encoding=wire.CODEC):
        self.camera, self.head, self.mount = camera, head, mount
        self.calibration_path = calibration_path
        if encoding not in (wire.CODEC, "npy"):
            raise ValueError("encoding is %r or 'npy'" % wire.CODEC)
        self.encoding = encoding
        self.lock = threading.Lock()
        self.source = "g1-rgbd-" + uuid.uuid4().hex[:8]

    def calibrated(self):
        if self.calibration_path is not None and self.calibration_path.exists():
            return Mount(json.loads(self.calibration_path.read_text()))
        return self.mount

    def aim(self, eye):
        """POST /aim: send the servos toward a setting (as /vision would) and return at once; the head turns while
        the body moves and the capture after it only finishes the turn. Skipped (busy) while a capture holds the bus;
        a head slower than the body is only waited for by that capture's settle, never by this."""
        if not isinstance(eye, dict) or set(eye) != {"yaw_deg", "pitch_deg"}:
            raise ValueError("aim requires eye {yaw_deg, pitch_deg}")
        if not self.lock.acquire(blocking=False):
            raise BusyError("camera/head is busy; not aimed")
        try:
            goal = self.head.clamp(self.calibrated().servo(eye))
            self.head.aim(goal)
            return {"servo_goal": goal}
        finally:
            self.lock.release()

    def capture(self, eye=None, native=False):
        if not self.lock.acquire(timeout=AIM_WAIT_S):
            raise BusyError("camera is busy; retry after the current capture")
        try:
            mount = self.calibrated()
            # A setting (POST /vision) means the simulation's view: the servos go where this calibrated mount
            # looks along it, within their range; /head and /calibration/frame stay in servo angles.
            goal = None if eye is None else self.head.clamp(mount.servo(eye))
            before = self.head.settle(goal)
            start = time.monotonic()
            rgb, depth, meta = self.camera.capture(native=native)
            state = self.head.read()
            after = self.head.eye(state)
            if (
                state["yaw_deg_moving"]
                or state["pitch_deg_moving"]
                or max(abs(before[k] - after[k]) for k in before) > 0.3
            ):
                raise RuntimeError("head moved during exposure; retry observation")
            # The camera pose is computed from the measured head angles.
            actual = {key: (before[key] + after[key]) / 2 for key in before}
            # /vision frames travel compressed (lossless, about a quarter of NPY); the calibration frame stays NPY
            compact = self.encoding == wire.CODEC and not native
            return dict(
                rgb=wire.encode_rgb(rgb) if compact else pack(rgb),
                depth=wire.encode_depth(depth) if compact else pack(depth),
                intrinsics=meta["intrinsics"],
                link="torso_link",
                link_to_camera=mount.pose(actual).tolist(),
                source_id=self.source,
                frame_id=meta["frame_id"],
                eye=actual,
                head_feedback=getattr(self.head, "last_feedback", {}),
                capture=dict(
                    meta,
                    monotonic_start_s=start,
                    monotonic_end_s=time.monotonic(),
                    eye_before=before,
                    eye_after=after,
                    setting=eye,
                    servo_goal=goal,
                ),
            )

        finally:
            self.lock.release()

    def head_request(self, eye=None):
        if not self.lock.acquire(blocking=False):
            raise BusyError("camera/head is busy")
        try:
            actual = (
                self.head.settle(eye)
                if eye is not None
                else self.head.eye(self.head.read())
            )
            return {"eye": actual, "feedback": getattr(self.head, "last_feedback", {})}

        finally:
            self.lock.release()

    def close(self):
        self.camera.close()
        self.head.close()


def handler_for(vision):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond(None)

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 4096:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(n))
                if not isinstance(payload, dict) or set(payload) - {"eye"}:
                    raise ValueError("expected {eye: {yaw_deg, pitch_deg}}")
                self.respond(payload.get("eye"))
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(
                    400, {"error": {"code": "INVALID_REQUEST", "message": str(exc)}}
                )

        def respond(self, eye):
            if self.path not in ("/vision", "/calibration/frame", "/head", "/aim"):
                self.send_json(
                    404, {"error": {"code": "NOT_FOUND", "message": self.path}}
                )
                return
            try:
                if self.path == "/calibration/frame" and self.command != "GET":
                    self.send_json(
                        405,
                        {
                            "error": {
                                "code": "METHOD_NOT_ALLOWED",
                                "message": "calibration/frame supports GET",
                            }
                        },
                    )
                    return
                if self.path == "/aim":
                    result = vision.aim(eye)
                elif self.path == "/head":
                    result = vision.head_request(eye)
                else:
                    result = vision.capture(eye, native=self.path == "/calibration/frame")
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(
                    400, {"error": {"code": "INVALID_REQUEST", "message": str(exc)}}
                )
            except BusyError as exc:
                self.send_json(503, {"error": {"code": "BUSY", "message": str(exc)}})
            except Exception as exc:
                self.send_json(
                    503, {"error": {"code": "UNREACHABLE", "message": str(exc)}}
                )
            else:
                self.send_json(200, result)

        def send_json(self, status, payload):
            envelope = (
                {"ok": True, "result": payload}
                if status == 200
                else dict(ok=False, **payload)
            )
            data = json.dumps(envelope, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18764)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    root = a.config.resolve().parent
    calibration_path = root / config["calibration"]
    calibration = (
        json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
    )
    config["camera"]["helper"] = str(root / config["camera"]["helper"])
    mount = Mount(calibration)
    head = Head(config["head"])  # Opening does not move or energize either servo.
    try:
        camera = Camera(config["camera"])
    except BaseException:
        head.close()
        raise
    vision = Vision(camera, head, mount, calibration_path, config.get("encoding", wire.CODEC))

    def terminate(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        with ThreadingHTTPServer((a.host, a.port), handler_for(vision)) as server:
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        vision.close()


if __name__ == "__main__":
    main()
