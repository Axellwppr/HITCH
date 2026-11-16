"""LingBot-Depth v0.5 as an HTTP service: an RGB image and its aligned measured depth in, the model's
depth out. The official code runs on its own default path (xFormers attention).

POST /refine  {"rgb": array uint8 H x W x 3, "depth": array float32 H x W in metres (NaN, inf or <= 0: missing),
               "mode": "full" (default: the model's depth wherever it gives one, measured depth only where it gives
                       none) or "fill" (measured depth kept, holes filled by the model),
               "resolution_level": 0-9 (default 9, the official default)}
          ->  {"depth": array float32 H x W (NaN: none), "provenance": array uint8 H x W (0 none, 1 measured kept,
               2 hole filled by the model, 3 measurement replaced by the model), "mode", "infer_ms", "model"}
GET /health -> {"ok", "model", "weights_sha256", "device", "torch", "xformers", "requests"}

Arrays travel as wbc.core.wire packs them: numpy .npy bytes in base64. One request at a time per process (one GPU).
usage: serve.sh [GPU] [PORT] [HOST]   (python lingbot_server.py --repo DIR --weights FILE [--host H] [--port P])
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

MODEL = "lingbot-depth-pretrain-vitl-14-v0.5"


def pack(array) -> str:
    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def unpack(text: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(text)), allow_pickle=False)


class Refiner:
    def __init__(self, repo: Path, weights: Path):
        import torch

        sys.path.insert(0, str(repo))
        from mdm.model.v2 import MDMModel

        self.torch = torch
        self.model = MDMModel.from_pretrained(weights).cuda().eval()
        self.lock = threading.Lock()
        self.requests = 0
        digest = hashlib.sha256()
        with open(weights, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 24), b""):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()

    def predict(self, rgb: np.ndarray, depth: np.ndarray, resolution_level: int = 9) -> np.ndarray:
        """The model's metric depth (H x W, inf or NaN where it gives none), by the official inference call."""
        torch = self.torch
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).cuda().float() / 255
        measured = torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32)).cuda()
        with torch.inference_mode():
            output = self.model.infer(image, depth_in=measured, apply_mask=True, resolution_level=resolution_level,
                                      use_fp16=True)
        return output["depth"].float().cpu().numpy()

    def refine(self, rgb, depth, mode="full", resolution_level=9) -> dict:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8 H x W x 3, not {rgb.dtype} {rgb.shape}")
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"depth {depth.shape} is not aligned with rgb {rgb.shape[:2]}")
        if mode not in ("full", "fill"):
            raise ValueError("mode is 'full' or 'fill'")
        depth = depth.astype(np.float32)
        measured = np.isfinite(depth) & (depth > 0)
        with self.lock:
            self.torch.cuda.synchronize()
            began = time.perf_counter()
            predicted = self.predict(rgb, np.where(measured, depth, 0).astype(np.float32), int(resolution_level))
            self.torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - began) * 1000
            self.requests += 1
        model = np.isfinite(predicted) & (predicted > 0)
        if mode == "full":
            out = np.where(model, predicted, np.where(measured, depth, np.nan))
            provenance = np.where(model, np.where(measured, 3, 2), np.where(measured, 1, 0))
        else:
            out = np.where(measured, depth, np.where(model, predicted, np.nan))
            provenance = np.where(measured, 1, np.where(model, 2, 0))
        return {"depth": out.astype(np.float32), "provenance": provenance.astype(np.uint8), "mode": mode,
                "infer_ms": infer_ms, "model": MODEL}

    def health(self) -> dict:
        import xformers

        torch = self.torch
        return {"ok": True, "model": MODEL, "weights_sha256": self.sha256, "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "xformers": xformers.__version__, "requests": self.requests}


def serve(refiner: Refiner, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/health":
                return self.reply(200, refiner.health())
            self.reply(404, {"ok": False, "error": "GET /health or POST /refine"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/refine":
                return self.reply(404, {"ok": False, "error": "POST /refine"})
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                result = refiner.refine(unpack(request["rgb"]), unpack(request["depth"]), request.get("mode", "full"),
                                        request.get("resolution_level", 9))
            except (KeyError, ValueError) as error:
                return self.reply(400, {"ok": False, "error": repr(error)})
            except Exception as error:  # reported to the caller, the service stays up
                return self.reply(500, {"ok": False, "error": repr(error)})
            self.reply(200, {"ok": True, **result, "depth": pack(result["depth"]),
                             "provenance": pack(result["provenance"])})

        def log_message(self, fmt, *args):  # one line per request on stdout
            print(f"{time.strftime('%H:%M:%S')} {self.address_string()} {fmt % args}", flush=True)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"serving {MODEL} on http://{host}:{port} ({refiner.health()})", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True, help="the official lingbot-depth checkout")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18890)
    args = parser.parse_args()
    refiner = Refiner(args.repo, args.weights)
    warm = refiner.refine(np.zeros((400, 640, 3), np.uint8), np.ones((400, 640), np.float32))  # kernels and caches
    print(f"warm-up {warm['infer_ms']:.0f} ms", flush=True)
    serve(refiner, args.host, args.port)


if __name__ == "__main__":
    main()
