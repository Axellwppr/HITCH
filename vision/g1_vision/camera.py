"""Own a persistent native capture worker; a failed/timeout stream is never reused."""

import json
import os
import select
import subprocess
import time
import numpy as np


class Camera:
    def __init__(self, config):
        self.process = subprocess.Popen(
            [config["helper"], config["serial"], str(config.get("depth_height", 360))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        self.buffer = bytearray()

    def _read(self, count, deadline):
        while len(self.buffer) < count:
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or not select.select([self.process.stdout], [], [], remaining)[0]
            ):
                raise TimeoutError("camera capture timeout")
            chunk = os.read(
                self.process.stdout.fileno(), max(4096, count - len(self.buffer))
            )
            if not chunk:
                raise RuntimeError("camera worker exited")
            self.buffer.extend(chunk)
        result = bytes(self.buffer[:count])
        del self.buffer[:count]
        return result

    def capture(self, native=False):
        try:
            self.process.stdin.write(b"capture_native\n" if native else b"capture\n")
            deadline = time.monotonic() + 10
            header = bytearray()
            while not header.endswith(b"\n"):
                header += self._read(1, deadline)
                if len(header) > 8192:
                    raise RuntimeError("invalid camera worker header")
            meta = json.loads(header)
            h, w = meta["height"], meta["width"]
            if (h, w) != ((800, 1280) if native else (400, 640)):
                raise RuntimeError("unexpected camera worker dimensions")
            rgb = (
                np.frombuffer(self._read(h * w * 3, deadline), np.uint8)
                .reshape(h, w, 3)
                .copy()
            )
            depth = (
                np.frombuffer(self._read(h * w * 4, deadline), "<f4")
                .reshape(h, w)
                .copy()
            )
            return rgb, depth, meta
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
