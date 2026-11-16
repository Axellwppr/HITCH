"""Simulator state ring for the processes that draw the scene (seqlock against torn reads).

Only the simulator writes; the simulated camera and the viewer window read. It carries the scene's
qpos solely so they can draw it.
"""

from pathlib import Path

import numpy as np


class FrameRing:
    def __init__(self, path, *, nq=None, nv=None, capacity=256):
        self.path = Path(path)
        if nq is not None:
            dtype = np.dtype(
                [
                    ("write_sequence", "<i8"),
                    ("frame", "<i8"),
                    ("stamp_ns", "<i8"),
                    ("time", "<f8"),
                    ("qpos", "<f8", (nq,)),
                    ("qvel", "<f8", (nv,)),
                ]
            )
            self.data = np.lib.format.open_memmap(self.path, mode="w+", dtype=dtype, shape=(capacity,))
            self.data["frame"] = -1
        else:
            self.data = np.load(self.path, mmap_mode="r", allow_pickle=False)

    def write(self, frame, stamp_ns, data):
        row = self.data[frame % len(self.data)]
        write_sequence = int(row["write_sequence"])
        row["write_sequence"] = write_sequence + 1
        row["frame"] = frame
        row["stamp_ns"] = stamp_ns
        row["time"] = data.time
        row["qpos"] = data.qpos
        row["qvel"] = data.qvel
        row["write_sequence"] = write_sequence + 2

    def latest(self):
        """The most recent complete sample."""
        frames = np.asarray(self.data["frame"])
        if frames.max() < 0:
            raise ValueError("no simulator frame yet")
        return self.read(int(frames.max()))

    def read(self, frame):
        row = self.data[int(frame) % len(self.data)]
        write_sequence = int(row["write_sequence"])
        if write_sequence % 2:
            raise ValueError("camera frame is being written; retry")
        result = row.copy()
        if write_sequence != int(row["write_sequence"]) or int(result["frame"]) != frame:
            raise ValueError("camera frame expired; retry")
        return result
