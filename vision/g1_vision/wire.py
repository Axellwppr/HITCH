"""Compact lossless encoding of the /vision images: ``zlib-delta-v1``.

Each image row is delta-coded along x (with wraparound in its unsigned type) and the whole array zlib-compressed:
RGB as uint8; depth as uint16 integer millimetres, 0 for missing (the D455's own depth unit is 1 mm, so depth in
range survives exactly). Encoded value: ``{"codec", "dtype", "shape", "data": base64[, "scale": metres per unit]}``.
The Harness decodes it in ``wbc.core.wire.unpack`` (which also reads the NPY strings of ``server.pack``);
this copy keeps the service free of Harness imports. numpy and the standard library only (Python >= 3.8).
"""

import base64
import zlib

import numpy as np

CODEC = "zlib-delta-v1"
DEPTH_SCALE = 0.001  # metres per unit


def _encode(array, dtype, level):
    a = np.ascontiguousarray(array, dtype=np.dtype(dtype).newbyteorder("<"))
    delta = np.diff(a, axis=1, prepend=np.zeros_like(a[:, :1]))  # unsigned: wraps, undone by a wrapping cumsum
    return {
        "codec": CODEC,
        "dtype": np.dtype(dtype).name,
        "shape": list(a.shape),
        "data": base64.b64encode(zlib.compress(delta.tobytes(), level)).decode("ascii"),
    }


def encode_rgb(rgb, level=1):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3:
        raise ValueError("rgb must be uint8 H x W x C")
    return _encode(rgb, "uint8", level)


def encode_depth(metres, level=1, scale=DEPTH_SCALE):
    metres = np.asarray(metres, np.float32)
    valid = np.isfinite(metres) & (metres > 0)
    units = np.where(valid, np.rint(metres / scale), 0)
    if units.max(initial=0) > 65535:
        raise ValueError("depth beyond the uint16 range of the encoding")
    out = _encode(units, "uint16", level)
    out["scale"] = scale
    return out


def decode(value):
    """The array an encoded value holds (as wbc.core.wire.unpack decodes it)."""
    if value.get("codec") != CODEC:
        raise ValueError("unknown array codec %r" % value.get("codec"))
    dtype = np.dtype(value["dtype"]).newbyteorder("<")
    delta = np.frombuffer(zlib.decompress(base64.b64decode(value["data"])), dtype).reshape(value["shape"])
    array = np.cumsum(delta, axis=1, dtype=dtype).astype(dtype.newbyteorder("="))
    if "scale" not in value:
        return array
    depth = array.astype(np.float32) * np.float32(value["scale"])
    depth[array == 0] = np.nan
    return depth
