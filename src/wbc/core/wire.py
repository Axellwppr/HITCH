"""JSON-safe encoding of numpy arrays: NPY in base64 (so NaN depth survives), and on reading also the compact lossless
image encoding a remote camera sends (``zlib-delta-v1``, written by vision/g1_vision/wire.py: rows
delta-coded along x with unsigned wraparound, zlib-compressed; depth as uint16 units of ``scale`` metres, 0 missing)."""

import base64
import io
import zlib

import numpy as np

CODEC = "zlib-delta-v1"


def pack(array) -> str:
    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def unpack(value) -> np.ndarray:
    if isinstance(value, dict):
        return _decode(value)
    return np.load(io.BytesIO(base64.b64decode(value, validate=True)), allow_pickle=False)


def _decode(value: dict) -> np.ndarray:
    if value.get("codec") != CODEC:
        raise ValueError(f"unknown array codec {value.get('codec')!r}")
    dtype = np.dtype(value["dtype"]).newbyteorder("<")
    raw = zlib.decompress(base64.b64decode(value["data"], validate=True))
    delta = np.frombuffer(raw, dtype).reshape(value["shape"])
    array = np.cumsum(delta, axis=1, dtype=dtype).astype(dtype.newbyteorder("="))
    if "scale" not in value:
        return array
    depth = array.astype(np.float32) * np.float32(value["scale"])
    depth[array == 0] = np.nan
    return depth
