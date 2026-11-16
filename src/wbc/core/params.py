"""Strict parsing of skill parameters from JSON into frozen dataclasses.

A Params class is a frozen dataclass. Field metadata may carry
``range=(lo, hi)``, ``length=n``, ``doc="..."`` and ``frame`` (``"xy"``,
``"yaw_deg"`` or ``"yaw"`` (internal radians): an observation-frame coordinate the planner
re-expresses in each step's own frame). Unknown keys are rejected,
missing keys take the field default, and every error names the offending path.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import field, replace
from typing import Any, Literal, TypeVar

import numpy as np

from wbc.core.errors import invalid
from wbc.core.frames import SE2, wrap

T = TypeVar("T")


def param(
    default: Any = dataclasses.MISSING,
    *,
    range: tuple[float, float] | None = None,
    length: int | tuple[int, int] | None = None,
    doc: str = "",
    frame: Literal["xy", "yaw_deg", "yaw"] | None = None,
) -> Any:
    """A Params field: its default (none: required) and the metadata ``parse`` and ``reframe`` read."""
    return field(default=default, metadata={"range": range, "length": length, "doc": doc, "frame": frame})


def _convert(value, hint, path, meta):
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        options = [a for a in args if a is not type(None)]
        if len(options) == 1:
            return _convert(value, options[0], path, meta)
        raise TypeError(f"unsupported union {hint}")
    if hint is Any:
        return value
    if origin is Literal:
        if value not in args:
            raise invalid(f"{path}: expected one of {list(args)}")
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise invalid(f"{path}: expected true/false")
        return value
    if hint in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise invalid(f"{path}: expected a number")
        if not np.isfinite(value):
            raise invalid(f"{path}: must be finite")
        if hint is int and float(value) != int(value):
            raise invalid(f"{path}: expected an integer")
        number = hint(value)
        bounds = meta.get("range")
        if bounds is not None and not bounds[0] <= number <= bounds[1]:
            raise invalid(f"{path}: {number} outside [{bounds[0]}, {bounds[1]}]")
        return number
    if hint is str:
        if not isinstance(value, str):
            raise invalid(f"{path}: expected a string")
        return value
    if origin in (tuple, list):
        if not isinstance(value, (list, tuple)):
            raise invalid(f"{path}: expected a list")
        item = args[0] if args else Any
        length = meta.get("length")
        if isinstance(length, int) and len(value) != length:
            raise invalid(f"{path}: expected {length} items")
        if isinstance(length, tuple) and not length[0] <= len(value) <= length[1]:
            raise invalid(f"{path}: expected {length[0]}..{length[1]} items")
        inner = {k: v for k, v in meta.items() if k == "range"}
        return tuple(_convert(v, item, f"{path}[{i}]", inner) for i, v in enumerate(value))
    if origin is dict:
        if not isinstance(value, dict):
            raise invalid(f"{path}: expected an object")
        return dict(value)
    if dataclasses.is_dataclass(hint):
        return parse(hint, value, path)
    raise TypeError(f"unsupported parameter type {hint}")


def parse(cls: type[T], data: dict | None, path: str = "params") -> T:
    """``data`` (JSON) as a ``cls``, checked strictly; every error is an INVALID_REQUEST naming ``path``."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise invalid(f"{path}: expected an object")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise invalid(f"{path}: unknown keys {sorted(unknown)}; allowed {sorted(fields)}")
    values = {}
    for name, f in fields.items():
        if name in data:
            values[name] = _convert(data[name], hints[name], f"{path}.{name}", dict(f.metadata))
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise invalid(f"{path}.{name} is required")
    try:
        return cls(**values)
    except ValueError as error:
        raise invalid(f"{path}: {error}") from None


def to_json(value: Any) -> Any:
    """A params tree (dataclasses, tuples, arrays) as plain JSON values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_json(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (tuple, list)):
        return [to_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def reframe(value: T, transform: SE2) -> T:
    """Copy of a params tree with every plan-frame coordinate mapped by ``transform``."""
    if isinstance(value, tuple):
        return tuple(reframe(v, transform) for v in value)
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        return value
    changes = {}
    for f in dataclasses.fields(value):
        item, kind = getattr(value, f.name), f.metadata.get("frame")
        if item is None:
            continue
        if kind == "xy":
            changes[f.name] = tuple(transform.apply_xy(np.asarray(item, float)).tolist())
        elif kind == "yaw":
            changes[f.name] = float(wrap(float(item) + transform.yaw))
        elif kind == "yaw_deg":
            # Integer degrees constrain public input. Frame transforms retain
            # full precision so a fractional starting heading cannot shift a goal.
            changes[f.name] = float(np.degrees(wrap(np.radians(float(item)) + transform.yaw)))
        else:
            changes[f.name] = reframe(item, transform)
    return replace(value, **changes)
