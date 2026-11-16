"""Single-owner ST3215 control; reads actual angles, never writes EEPROM."""

import json
import logging
from pathlib import Path
import math
import time
from .bus import Bus, configuration, feedback, move_data


class Head:
    def __init__(self, config, bus=None):
        self.config = config
        self.axes = [config["yaw"], config["pitch"]]
        if not 1 <= config["speed"] <= 3400 or not 1 <= config["acceleration"] <= 255:
            raise ValueError("invalid servo speed/acceleration")
        for axis in self.axes:
            if axis["direction"] not in (-1, 1) or not 0 <= axis["zero_tick"] <= 4095:
                raise ValueError("invalid encoder direction/zero")
        if self.axes[0]["id"] == self.axes[1]["id"]:
            raise ValueError("yaw and pitch require distinct IDs")
        self.bus = bus or Bus(config["port"], config.get("baud", 1000000))

    def read(self):
        result = {}
        self.last_feedback = {}
        for name, axis in zip(("yaw_deg", "pitch_deg"), self.axes):
            pos, speed, volts, temperature, moving = feedback(self.bus, axis["id"])
            outside = []
            if not 0 <= pos <= 4095:
                outside.append("position_tick")
            if not 5 <= volts <= 8.4:
                outside.append("voltage_v")
            if temperature >= 60:
                outside.append("temperature_c")
            measurement = dict(
                id=axis["id"],
                position_tick=pos,
                speed=speed,
                voltage_v=volts,
                temperature_c=temperature,
                moving=bool(moving),
                outside_range=outside,
            )
            self.last_feedback[name] = measurement
            if outside:
                record = dict(time_s=time.time(), axis=name, **measurement)
                logging.warning("servo feedback recorded: %s", record)
                try:
                    path = (
                        Path(__file__).resolve().parents[1]
                        / "captures/servo-feedback.jsonl"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                except OSError:
                    logging.exception("Could not persist servo feedback")
            result[name] = axis["direction"] * (pos - axis["zero_tick"]) * 360 / 4096
            result[name + "_moving"] = bool(moving) or abs(speed) > 2
        return result

    @staticmethod
    def eye(state):
        return {key: state[key] for key in ("yaw_deg", "pitch_deg")}

    def settle(self, target=None):
        """Move to ``target`` ({yaw_deg, pitch_deg}; None: only wait for stillness) and return the measured eye.

        Under load an ST3215 stops short of its goal. Once still, the goal is moved by what is left, up to
        ``trim_steps`` times and ``trim_limit_deg`` from the target, until both axes are within
        ``trim_tolerance_deg`` of it. Once still it returns, whatever is left: beyond ``target_tolerance_deg`` only a
        warning goes to the service's log."""
        deadline = time.monotonic() + self.config.get("timeout_s", 8)
        if target is None:
            return self._still(deadline)
        target, ticks, limits = self._checked(target)
        start = self.eye(self.read())
        self._goal(ticks, limits)
        # still before it started moving is not arrival: allow the travel at the commanded speed first
        travel = max(abs(target[k] - start[k]) for k in target) * 4096 / 360 / self.config["speed"]
        eye = self._still(deadline, time.monotonic() + travel + 0.3, target)
        aim = dict(target)
        bound = self.config.get("trim_limit_deg", 3)
        for _ in range(self.config.get("trim_steps", 3)):
            if max(abs(target[k] - eye[k]) for k in target) <= self.config.get("trim_tolerance_deg", 0.3):
                break
            aim = self.clamp({k: min(target[k] + bound, max(target[k] - bound, aim[k] + target[k] - eye[k]))
                              for k in aim})
            ticks = [min(high, max(low, self._tick(axis, aim[name])))
                     for name, axis, (low, high, _) in zip(("yaw_deg", "pitch_deg"), self.axes, limits)]
            self._goal(ticks, limits)
            eye = self._still(deadline, time.monotonic() + 0.3)
        # Still is enough to take the image (its pose uses these measured angles); a large miss is only logged.
        if max(abs(eye[k] - target[k]) for k in eye) > self.config.get("target_tolerance_deg", 2):
            logging.warning("head stopped at %s for goal %s after %d trims", eye, target, self.config.get("trim_steps", 3))
        return eye

    def aim(self, target):
        """Send both servos toward ``target`` and return at once, without waiting or trimming: the head turns while
        the body moves, and the next ``settle`` to the same target only finishes the turn."""
        _, ticks, limits = self._checked(target)
        self._goal(ticks, limits)

    def _checked(self, target):
        """(target as floats, goal ticks, each servo's (low, high, torque)); both axes validated before any write."""
        if not isinstance(target, dict) or set(target) != {"yaw_deg", "pitch_deg"}:
            raise ValueError("eye requires yaw_deg and pitch_deg")
        target = {name: float(target[name]) for name in ("yaw_deg", "pitch_deg")}
        for name, axis in zip(("yaw_deg", "pitch_deg"), self.axes):
            lo, hi = axis["range_deg"]
            if not math.isfinite(target[name]) or not lo <= target[name] <= hi:
                raise ValueError("%s outside [%s, %s]" % (name, lo, hi))
        limits = []
        for axis in self.axes:
            mode, low, high, torque = configuration(self.bus, axis["id"])
            if mode != 0 or torque not in (0, 1) or not 0 <= low <= high <= 4095:
                raise RuntimeError("servo mode or hardware limits reject target")
            limits.append((low, high, torque))
        ticks = [self._tick(axis, target[name]) for name, axis in zip(("yaw_deg", "pitch_deg"), self.axes)]
        if any(not low <= tick <= high for tick, (low, high, _) in zip(ticks, limits)):
            raise RuntimeError("servo mode or hardware limits reject target")
        return target, ticks, limits

    def clamp(self, eye):
        """``eye`` with each axis held inside its ``range_deg`` (pitch's lower end is the mount's manual stop)."""
        out = {}
        for name, axis in zip(("yaw_deg", "pitch_deg"), self.axes):
            value = float(eye[name])
            if not math.isfinite(value):
                raise ValueError("%s is not finite" % name)
            lo, hi = axis["range_deg"]
            out[name] = min(hi, max(lo, value))
        return out

    @staticmethod
    def _tick(axis, value):
        return round(axis["zero_tick"] + axis["direction"] * value * 4096 / 360)

    def _goal(self, ticks, limits):
        for axis, tick, (_, _, torque) in zip(self.axes, ticks, limits):
            self.bus.write(axis["id"], 41, move_data(tick, self.config["speed"], self.config["acceleration"]))
            if torque == 0:
                self.bus.write(axis["id"], 40, b"\x01")

    def _still(self, deadline, earliest=0.0, target=None):
        """The eye once both axes are still for three reads; before ``earliest`` only within the target's tolerance
        (a servo that has not started yet reads still)."""
        previous, stable = None, 0
        while time.monotonic() < deadline:
            state = self.read()
            eye = self.eye(state)
            quiet = not state["yaw_deg_moving"] and not state["pitch_deg_moving"]
            quiet &= previous is not None and max(abs(eye[k] - previous[k]) for k in eye) <= 0.2
            if time.monotonic() < earliest:
                quiet &= target is not None and max(abs(eye[k] - target[k]) for k in eye) <= self.config.get(
                    "target_tolerance_deg", 2)
            stable = stable + 1 if quiet else 0
            if stable >= 3:
                return eye
            previous = eye
            time.sleep(0.05)
        raise TimeoutError("head did not settle; no image returned; last goal remains active")

    def close(self):
        # Hold the last pose. Closing a camera service must not drop the camera head.
        self.bus.close()
