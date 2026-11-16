"""The controller's UDP protocol as the simulated robot speaks it (``sim_bridge``): lockstep, one 20 ms control step
per received command, each command answering one published state (its ``state_receive_time_ns``); joint PD targets
and gains in wire order, the targets clipped to the joint range and the gains to MAX_KP / MAX_KD; the robot held in
place until the first tracking target; an emulated operator who holds ``start`` and presses ``A`` after
A_PRESS_FRAME frames; with ``--paused``, no state while the controller's reference idles at its tail, which freezes
time until the next append. The bridge keeps a physics log in chunks of LOG_CHUNK control frames; on stopping it
writes the last chunk and ``shutdown.json``.

This module is light to import: only the stdlib and ``wbc.core.layout`` are imported at module level, and numpy and
``wbc.core.jsonhttp`` where they are used. The controller's own modules (``common.udp_transport``,
``motion_channel``) are imported where they are used too: they are on the path only in a bridge process.
"""

from __future__ import annotations

import signal
import time
import uuid
from pathlib import Path

from wbc.core.layout import CANONICAL_TO_WIRE, WIRE_TO_CANONICAL

CONTROL_DT = 0.02
A_PRESS_FRAME = 110
MAX_KP, MAX_KD = 1000.0, 50.0
LOG_CHUNK = 1500  # control frames per physics log file
STARTUP_TICKS = 30  # controller ticks before an idle tail may pause the clock (the controller's start-up grace)


def new_directory(directory) -> Path:
    """The simulator directory a bridge writes, created; refused when a bridge has written it before."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "bridge.json").exists():
        raise ValueError("use a new simulator directory")
    return directory


class ProtocolBridge:
    """The protocol loop (``run``) around a simulated robot.

    A subclass's constructor calls ``super().__init__(args)`` first, then builds its robot and scene, setting
    ``bounds`` (canonical joint ranges) and ``initial`` (the held start pose: root xyz, quat wxyz, the 29 joints). It
    supplies ``step`` (which takes the command with ``take_command`` and logs the frame's row with ``log``),
    ``measured``, ``write_chunk`` and ``shutdown_record``. The hook ``finish`` does nothing here.
    """

    TO_WIRE = list(CANONICAL_TO_WIRE)
    FROM_WIRE = list(WIRE_TO_CANONICAL)
    paced = False  # hold each control step to CONTROL_DT of wall time (else step as soon as the command arrives)

    def __init__(self, args) -> None:
        """The protocol's state: ``args`` (``directory``, ``channel``, ``paused``, ``state_port``, ``command_port``),
        the new simulator directory, this run's source id, frame 0 and its stamp, not started, an empty physics log.
        The UDP transport is opened by ``run`` (``connect``): a bridge built offline has none."""
        self.args = args
        self.directory = new_directory(args.directory)
        self.source = uuid.uuid4().hex[:8]
        self.frame = 0
        self.stamp = time.perf_counter_ns()
        self.started = False
        self.history = []  # the physics log's rows since its last chunk
        self.chunk = 0  # the next chunk's number
        self.transport = None

    # ------------------------------------------------------------ one control step
    def take_command(self, command):
        """The command's joint targets and gains (canonical order, clipped); marks the robot started once the
        controller enables a target away from the held pose."""
        import numpy as np  # (module docstring)

        target, kp, kd = (np.asarray(command[k], float) for k in ("q_des", "kp", "kd"))
        if any(a.shape != (29,) or not np.isfinite(a).all() for a in (target, kp, kd)):
            raise ValueError("invalid actuator command")
        target = np.clip(target[self.FROM_WIRE], self.bounds[:, 0], self.bounds[:, 1])
        kp = np.clip(kp[self.FROM_WIRE], 0, MAX_KP)
        kd = np.clip(kd[self.FROM_WIRE], 0, MAX_KD)
        if int(command.get("enable", 0)) and np.linalg.norm(target - self.initial[7:]) > 1e-4:
            self.started = True
        return target, kp, kd

    def step(self, command) -> None:
        raise NotImplementedError

    def measured(self):
        """(q, dq) of the 29 joints in canonical order, the root quaternion (wxyz) and the root angular velocity."""
        raise NotImplementedError

    # ------------------------------------------------------------ the physics log
    def log(self, row) -> None:
        """One control frame's row of the physics log; a full chunk (LOG_CHUNK rows) goes to ``write_chunk``."""
        self.history.append(row)
        if len(self.history) >= LOG_CHUNK:
            self.flush()

    def flush(self, final: bool = False) -> None:
        """The rows logged since the last chunk, as chunk number ``chunk``; ``final``: the last one (``end``)."""
        if not self.history:
            return
        rows, self.history = self.history, []
        if self.write_chunk(self.chunk, rows, final):
            self.chunk += 1

    def write_chunk(self, number: int, rows: list, final: bool) -> bool:
        """Write ``rows`` as chunk ``number``; False when they were dropped instead (the number is used by the next
        chunk). The final chunk is always written."""
        raise NotImplementedError

    def shutdown_record(self) -> dict:
        """``shutdown.json``, without ``error`` (``end`` adds it)."""
        raise NotImplementedError

    # ------------------------------------------------------------ hook
    def finish(self) -> None:
        """On stopping, after the final chunk went to ``write_chunk`` and before ``shutdown.json``."""

    # ------------------------------------------------------------ the protocol
    def connect(self) -> None:
        from common.udp_transport import UDPRobotLow  # controller package

        self.transport = UDPRobotLow(
            dict(
                state_host="127.0.0.1",
                state_port=self.args.state_port,
                cmd_bind_host="127.0.0.1",
                cmd_port=self.args.command_port,
            )
        )

    def publish(self) -> None:
        import numpy as np  # (module docstring)
        from common.udp_transport import _state_payload  # controller package

        q, dq, quat, gyro = self.measured()
        payload = _state_payload(
            q=q[self.TO_WIRE],
            dq=dq[self.TO_WIRE],
            quat_wxyz=quat,
            gyro=gyro,
            linacc=np.zeros(3),
            buttons={"start": True, "A": self.frame >= A_PRESS_FRAME},
            sticks={},
            state_receive_time_ns=self.stamp,
        )
        payload.update(sensor_source_id=self.source, sensor_frame_id=self.frame, sample_timestamp_unix_s=time.time())
        self.transport.state_sender.send(payload)

    def idle_at_tail(self, channel) -> bool:
        """Paused mode: has the controller reached the end of its reference?"""
        deadline = time.monotonic() + 0.01
        state = channel.read()
        while (state is None or state["stamp"] != self.stamp) and time.monotonic() < deadline:
            time.sleep(0.0002)
            state = channel.read()
        return (
            state is not None
            and state["stamp"] == self.stamp
            and state["tick"] >= STARTUP_TICKS
            and state["index"] == state["length"] - 1
        )

    def end(self, error: BaseException | None = None) -> dict:
        """Stopping, online (``run``) or offline: the final chunk of the physics log, ``finish``, and
        ``shutdown.json``: the ``shutdown_record`` and ``error``, the error the bridge stopped on (``repr``; None when it
        was told to stop). Returns the record. When the final chunk or ``finish`` fails (the physics log not written),
        ``shutdown.json`` is still written, that error in ``error`` (after the one the bridge stopped on, if any), and
        then the error is raised."""
        from wbc.core.jsonhttp import write_json_atomic  # (module docstring)

        failure = None
        try:
            self.flush(final=True)
            self.finish()
        except BaseException as caught:
            failure = caught
        errors = [repr(e) for e in (error, failure) if e is not None]
        record = {**self.shutdown_record(), "error": "; ".join(errors) or None}
        write_json_atomic(self.directory / "shutdown.json", record)
        if failure is not None:
            raise failure
        return record

    def run(self) -> None:
        from motion_channel import MotionChannel  # controller package

        alive = True

        def stop(*_):
            nonlocal alive
            alive = False

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, stop)
        channel = None
        error = None
        try:
            self.connect()
            seen = -1
            waiting = False
            self.publish()
            sent = time.monotonic()
            deadline = sent + CONTROL_DT
            while alive:
                if channel is None and (self.args.channel / "ready").exists():
                    channel = MotionChannel(self.args.channel)
                if waiting:
                    if not channel.pending():
                        time.sleep(0.001)
                        continue
                    waiting = False
                    self.stamp = time.perf_counter_ns()
                    self.publish()
                    sent = time.monotonic()
                    deadline = sent + CONTROL_DT
                packet = self.transport.read_latest_command(with_meta=True)
                if packet is None or packet.seq == seen:
                    if self.frame == 0 and time.monotonic() - sent > 0.1:
                        self.publish()  # same stamp: the controller ignores the duplicate
                        sent = time.monotonic()
                    time.sleep(0.0005)
                    continue
                seen = packet.seq
                if int(packet.data.get("state_receive_time_ns", -1)) != self.stamp:
                    continue  # a command for an older state
                self.step(packet.data)
                if self.args.paused and channel is not None and self.idle_at_tail(channel):
                    waiting = True
                    continue
                if self.paced:
                    time.sleep(max(0.0, deadline - time.monotonic()))
                self.stamp = time.perf_counter_ns()
                deadline = max(deadline + CONTROL_DT, time.monotonic() + 0.001)
                self.publish()
                sent = time.monotonic()
        except BaseException as caught:
            error = caught
            raise
        finally:
            try:
                self.end(error)
            finally:
                if channel is not None:
                    channel.close()
                if self.transport is not None:
                    self.transport.close()
