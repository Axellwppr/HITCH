"""ST3215-C044 single-turn positioning. Linux and the Python 3 standard library only.

Follows the servo's communication protocol, its instruction table and the sms_sts register definitions of
FTServo_Python. The data of a move equals WritePosEx(id, position, speed, acc).
"""
import errno
import fcntl
import os
import select
import struct
import termios
import time


def serial_setting(operation, *args):
    # termios.error is not an OSError here; convert it so the connection layer handles one type.
    try:
        return operation(*args)
    except termios.error as exc:
        code = exc.args[0] if exc.args and isinstance(exc.args[0], int) else errno.EIO
        message = exc.args[1] if len(exc.args) > 1 else str(exc)
        raise OSError(code, message) from exc


class ServoError(Exception):
    pass


class ReplyTimeout(ServoError):
    pass


class ServoFault(ServoError):
    pass


def packet(servo_id, instruction, parameters=b""):
    if not 0 <= servo_id <= 253:
        raise ServoError("The ID must be 0-253; the broadcast ID is not allowed.")
    body = bytes([servo_id, len(parameters) + 2, instruction]) + bytes(parameters)
    return b"\xff\xff" + body + bytes([(~sum(body)) & 255])


def signed_magnitude(value):
    return -(value & 0x7fff) if value & 0x8000 else value


def move_data(position, speed, acc):
    return bytes([acc]) + struct.pack("<HHH", position, 0, speed)


def fault_text(error):
    known = {1: "voltage", 2: "angle sensor", 4: "overheat", 8: "overcurrent", 32: "overload"}
    names = [label for bit, label in known.items() if error & bit]
    return f"servo status error 0x{error:02X} ({', '.join(names) or 'see the memory table of the servo model'})"


class Bus:
    def __init__(self, path, baud, timeout=0.2):
        self.fd = None
        self.path = path
        self.timeout = timeout
        baud_constant = getattr(termios, f"B{baud}", None)
        if baud_constant is None:
            raise ServoError(f"The serial API of this machine does not support {baud} baud; 1000000 (default), 500000 and 115200 are supported.")
        try:
            self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            opened = os.fstat(self.fd)
            self.device_identity = (opened.st_dev, opened.st_ino, opened.st_rdev)
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            settings = serial_setting(termios.tcgetattr, self.fd)
            settings[:6] = [0, 0, termios.CS8 | termios.CREAD | termios.CLOCAL,
                            0, baud_constant, baud_constant]
            settings[6][termios.VMIN] = 0
            settings[6][termios.VTIME] = 0
            serial_setting(termios.tcsetattr, self.fd, termios.TCSANOW, settings)
            serial_setting(termios.tcflush, self.fd, termios.TCIOFLUSH)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def request(self, servo_id, instruction, parameters=b"", expected=0, allow_fault=False):
        tx = packet(servo_id, instruction, parameters)
        # by-id may point to a new node after a replug; the old descriptor has to be closed and reopened.
        current = os.stat(self.path)
        if (current.st_dev, current.st_ino, current.st_rdev) != self.device_identity:
            raise OSError(errno.ENODEV, "The USB serial port was re-enumerated; reconnect.", self.path)
        serial_setting(termios.tcflush, self.fd, termios.TCIFLUSH)
        deadline = time.monotonic() + self.timeout
        pending = memoryview(tx)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [self.fd], [], remaining)[1]:
                raise ServoError("Serial write timed out.")
            try:
                sent = os.write(self.fd, pending)
                if sent == 0:
                    raise ServoError("The serial port is disconnected.")
                pending = pending[sent:]
            except BlockingIOError:
                continue
        received = bytearray()
        while True:
            while len(received) >= 4:
                start = received.find(b"\xff\xff")
                if start < 0:
                    received[:] = received[-1:] if received[-1] == 255 else b""
                    break
                if start:
                    del received[:start]
                if len(received) < 4:
                    break
                length = received[3]
                if received[2] > 253 or not 2 <= length <= 250:
                    del received[0]
                    continue
                total = length + 4
                if len(received) < total:
                    break
                frame = bytes(received[:total])
                if sum(frame[2:]) & 255 != 255:
                    del received[0]
                    continue
                del received[:total]
                # Some adapter boards echo the sent frame; an echo is not a servo reply.
                if frame == tx or frame[2] != servo_id or frame[4] > 127:
                    continue
                if frame[4] and not allow_fault:
                    raise ServoFault(fault_text(frame[4]))
                if length - 2 == expected:
                    return frame[5:-1]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ReplyTimeout(f"No valid reply from ID {servo_id}. Check the power, the TTL connection, the ID and the baud rate.")
            if select.select([self.fd], [], [], remaining)[0]:
                try:
                    data = os.read(self.fd, 512)
                except BlockingIOError:
                    continue
                if not data:
                    raise ServoError("The serial port is disconnected.")
                received.extend(data)

    def read(self, servo_id, address, size, allow_fault=False):
        return self.request(servo_id, 2, bytes([address, size]), size, allow_fault)

    def write(self, servo_id, address, data, allow_fault=False):
        # Some configurations reply only to reads; a write without a reply is confirmed by reading back, never resent.
        try:
            self.request(servo_id, 3, bytes([address]) + data, allow_fault=allow_fault)
        except ReplyTimeout:
            pass
        actual = self.read(servo_id, address, len(data), allow_fault)
        if actual != data:
            raise ServoError(f"Register {address} reads back a different value; no further commands are sent.")


def configuration(bus, servo_id):
    mode = bus.read(servo_id, 33, 1)[0]
    low, high = struct.unpack("<HH", bus.read(servo_id, 9, 4))
    torque = bus.read(servo_id, 40, 1)[0]
    return mode, low, high, torque


def feedback(bus, servo_id):
    data = bus.read(servo_id, 56, 11)
    position, speed = struct.unpack("<HH", data[:4])
    return (signed_magnitude(position), signed_magnitude(speed),
            data[6] / 10, data[7], data[10])
