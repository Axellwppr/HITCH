"""One error type with an explicit code. Callers branch on the code, never the class."""

from enum import Enum


class ErrorCode(str, Enum):
    # Raised by the robot link.
    STALE_TAIL = "STALE_TAIL"  # the reference tail moved; nothing was appended
    EPOCH_CHANGED = "EPOCH_CHANGED"  # the controller restarted
    NOT_TRACKING = "NOT_TRACKING"  # the controller is not tracking a reference
    INVALID_SEGMENT = "INVALID_SEGMENT"  # malformed frames; nothing was appended
    BUSY = "BUSY"  # the link did not process the request; retry is safe
    PENDING = "PENDING"  # handed to the controller, not taken yet; the segment id shows up in state once it is
    UNREACHABLE = "UNREACHABLE"  # transport failure; the outcome is unknown
    INTERNAL = "INTERNAL"  # unexpected failure; the outcome is unknown
    # Raised by the harness.
    INVALID_REQUEST = "INVALID_REQUEST"  # bad parameters
    PRECONDITION = "PRECONDITION"  # something must happen first: observe, plan, stop moving, a start posture
    PLAN_REJECTED = "PLAN_REJECTED"  # a plan failed its checks
    PLAN_STALE = "PLAN_STALE"  # the plan's anchor no longer matches the robot
    SESSION_STATE = "SESSION_STATE"  # the session cannot act now: closed, not tracking, stale or fault
    NOT_FOUND = "NOT_FOUND"  # no artifact or run with that id, no route or file at that path


# Codes after which a write may or may not have taken effect.
UNCERTAIN = frozenset({ErrorCode.PENDING, ErrorCode.UNREACHABLE, ErrorCode.INTERNAL})


class WbcError(Exception):
    def __init__(self, code: ErrorCode, message: str, **details):
        super().__init__(message)
        self.code = ErrorCode(code)
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        return {"code": self.code.value, "message": self.message, **self.details}

    @classmethod
    def from_dict(cls, value: dict) -> "WbcError":
        details = {k: v for k, v in value.items() if k not in ("code", "message")}
        try:
            code = ErrorCode(value.get("code"))
        except ValueError:
            code = ErrorCode.INTERNAL
        return cls(code, value.get("message", ""), **details)

    def __str__(self):
        return f"{self.code.value}: {self.message}"


def invalid(message: str, **details) -> WbcError:
    return WbcError(ErrorCode.INVALID_REQUEST, message, **details)


class Cause(str, Enum):
    """Why the session refused an operation, where the agent's tools act on it: an error's ``cause`` detail.

    The model never reads it (``wbc.agent.render`` leaves it out); the tools observe again after a stale observation,
    and the loop of a real robot stops when the controller is down or the robot cannot be read. ``controller_down`` also reads the message.
    """

    MOVING = "moving"  # the robot still has motion queued
    OBSERVED_MOVING = "observed_moving"  # the latest observation was taken while the robot moved
    REFERENCE_CHANGED = "reference_changed"  # the latest observation predates the current reference
    CONTROLLER_DOWN = "controller_down"  # the controller reports a fault or does not track
    ROBOT_UNREACHABLE = "robot_unreachable"  # the robot's state cannot be read, or has stopped updating


STALE = frozenset({Cause.MOVING, Cause.OBSERVED_MOVING, Cause.REFERENCE_CHANGED})  # observe again, then plan
# The messages that say the controller is down (the session, its runs; the link servers).
_DOWN_TEXT = (
    "controller fault",
    "controller restarted",
    "controller is fault",
    "controller is not_tracking",
    "controller reports a fault",
)
_DOWN_CODES = frozenset({ErrorCode.NOT_TRACKING.value, ErrorCode.EPOCH_CHANGED.value})


def cause_of(error: dict) -> Cause | None:
    """An error's (``WbcError.to_dict``) cause: None for none, or for a ``cause`` this version does not know."""
    try:
        return Cause(error["cause"]) if error.get("cause") else None
    except ValueError:
        return None


def controller_down(error: dict) -> bool:
    """Whether an error says the controller faulted, restarted or stopped tracking. Besides the cause and the link's
    codes, the message: a session that could not read the robot relays the link's error as text."""
    return (
        error.get("cause") == Cause.CONTROLLER_DOWN
        or error.get("code") in _DOWN_CODES
        or any(text in error.get("message", "") for text in _DOWN_TEXT)
    )
