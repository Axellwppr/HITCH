"""``wbc`` command line: an HTTP client of one session host, JSON out.

``wbc connect TARGET`` starts a session and prints its URL; every other
command goes to the session at ``$WBC_SESSION`` (or ``--session URL``) as the
Session method of the same name, its arguments named like the method's.
Ids default to the latest of their kind.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from wbc.core.errors import ErrorCode, WbcError
from wbc.session import client


def json_value(text: str):
    """A JSON literal, or @path to a JSON file."""
    try:
        return json.loads(Path(text[1:]).read_text() if text.startswith("@") else text)
    except (OSError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


# command (one per wbc.session.session.OPS, in its order) -> (help, [(flag or name, argparse options)]); each dest
# is the Session method's parameter name
COMMANDS = {
    "status": ("session state, queued motion and the latest ids", []),
    "observe": (
        "robot state and RGB (allowed while moving)",
        [("--since", {"metavar": "OBS", "help": "also say where the new frame O lies in this observation's"})],
    ),
    "perceive": (
        "3D points in the latest observation's frame O for image points normalized to 0-1000",
        [
            ("points", {"type": json_value, "help": "JSON list of [x, y], e.g. [[500, 800]]"}),
            ("--show", {"action": "store_true", "default": None, "help": "also the image with the points marked"}),
        ],
    ),
    "plan": (
        "plan a request (docs/REFERENCE.md) from the reference tail, in the latest observation's frame",
        [("request", {"type": json_value, "help": 'JSON object, e.g. {"height": 0.65, "walk": {"target": [0.4, 0]}}'})],
    ),
    "execute": ("append a ready plan to the robot", [("plan", {"nargs": "?"})]),
    "grip": (
        "close or open the fingers now, then stand still so they move",
        [
            ("fingers", {"type": json_value, "help": 'JSON object, e.g. {"right": "closed"}'}),
            ("--hold-s", {"dest": "hold_s", "type": float, "help": "stand-still after the command, s (default 1)"}),
        ],
    ),
    "look": (
        "turn the head camera for the next observations (relative to the torso)",
        [("eye", {"type": json_value, "help": 'JSON object, e.g. {"yaw_deg": 0, "pitch_deg": 30}'})],
    ),
    "aim": (
        "start turning the head camera toward a setting while the body moves (look still sets it)",
        [("eye", {"type": json_value, "help": 'JSON object, e.g. {"yaw_deg": 0, "pitch_deg": 30}'})],
    ),
    "wait": (
        "a run once it finishes (--timeout 0: as it is now)",
        [("run", {"nargs": "?"}), ("--timeout", {"type": float})],
    ),
    "close": ("end the session (and a simulator it started)", []),
    "show": ("an artifact or run as stored (debugging)", [("identifier", {"metavar": "ID"})]),
    "log": ("recent journal entries", [("--count", {"type": int})]),
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wbc", description="G1 harness: observe -> plan -> execute -> wait")
    p.add_argument("--session", help="session URL (default: $WBC_SESSION)")
    p.add_argument("--debug", action="store_true", help="print tracebacks")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("connect", help="start a session on http://HOST:PORT or sim://SCENE[?paused=1&vision=0]")
    c.add_argument("target")
    c.add_argument("--vision", help="vision service URL when attaching to a robot")
    c.add_argument("--port", type=int, default=0, help="port for the session (default: any free port)")
    for name, (help_text, arguments) in COMMANDS.items():
        c = sub.add_parser(name, help=help_text)
        for flag, options in arguments:
            c.add_argument(flag, **options)
    return p


def run(args) -> object:
    if args.command == "connect":
        return client.connect(args.target, vision=args.vision, port=args.port)
    values = {k: v for k, v in vars(args).items() if k not in ("command", "session", "debug") and v is not None}
    return client.call(client.session_url(args.session), args.command, **values)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except WbcError as error:
        if args.debug:
            traceback.print_exc()
        print(json.dumps({"error": error.to_dict()}, ensure_ascii=False), file=sys.stderr)
        return 2
    except Exception as error:  # an unexpected failure is a bug; always show where
        traceback.print_exc()
        print(json.dumps({"error": {"code": ErrorCode.INTERNAL.value, "message": str(error)}}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
