"""Prepare a runtime directory and exec the controller's own entry point (deploy.py).

Writes controller.yaml / tracking.yaml from one checkpoint bundle (deploy.yaml),
creates the motion channel, records
runtime.json (runtime id, backend, pid, the checkpoint's metadata), then replaces this process with deploy.py.
The operator drives start / A / stop on the robot.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import sys
import uuid
from pathlib import Path

import numpy as np
import yaml

from wbc.core import logs
from wbc.core.jsonhttp import write_json_atomic
from wbc.core.layout import CANONICAL_TO_WIRE, WIRE_JOINTS
from wbc.paths import CHECKPOINT, CONTROLLER, INITIAL_REFERENCE

log = logging.getLogger("wbc_link_server.launch.policy")


def load_bundle(bundle: Path) -> dict:
    spec = yaml.safe_load((bundle / "deploy.yaml").read_text())
    if spec["control_hz"] != 50 or tuple(spec["action"]["joint_names"]) != WIRE_JOINTS:
        raise ValueError("checkpoint must run at 50 Hz in the G1 wire joint order")
    return spec


def write_configs(directory: Path, bundle: Path, initial, ports: dict) -> dict:
    """Controller and tracking YAML for deploy.py; returns runtime metadata."""
    spec = load_bundle(bundle)
    action = spec["action"]
    wire_initial = np.asarray(initial, float)[7:][list(CANONICAL_TO_WIRE)].tolist()
    controller = yaml.safe_load((CONTROLLER / "config/g1/controller.yaml").read_text())
    controller.update(
        policy_joint_names=list(WIRE_JOINTS),
        control_freq=50,
        udp=dict(
            state_bind_host=ports["state_host"],
            state_port=ports["state"],
            cmd_host=ports["command_host"],
            cmd_port=ports["command"],
        ),
        kps=action["kps"],
        kds=action["kds"],
        default_qpos=action["default_joint_pos"],
        init_qpos=wire_initial,
    )
    observation = dict(spec["observation"])
    target = observation.pop("target_joint")
    compliance = observation.pop("compliance")
    tracking = {
        **observation,
        "target_joint_include_diff": target["layout"] == "full",
        "use_compliance_flag_obs": compliance["enabled"],
        "compliance_flag_value": compliance["value"],
        "compliance_flag_threshold": compliance["threshold"],
        "_config_dir": str(bundle),
        "policy_path": str(bundle / spec["policy"]),
        "action_scale": action["scale"],
        "action_clip": action["clip"],
        "action_joint_names": list(WIRE_JOINTS),
        "dataset_joint_names": list(WIRE_JOINTS),
        "transition_steps": 0,
        "switch_tail_keep_steps": 8,
        "motion_source": {"type": "external", "external": {"directory": str(directory / "channel")}},
        "motions": [],
        "motion_clips": [
            {
                "name": "default",
                "joint_pos": wire_initial,
                "root_pos": list(map(float, initial[:3])),
                "root_quat": list(map(float, initial[3:7])),
            }
        ],
    }
    for name, config in (("controller", controller), ("tracking", tracking)):
        (directory / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return {"horizon": int(max(tracking["future_steps"])), "checkpoint": bundle.name}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["sim", "real"], required=True)
    p.add_argument("--directory", type=Path, required=True, help="new runtime directory")
    p.add_argument("--initial", type=Path, default=INITIAL_REFERENCE)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT, help="bundle with deploy.yaml + ONNX")
    p.add_argument("--state-port", type=int, default=55001)
    p.add_argument("--command-port", type=int, default=55002)
    p.add_argument("--state-host", default="127.0.0.1")
    p.add_argument("--command-host", default="127.0.0.1")
    p.add_argument("--paused", action="store_true", help="the simulator pauses while the reference is idle")
    p.add_argument("--record", action=argparse.BooleanOptionalAction, default=True, help="per-step log")
    args = p.parse_args()
    if args.backend == "real" and args.paused:
        p.error("a real robot cannot pause")
    directory = args.directory.resolve()
    logs.setup("policy", logs.session_of(directory))
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "channel").exists():
        p.error(f"{directory} was used before; give a new runtime directory")
    sys.path.insert(0, str(CONTROLLER / "src"))
    from motion_channel import MotionChannel  # the controller's sources, just put on the path

    initial = np.load(args.initial, allow_pickle=False)["qpos"][0]
    ports = {
        "state": args.state_port,
        "command": args.command_port,
        "state_host": args.state_host,
        "command_host": args.command_host,
    }
    metadata = write_configs(directory, args.checkpoint.resolve(), initial, ports)
    MotionChannel(directory / "channel", create=True)
    runtime = {
        "runtime_id": uuid.uuid4().hex[:12],
        "backend": args.backend,
        "paused": args.paused,
        "policy_pid": os.getpid(),
        "metadata": metadata,
    }
    write_json_atomic(directory / "runtime.json", runtime)  # atomic: the link server polls for it
    command = [
        sys.executable,
        str(CONTROLLER / "src/deploy.py"),
        "--robot",
        "g1",
        "--controller-config",
        str(directory / "controller.yaml"),
        "--tracking-config",
        str(directory / "tracking.yaml"),
    ]
    command += ["--record-dir", str(directory / "steps")] if args.record else ["--no-record"]
    log.info("runtime %s: exec %s", runtime["runtime_id"], shlex.join(command))
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
