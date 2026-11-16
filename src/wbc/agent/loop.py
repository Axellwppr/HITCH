"""One episode: a robot session, a model, and the conversation between them, recorded.

The conversation is append-only: system prompt, one task message showing the
robot as it is (an observation taken before the first turn), then the model's
replies and the results of the tools it called (text, with the session
results behind it recorded as ``data``). Everything goes to the episode's
directory as it happens (``wbc.agent.episode``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from wbc.agent import prompts, transcript
from wbc.agent.backends import Backend, BackendError
from wbc.agent.conversation import Message
from wbc.agent.episode import Episode
from wbc.agent.tools import Tools
from wbc.core import target as targets
from wbc.core.errors import Cause, ErrorCode, controller_down
from wbc.paths import data_root
from wbc.session import client

NUDGE = "No tool was called. Continue with tool calls, or call finish if the task is done or cannot be done."
AFTER_MOTION = (
    "not run: an earlier execute in this reply moved the robot, so these arguments refer to an outdated frame O; "
    "read the new observation, then call again"
)
FINISH_ALONE = "not run: finish must be the only call of its reply, after you have read the results of the others"


def run(
    task: prompts.Task,
    backend: Backend,
    *,
    target: str | None = None,
    vision: str | None = None,
    session: str | None = None,
    max_turns: int | None = None,
    keep: bool = False,
    root: Path | None = None,
) -> Episode:
    """Run ``task`` on a new session for ``target`` (default: the task's), or on an open ``session``.

    ``vision`` is the camera service of a real robot (``http://``); a simulator brings its own.

    A task whose text is ``{{instruction}}`` gets the task sentence of the simulated scene this call starts. When
    that scene has a success check, it runs once the simulator has stopped; its result is the record's ``check``.
    """
    root = root or data_root()
    target = target or task.target
    max_turns = max_turns or task.max_turns
    record = {
        "task": task.name,
        "target": target,
        "backend": backend.profile.public(),
        "max_turns": max_turns,
        "started": time.time(),
    }
    episode = Episode.create(root, record)
    started = sim = None
    try:
        if session is None:
            reply = client.connect(target, vision=vision, root=root)
            started = session = reply["session"]
            if targets.is_sim(target):
                sim = root / "sessions" / reply["session_id"] / "runtime" / "sim"
        record["session"] = session
        if prompts.SCENE_TASK in task.text:
            task = replace(task, text=task.text.replace(prompts.SCENE_TASK, _scene_task(task, sim)))
        status = record["status_at_start"] = client.call(session, "status")
        episode.save()
        tools = Tools(session, episode.dir / "images")
        episode.add(Message("system", prompts.system_prompt()))
        now = tools.observe()
        text = prompts.task_prompt(task, status["robot"], target, max_turns, now.text)
        episode.add(Message("user", text, now.images, data=now.data))
        down = _robot_down(now.data) if _stop_on_fault() else None
        if down:  # the first observation already failed: the model is not asked
            record["stop"] = {"reason": "robot_fault", "detail": down, "turns": 0}
        else:
            record["stop"] = _converse(episode, backend, tools, max_turns)
    except KeyboardInterrupt:
        record["stop"] = {"reason": "interrupted"}
    except Exception as error:  # the environment failed; record it, then let it surface
        record["stop"] = {"reason": "error", "detail": repr(error)}
        raise
    finally:
        if started and not keep:
            try:
                client.call(started, "close")
            except Exception as error:
                record["close_error"] = repr(error)
            else:
                if sim is not None and "task_spec" in _scene(sim):
                    record["check"] = _check(sim)
        record["ended"] = time.time()
        episode.save()
        transcript.write(episode.dir)
    return episode


def _scene(sim: Path | None) -> dict:
    """The spec of the scene in a simulator directory (the simulator writes it before the session answers)."""
    path = sim / "scene_spec.json" if sim else None
    return json.loads(path.read_text()) if path and path.is_file() else {}


def _scene_task(task: prompts.Task, sim: Path | None) -> str:
    instruction = _scene(sim).get("task", {}).get("instruction")
    if instruction is None:
        raise ValueError(
            f"task {task.name!r} takes its text from the scene: run it on a simulated scene with a task "
            "(--target 'sim://tabletop_pick?seed=N'), not on an open session or a robot"
        )
    return instruction


def _check(sim: Path) -> dict:
    """The scene's success check (``wbc_link_server.scenes.success``, its own process) on a stopped simulator."""
    command = [sys.executable, "-m", "wbc_link_server.scenes.success", str(sim)]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"error": "the success check did not finish within 300 s"}
    if done.returncode:
        return {"error": done.stderr.strip()[-500:]}
    return json.loads(done.stdout)


def _stop_on_fault() -> bool:
    """A controller that faulted, restarted or stopped tracking, a robot side that cannot be reached, or a camera that
    gave no image ends the episode as an infrastructure stop; with WBC_STOP_ON_FAULT=0 the model is handed the errors
    instead."""
    return os.environ.get("WBC_STOP_ON_FAULT", "1") != "0"


def _robot_down(data: dict | None) -> str | None:
    """What says the robot side is gone in one tool's session calls: a lost run, a fault or not-tracking error of the
    controller, a robot or session that cannot be reached or whose state stopped updating, or an observation whose
    camera image failed."""
    for call in (data or {}).get("calls", ()):
        result, error = call.get("result"), call.get("error")
        if isinstance(result, dict) and result.get("state") == "lost":
            return f"{call['op']}: run lost ({result.get('detail')})"
        if isinstance(result, dict) and result.get("vision_error"):
            return f"{call['op']}: camera failed ({str(result['vision_error'])[:200]})"
        if isinstance(error, dict) and (
            controller_down(error)
            or error.get("code") == ErrorCode.UNREACHABLE.value
            or error.get("cause") == Cause.ROBOT_UNREACHABLE.value
        ):
            return f"{call['op']}: {error.get('code')}: {error.get('message', '')[:200]}"
    return None


def _converse(episode: Episode, backend: Backend, tools: Tools, max_turns: int) -> dict:
    stop_on_fault = _stop_on_fault()  # read once per episode
    idle = 0  # replies in a row without a tool call
    for turn in range(1, max_turns + 1):
        try:
            reply = backend.complete(episode.messages, tools.specs, cache_key=episode.dir.name)
        except BackendError as error:
            return {"reason": "backend_error", "detail": str(error), "turns": turn - 1}
        episode.request(turn, reply)
        episode.add(reply.message)
        if reply.note:
            return {"reason": "incomplete_response", "detail": reply.note, "turns": turn}
        if not reply.message.tool_calls:
            idle += 1
            if idle == 2:
                return {"reason": "no_tool_calls", "turns": turn}
            episode.add(Message("user", NUDGE))
            continue
        idle, finished, moved = 0, None, False
        calls = reply.message.tool_calls
        for call in calls:
            if moved:  # its arguments were chosen before the model saw the robot move
                result = tools.skip(call, AFTER_MOTION)
            elif call.name == "finish" and len(calls) > 1:
                result = tools.skip(call, FINISH_ALONE)
            else:
                result = tools.run(call)
            moved = moved or result.moved
            episode.add(Message("tool", result.text, result.images, tool_call_id=call.id, data=result.data))
            finished = finished or result.finished
            down = _robot_down(result.data) if stop_on_fault else None
            if down:
                episode.save()
                return {"reason": "robot_fault", "detail": down, "turns": turn}
        episode.save()
        if finished:
            return {"reason": "finished", "turns": turn, **finished}
    return {"reason": "max_turns", "turns": max_turns}
