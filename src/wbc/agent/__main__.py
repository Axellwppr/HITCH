"""``python -m wbc.agent``: run a task with a model, or rewrite an episode's transcript.

From the repository's root (the episodes are in data/episodes, or WBC_DATA's episodes/):

scripts/agent run tabletop_pick --backend fake
scripts/agent run tabletop_pick --backend hitch --seed 3
scripts/agent run TASK.md --backend hitch --session http://127.0.0.1:PORT
scripts/agent transcript data/episodes/ep-...
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import sys
from pathlib import Path

from wbc.agent import loop, prompts, transcript
from wbc.agent.backends import BackendError, Profile, open_backend
from wbc.agent.fake import FakeModel, FakeServer
from wbc.core import logs
from wbc.core import target as targets


@contextlib.contextmanager
def backend(name: str, model: str | None):
    """The backend for a profile; ``base_url: fake`` starts the scripted model on a local port."""
    profile = Profile.load(name, model=model)
    if profile.base_url != "fake":
        yield open_backend(profile)
        return
    server = FakeServer(FakeModel())
    try:
        yield open_backend(dataclasses.replace(profile, base_url=server.url))
    finally:
        server.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run a task (a name in src/wbc/agent/tasks or a .md path)")
    r.add_argument("task")
    r.add_argument("--backend", default="fake", help="profile in src/wbc/agent/backends.yaml")
    r.add_argument("--model", help="model name for the backend")
    r.add_argument("--target", help="sim://SCENE[?...] or http://HOST:PORT (default: the task's)")
    r.add_argument("--seed", type=int, help="scene seed, added to a sim:// target")
    r.add_argument("--viewer", action="store_true", help="open a window on the simulation (needs a display)")
    r.add_argument("--vision", help="camera service URL of a real robot")
    r.add_argument("--session", help="use this open session instead of starting one")
    r.add_argument("--max-turns", type=int)
    r.add_argument("--keep", action="store_true", help="leave the session it started open")
    t = sub.add_parser("transcript", help="rewrite transcript.md of an episode")
    t.add_argument("episode", type=Path)
    sub.add_parser("tasks", help="list the tasks")
    args = p.parse_args(argv)
    logs.setup("agent")

    if args.command == "tasks":
        for path in sorted(prompts.TASKS.glob("*.md")):
            task = prompts.Task.load(str(path))
            title = "the scene's own task sentence" if task.text == prompts.SCENE_TASK else task.text.splitlines()[0]
            print(f"{task.name:20s} {task.target:36s} {title.lstrip('# ')}")
        return 0
    if args.command == "transcript":
        print(transcript.write(args.episode))
        return 0
    task = prompts.Task.load(args.task)
    target = args.target or task.target
    if args.session is None and not targets.is_sim(target) and not args.vision:
        p.error(f"{target} is a robot: give its camera server with --vision URL")
    if (args.seed is not None or args.viewer) and not targets.is_sim(target):
        p.error(f"--seed and --viewer apply to a sim:// target, not {target}")
    if args.seed is not None:
        target = targets.with_option(target, "seed", args.seed)
    if args.viewer:
        target = targets.with_option(target, "viewer", 1)
    if task.hand:  # the simulator, the session and the prompt of this run read it
        os.environ.setdefault("WBC_HAND", task.hand)
    try:
        with backend(args.backend, args.model) as chosen:
            episode = loop.run(
                task,
                chosen,
                target=target,
                vision=args.vision,
                session=args.session,
                max_turns=args.max_turns,
                keep=args.keep,
            )
    except BackendError as error:
        print(f"backend: {error}", file=sys.stderr)
        return 2
    report = {"episode": str(episode.dir), **episode.record.get("stop", {})}
    if "check" in episode.record:  # the scene's own verdict; "success" above is the model's claim
        report["check"] = episode.record["check"]
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
