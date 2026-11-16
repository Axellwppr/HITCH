"""Prompts assembled from the markdown templates beside this module.

``prompts/system.md`` is fixed for a harness version and a hand: it describes the
``execute`` request of docs/REFERENCE.md (its torso limits are rendered from
``wbc.core.torso``, so they never drift from the code); ``prompts/task.md``
wraps one task file from ``tasks/`` with the facts of this episode. Placeholders
are ``{{name}}``; a missing or unused value is an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from wbc.core import target as targets
from wbc.core.hands import hand_grips, hand_prompt
from wbc.core.torso import TORSO_LIMITS, TURN_LEAN

HERE = Path(__file__).parent
TASKS = HERE / "tasks"
PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
SCENE_TASK = "{{instruction}}"  # in a task file: the task sentence of the simulated scene (its seed chooses it)


def fill(template: str, **values) -> str:
    """Replace every ``{{name}}``; refuse placeholders without a value and values without a placeholder."""
    used = set(PLACEHOLDER.findall(template))
    if used != set(values):
        raise ValueError(f"template needs {sorted(used)}, got {sorted(values)}")
    return PLACEHOLDER.sub(lambda m: str(values[m.group(1)]).strip(), template)


def template(name: str) -> str:
    return (HERE / "prompts" / f"{name}.md").read_text()


def tool_descriptions() -> dict[str, str]:
    """``## name`` sections of prompts/tools.md."""
    sections = re.split(r"^## (\w+)\s*$", template("tools"), flags=re.M)
    return {name: body.strip() for name, body in zip(sections[1::2], sections[2::2], strict=True)}


@dataclass(frozen=True)
class Task:
    name: str
    target: str  # sim://SCENE or http://HOST:PORT (a real robot's link server)
    max_turns: int
    text: str
    hand: str | None = None  # the hands the task needs (WBC_HAND), when it names them

    @classmethod
    def load(cls, name_or_path: str) -> Task:
        path = Path(name_or_path)
        if not path.suffix:
            path = TASKS / f"{name_or_path}.md"
        raw = path.read_text()
        match = re.match(r"---\n(.*?)\n---\n(.*)", raw, re.S)
        if match is None:
            raise ValueError(f"{path}: expected '---' front matter with target and max_turns")
        meta = yaml.safe_load(match.group(1))
        text = match.group(2).strip()
        return cls(path.stem, str(meta["target"]), int(meta.get("max_turns", 40)), text, meta.get("hand"))


GRIP_FIELD = '- and `grip: "open" | "closed"`: the fingers after the in-place part (with or without one of the above).'
GRIP_STEP = (
    "- Then the fingers (`grip`), with 1 s standing still so they move (plus `wait_s`): they come after the in-place "
    "part has been played, and take the request's `wait_s`; if the motion does not finish, the fingers are not moved "
    "and the result says so. A request with `grip` has no `walk`."
)


def system_prompt(hand: str | None = None) -> str:
    """The fixed instructions for the robot's hands (``hand``, else this process's ``WBC_HAND``); finger commands
    only for hands that have them."""
    grips = hand_grips(hand)
    return fill(
        template("system"),
        hands=hand_prompt(hand),
        torso_limits=torso_table(),
        grip_field=GRIP_FIELD if grips else "",
        grip_step=GRIP_STEP if grips else "",
        grip_state="; whether the fingers are open or closed" if grips else "",
    )


def torso_table() -> str:
    """The torso limits by pelvis height (wbc.core.torso.TORSO_LIMITS) as a markdown table."""
    rows = [
        "| pelvis height from | largest lean | largest turn, lean up to "
        f"{TURN_LEAN}° | largest turn, lean beyond {TURN_LEAN}° |",
        "|---|---|---|---|",
    ]
    rows += [f"| {_metres(low)} m | {lean}° | {turn}° | {turned}° |" for low, lean, turn, turned in TORSO_LIMITS]
    rows.append(f"| below {_metres(TORSO_LIMITS[-1][0])} m | 0° | 0° | 0° |")
    return "\n".join(rows)


def _metres(value: float) -> str:
    """Centimetre precision, or millimetres where a limit needs them: 0.50, 0.785."""
    return f"{value:.2f}" if abs(round(value, 2) - value) < 1e-9 else f"{value:.3f}"


def task_prompt(task: Task, robot: str, target: str, max_turns: int, observation: str) -> str:
    return fill(
        template("task"),
        task=task.text,
        environment=environment(robot, target),
        max_turns=max_turns,
        observation=observation,
    )


def environment(robot: str, target: str) -> str:
    """What the robot is, from the link's own report (``sim``, ``real`` or ``fake``) and the target URL."""
    if robot == "real":
        return "the real G1. An operator watches with the emergency stop; move conservatively."
    if robot == "fake":
        return "a kinematic stand-in for testing: motions play exactly and nothing in the scene ever moves."
    scene = f", scene `{targets.scene(target)}`" if targets.is_sim(target) else ""
    clock = " Simulated time stands still while the robot is idle." if targets.paused(target) else ""
    return f"a simulated G1 (MuJoCo physics{scene}).{clock}"
