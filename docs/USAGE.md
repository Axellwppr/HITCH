# Usage reference

Commands, options, files and environment variables. The quick path is in the [README](../README.md), the real robot
in [REAL_ROBOT.md](REAL_ROBOT.md), the request a model sends in [REFERENCE.md](REFERENCE.md).

- [Commands](#commands)
- [Session: `scripts/wbc`](#session-scriptswbc)
- [Simulation targets](#simulation-targets)
- [Agent loop: `scripts/agent`](#agent-loop-scriptsagent)
- [Model backends](#model-backends)
- [Environment variables](#environment-variables)
- [Repository layout](#repository-layout)

## Commands

Run everything through these scripts; they use the environment `scripts/setup` made. `wbc`, `agent` and `robot`
print their usage with `--help`.

| Command | Purpose |
| --- | --- |
| `scripts/setup` | fetch the controller and create the environment |
| `scripts/run ...` | run Python inside the environment (`scripts/run -m MODULE`, `scripts/run FILE`) |
| `scripts/wbc COMMAND` | the session command line |
| `scripts/agent run TASK` / `tasks` / `transcript EPISODE` | the agent loop; list the tasks; rewrite an episode's `transcript.md` |
| `scripts/robot COMMAND` | the programs on the robot computer ([REAL_ROBOT.md](REAL_ROBOT.md)) |

## Session: `scripts/wbc`

One session is one controller and one HTTP address. `connect` starts the session and prints its address; the other
commands go to `$WBC_SESSION` (or `scripts/wbc --session URL COMMAND`). All output is JSON; a JSON argument may be
written `@file.json`.

| Command | Effect |
| --- | --- |
| `connect TARGET [--vision URL] [--port P]` | start a session on `sim://SCENE[?options]` or on a robot's link server `http://HOST:PORT` |
| `status` | session state, queued motion and the latest observation, plan and run |
| `observe [--since OBS]` | robot state and RGB image; allowed while the robot moves |
| `perceive POINTS [--show]` | 3D points for image points normalised to 0-1000, in the latest observation's frame |
| `plan REQUEST` | plan a request ([REFERENCE.md](REFERENCE.md)) while the robot is idle: `ready` or `rejected` |
| `execute [PLAN]` | append a ready plan to the robot |
| `grip FINGERS [--hold-s S]` | open or close the fingers (`{"right": "closed"}`), then stand still; needs `WBC_HAND=dex3` when `connect` runs |
| `look EYE` / `aim EYE` | set the head camera for the next observations / turn it now (`{"yaw_deg": 0, "pitch_deg": 30}`) |
| `wait [RUN] [--timeout S]` | wait for a run to finish (at most `S` seconds, default 30) |
| `close` | end the session, and the simulator it started |
| `show ID`, `log [--count N]` | debugging: a stored artifact or run; recent journal entries |

Observation images are under the session's directory, `data/sessions/SESSION_ID/`. The same operations in Python
(run the file with `scripts/run FILE`):

```python
from wbc import Session

s = Session.connect("sim://flat")
s.observe()
s.plan({"walk": {"target": [0.3, 0.0]}})
s.execute()
s.wait(timeout=30)
s.close()
```

## Simulation targets

`sim://flat` is an empty floor; `sim://tabletop_pick?seed=N` is the pick-and-place scene of the README.

| Option | Meaning |
| --- | --- |
| `seed=N` | scene seed (`tabletop_pick`) |
| `heading=DEG` | the robot's initial heading |
| `paused=1` | simulated time stands still while the robot is idle |
| `vision=0` | no head camera |
| `viewer=1` | open a window on the simulation; needs a display |
| `checkpoint=DIR` | another tracker bundle (a directory with `deploy.yaml` and the ONNX file) |

`scripts/agent run` checks a `tabletop_pick` episode when it closes the session (`check` in `episode.json`): the
object ends lifted (pick) or in or on its destination (place), and the robot is still standing. For a session closed
by hand: `scripts/run -m wbc_link_server.scenes.success data/sessions/SESSION_ID/runtime/sim`.

## Agent loop: `scripts/agent`

```bash
scripts/agent tasks                                          # list the tasks
scripts/agent run tabletop_pick --backend hitch --seed 0     # a task in simulation
scripts/agent run my_task.md --backend hitch --vision URL    # a task file whose target is a robot
scripts/agent transcript data/episodes/ep-...                # rewrite transcript.md
```

| Option | Meaning |
| --- | --- |
| `TASK` | a task name in `src/wbc/agent/tasks/`, or the path of a task file |
| `--backend NAME` | a profile in `src/wbc/agent/backends.yaml` (default `fake`) |
| `--model M` | model name; overrides the profile's |
| `--target T` | `sim://SCENE[?...]` or `http://HOST:PORT`; default: the task file's `target` |
| `--seed N` | the scene seed (`sim://` targets only) |
| `--viewer` | open a window on the simulation (`sim://` targets only) |
| `--vision URL` | the camera server; required for a robot target unless `--session` is given |
| `--session URL` | use an open session; it is not closed at the end |
| `--max-turns N` | turn limit; default: the task file's `max_turns` |
| `--keep` | do not close the session the command started |

A task file is Markdown with a front matter (`target` is required; `hand: dex3` sets `WBC_HAND`; `{{instruction}}`
stands for a scene's own task sentence):

```markdown
---
target: sim://flat
max_turns: 20
---
Walk to the wall in front of you.
```

The model has three tools: `execute` plans a request, executes it, waits and observes; `perceive` measures image
points; `finish` ends the episode. Their prompts are in `src/wbc/agent/prompts/`.

An episode directory `data/episodes/ep-YYYYMMDD-HHMMSS/` holds `episode.json` (task, target, backend, why it stopped),
`messages.jsonl` (the conversation and the session results), `requests/` (each model request and response),
`images/` and `transcript.md`. Stop reasons: `finished`, `max_turns`, `no_tool_calls`, `incomplete_response`,
`backend_error`, `robot_fault`, `interrupted`, `error`.

## Model backends

The loop speaks the OpenAI Responses API (`/responses`); every request carries the whole conversation. A backend is a
profile in `src/wbc/agent/backends.yaml`:

| Profile | Server |
| --- | --- |
| `fake` | the scripted model, started by the command itself |
| `hitch` | our model at `http://localhost:8000/v1` on a server with the Responses API (for example vLLM), under the name `hitch` |
| `openai` | the hosted API; the key is read from `OPENAI_API_KEY` |

For any other server, add a profile: `api: responses`, `base_url`, `model`, `api_key_env` (the variable holding the
key), `extra_body` (merged into every request), `max_output_tokens`, `timeout_s`, `retries`.

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `WBC_SESSION` | | the session address the `wbc` commands go to |
| `WBC_DATA` | `data/` | root of sessions, episodes and robot run directories |
| `WBC_HAND` | `rubber` | the robot's hands: `rubber` or `dex3` (`dex3` in simulation only). It changes the prompt and the hand model |
| `WBC_CHECKPOINT` | `G1_Harness` in the controller | another tracker bundle (a directory with `deploy.yaml` and the ONNX file) |
| `WBC_WAIT_SCALE` | `1` | wall-clock seconds the agent allows per second of planned motion; raise it when the simulation runs slower than real time |
| `WBC_STOP_ON_FAULT` | `1` | the agent ends the episode (`robot_fault`) when the robot side fails or cannot be reached, or the camera gives no image; `0`: the model is handed the error and goes on |
| `WBC_DEPTH_REFINE`, `WBC_DEPTH_REFINE_MODE` | | URL of the depth-repair service and its mode, `full` or `fill` ([scripts/lingbot/README.md](../scripts/lingbot/README.md)) |
| `WBC_SIM_UNPACED` | | `1`: the simulator does not sleep to keep real time |
| `MUJOCO_GL` | `egl` | MuJoCo's rendering backend, for example `osmesa` on a machine without a GPU |

## Repository layout

```text
scripts/                 setup, run, wbc, agent, robot; lingbot/ (depth-repair service)
src/wbc/                 session, planner, skills, link client, agent loop
src/wbc_link_server/     robot-side processes, the MuJoCo bridge and the scene
vision/                  head-camera server
assets/                  G1 model, start pose, scene materials
docs/                    REAL_ROBOT.md, USAGE.md, REFERENCE.md
third_party/             the tracking controller, fetched by scripts/setup
```
