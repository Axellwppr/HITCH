# Interface reference

What the model sends and reads through its three tools: `execute`, `perceive` and `finish`. The request schema is in
`src/wbc/core/request.py`.

## 1. Frames

- **Frame O** is set by each observation: origin on the ground under the pelvis, x along the pelvis heading, y left,
  z up, ground at z = 0, metres. It stays fixed during one `execute`. All positions and directions in a request
  (hand waypoints, `walk.target`, `footsteps`) are in the O of the latest observation.
- The **root frame** moves with the pelvis position and heading (not its pitch or roll). Only `follow: "root"` uses
  it; the model never writes root-frame coordinates.

## 2. The execute request

```json
{"height": 0.65, "torso": {"pitch_deg": 20, "yaw_deg": 0}, "wait_s": 1.0,
 "hands": {"left": {"follow": "root"},
           "right": {"waypoints": [{"position": [0.35, -0.15, 0.80], "duration_s": 1.5,
                                    "orientation": {"palm": "down"}}], "grip": "closed"}}}
```

Every field is optional; give at least one. Angles are integer degrees.

| Field | Values | Meaning |
|---|---|---|
| `height` | 0.30–0.79 m, or `"auto"` (3.5) | Pelvis height at the end of the in-place part. Omitted: keep the current height |
| `torso` | `pitch_deg` 0–60, `yaw_deg` −45–45, multiples of 5, both required | Extra forward lean and waist turn (left +) on top of the standard posture of the height (3.2). `{0, 0}` returns to it |
| `body_duration_s` | 0.6–8 s | Time of the height and torso change; needs `height` or `torso` |
| `hands.left`, `hands.right` | 2.1 | What each hand does in the in-place part |
| `walk` | `target: [x, y]` with optional `turn_deg` (−180–180, left +); or `footsteps`: 1–12 of `side`, `xy`, optional `yaw_deg`, `root_yaw_deg` (absolute headings in O) | Walk after the in-place part. A `target` walk ends with the feet parallel |
| `wait_s` | 0.2–10 s | Stand still at the end |
| `eye` | `yaw_deg` −45–45, `pitch_deg` 0–60, multiples of 15, both required | Camera setting for the next observation (3.7). Alone, it only turns the camera and observes again |

### 2.1 Each hand

Give `waypoints`, `home` or `follow` (`follow` may go with `waypoints`), plus an optional `grip`.

| Option | Meaning |
|---|---|
| `waypoints` | 1–12 of `position` (palm in O; z is height above the ground), `duration_s` (0.4–8 s from the previous waypoint or the start) and optional `orientation` |
| `orientation` | `{"palm": "inward" \| "down" \| "up" \| "forward"}`, or `{"normal": [..], "fingers": [..]}`, two perpendicular directions in O. Omitted: the previous waypoint's orientation; for the first waypoint, the hand's intent (3.3) |
| `home: true` | The arm returns to its rest posture (elbow bent about 90°) within 1.2 s and stays there, making room for the knees and body |
| `follow` | `"root"`, `"world"` or `"body"` (3.3). With `waypoints`, it applies only while the body changes |
| `grip` | `"open"` or `"closed"`, after the in-place part. Only for hands with fingers; not with `walk` |

A hand given none of these stays at rest if its arm is near its rest posture, keeps its joint angles if it still
hangs as at the start, and otherwise follows `"root"`.

## 3. Execution rules

### 3.1 Order and timing

1. **In-place part**: height, torso and both hands start together; the feet stay planted. It lasts as long as the
   longest of them; whatever finishes first holds still. Default body time: 4 s per metre of height (at least
   1.2 s) or 25°/s of torso change (at least 1 s), whichever is longer.
2. **Walk**: both arms keep their joint angles and move with the torso.
3. **Fingers**, if `grip` is given, then 1 s of standing still plus `wait_s`. Otherwise just `wait_s`.
4. **Camera**: turns to `eye` (not after a rejected plan), and the robot is observed.

The in-place part, walk and wait are planned and checked together: if any part fails, nothing moves. If the motion
does not finish, the fingers are not moved.

An in-place interval that would exceed the speed limits is slowed, by at most 2 times; the result says which interval
took how long. A request with `height`, `torso` or `walk` rejected only for speed is planned once more with the body
change (at most 8 s) and the walk's swing 1.5 times slower.

### 3.2 Height and torso

Each pelvis height has a standard posture, shared with walking: the torso is upright at 0.62 m and above, and leans
forward from vertical by 4° at 0.60 m, 12° at 0.55, 20° at 0.50, 29° at 0.45 and 37° at 0.40 m.
`torso.pitch_deg` adds to this, waist first, then hips. The centre of mass stays over the feet, so the more the torso
leans, the further back the pelvis moves. Height, lean and turn change along one smooth profile, and these limits
(degrees) hold at every moment, at the height of that moment:

| Pelvis height ≥ | 0.785 m | 0.55 m | 0.50 m | 0.46 m | 0.42 m | 0.38 m | 0.30 m |
|---|---|---|---|---|---|---|---|
| Lean | 35 | 60 | 55 | 45 | 25 | 20 | 10 |
| Turn, lean ≤ 20° | 45 | 45 | 45 | 40 | 20 | 20 | 10 |
| Turn, lean > 20° | 35 | 40 | 40 | 20 | 10 | 15 | 10 |

### 3.3 How hands move

- **Waypoints** are fixed points in this request's O. The hand starts with the body, from where it is, moves
  smoothly through them, and stays at the last one. Position comes first (within 5 mm); orientation may be up to
  60° off.
- **`follow` + waypoints**: the hand follows while the body changes, then takes its waypoints, timed from then.
  Without a body change this is the same as waypoints alone.
- **`follow` alone**, for the whole in-place part:
  - `root`: the palm keeps its pose relative to the pelvis position and heading (within 5 mm and 5°), so it stays
    level. Leaning while crouching may make this unsolvable; then use waypoints or `body`.
  - `world`: the palm stays where it is in O (within 5 mm and 5°).
  - `body`: the arm keeps its joint angles and the torso carries the hand. Always solvable, but the hand sweeps
    through space.
- **Intent**: each hand remembers the pose its requests last asked for (its last waypoint, as asked). `root` carries
  it with the pelvis; `world` keeps it. A hand on joint angles (`body`, `home`, at rest, hanging) and every hand after
  a walk has none.

### 3.4 Walking

A walk starts where the in-place part ends. It needs the torso at `{0, 0}`, the pelvis 0.58–0.78 m high and both
feet flat, so one request can straighten up and then walk.

### 3.5 `height: "auto"`

The planner chooses the pelvis height for the hand waypoints: per waypoint interval it tries the starting height and
5 or 10 cm above or below it (within 0.58–0.77 m), and keeps the smallest change that puts the palms within 5 mm and
10°. It needs waypoints, no `torso` or `body_duration_s`, the torso in its standard posture and the pelvis
0.58–0.77 m high.

### 3.6 Stance for height changes

A height change over 10 cm needs flat, parallel, side-by-side feet, as a `target` walk leaves them. With staggered
feet, first execute `walk: {"target": [0, 0]}`.

### 3.7 Camera (`eye`)

The head camera sits on a two-axis mount on the torso (geometry in `src/wbc/robot/eye.py`). Yaw turns it left
(positive), pitch tilts it down, both relative to the torso, so leaning or turning the torso moves the view. Allowed:
yaw −45–45° and pitch 0–60° in steps of 15°; it starts at yaw 0, pitch 45°. The image is 640×400, with a 90°
horizontal and about 64° vertical field of view.

## 4. Rejection reasons

A rejection moves nothing; the last observation and its measurements stay valid. The result gives the reason, the
moment, and the numbers against their limits.

### 4.1 Arguments (`INVALID_ARGUMENTS`)

A field out of range or of the wrong type; fields that do not go together (`home` with `waypoints` or `follow`,
`grip` with `walk`, `"auto"` with `torso`); no field at all; `grip` for hands without fingers.

### 4.2 Current state (`PRECONDITION`)

| Reason | Case |
|---|---|
| `TORSO_LIMIT` | At some moment the lean or turn exceeds the limit for that height (3.2) |
| `STANCE_REQUIRED` | A height change over 10 cm without flat, parallel, side-by-side feet (3.6) |
| `WALK_TORSO`, `WALK_HEIGHT` | When the walk would start, the torso is not at `{0, 0}` or the pelvis is not 0.58–0.78 m high |
| `AUTO_HEIGHT` | `"auto"` while the torso is not in its standard posture or the pelvis is outside 0.58–0.77 m |
| `OBSERVATION_STALE` | The robot was still moving or its reference changed; it is observed again, so plan in the new O |

### 4.3 Plan checks (`PLAN_REJECTED`)

| Reason | Case |
|---|---|
| `BODY_UNREACHABLE` | The legs cannot reach the height or posture with the feet where they are |
| `HAND_REACH_LIMIT` | The palm target is farther from the shoulder than the arm reaches (about 0.49 m) |
| `HAND_POSE_UNREACHABLE` | The position is reachable, but not the orientation, joint limits or clearance from the body |
| `HAND_REST` | An arm at rest cannot make room for the body |
| `HAND_PATH_RATE_LIMIT` | An interval is too fast even slowed 2 times; the result says how much more time it needs |
| `SELF_COLLISION`, `GROUND_CONTACT`, `JOINT_LIMIT`, `SPEED_LIMIT` | Parts touch each other or the ground; a joint or speed exceeds its limit |
| `FEET_MOVED` | A foot is dragged during the in-place part |
| Walk reasons | `STANCE_REQUIRED`, `FLAT_SUPPORT_REQUIRED`, `REQUEST_LIMIT`, `TARGET_NOT_REACHED`, `SPEED_LIMIT` and others, with the attempts made |
| Internal | `INVALID_TRAJECTORY`, `HAND_TIMING_GEOMETRY`, `HAND_TIMING_SOLVE_FAILED`, `HAND_START_LIMIT` |

Collisions with the scene (a hand on a table, a foot against a box) are judged by the model from the images and
measurements.

### 4.4 Execution

A plan that passes can still fail: the controller refuses it (the robot does not move), the run is lost (the
controller restarted or failed), or the plan went stale before it was sent. The result says which.

## 5. Feedback

An `execute` result gives, in order:

1. **Plan verdict**: the duration of the in-place part and the walk, and what the planner changed (slowed intervals,
   the heights `"auto"` chose, a stance correction before a walk, a slower retry).
2. **Execution status**: executed to the end, refused, lost, or still running when the wait ended.
3. **Fingers**, if `grip` was given.
4. **The new observation** (5.1).

The exact wording comes from `src/wbc/agent/render.py`; the `transcript.md` of any episode shows it.

### 5.1 Observation

- **Frame change**: where the new O lies in the previous one and how to carry a point over, if the feet stayed
  planted. After a walk or a controller restart no numbers are given; measure again.
- **Body**: posture; pelvis height (measured and planned), roll and pitch; torso lean and turn beyond the standard
  posture, and the total lean from vertical.
- **Feet**: ankle position and yaw, measured and planned.
- **Hands**: palm position, `normal` and `fingers`, measured and planned, and the gap between them; the farthest the
  palm was from its reference during the motion; "planned as requested" or the distance from the intent; fingers
  open or closed.
- **Camera**: the `eye` setting and where the view points (angle below horizontal, heading in O).
- **Images**.

Planned values are the controller's reference; measured values come from the joint encoders and the IMU. Both are
relative to the measured pelvis, so they do not show the whole robot drifting during a walk.

## 6. Other tools

- `perceive(points)`: 1–100 points on the latest image, as integers normalized to 0–1000 (top-left [0, 0]). Returns
  their 3D positions in O and the image with the points marked.
- `finish(success, summary)`: ends the episode. It must be the only call of its reply.
- One reply may hold several calls, run in order. After an `execute` has moved the robot, the later calls of that
  reply are not run, because their coordinates refer to the old O.
