# G1 vision server

An HTTP server for the robot's head camera: an Intel RealSense D455 on a two-axis head (two ST3215 servos for yaw and
pitch; CAD, 3D-printed parts and wiring:
[head assembly files](https://drive.google.com/drive/folders/1Sjem4y6XnZZKC2NSFvBJp-6FXwfXIiPq?usp=drive_link)).
It turns the head and returns RGB-D frames with the camera pose in the torso frame. The harness connects to it with
`scripts/wbc connect LINK_URL --vision VISION_URL`.

## Start

Wiring, building and the udev rule are in [docs/REAL_ROBOT.md](../docs/REAL_ROBOT.md). `scripts/robot vision` serves
on port 18764 (127.0.0.1; reach it through SSH port forwarding, as the server has no authentication). If
`scripts/robot build` cannot find librealsense, set `REALSENSE_PREFIX` to its install prefix. Stop other programs
that use the camera or the serial port first.

## `config.json`

The fields you usually edit:

- `camera.serial`: the D455's serial number. Empty takes the first camera found; set it if several are connected.
- `head.port`: the servo adapter's serial device.
- `head.yaw`, `head.pitch`: per axis, `zero_tick` (encoder tick at yaw centred / pitch level), `direction` (1 or -1)
  and `range_deg` (allowed range in degrees).

## Calibration

The camera pose comes from the head's design geometry (`g1_vision/geometry.py`). If you measure how your head is
mounted, put the corrections in `calibration.json` next to the config, in the format of `calibration.nominal.json`
(two 4x4 transforms, identity there); the server picks it up on the next request.

## Requests

Angles are in degrees relative to the torso: yaw positive to the left, pitch positive downward. Replies are
`{"ok": true, "result": {...}}` or `{"ok": false, "error": {"code", "message"}}`; a request that arrives while the
camera is busy may get `BUSY`.

| Request | Effect |
| --- | --- |
| `GET /vision` | frame at the current head angles |
| `POST /vision {"eye": {"yaw_deg": 0, "pitch_deg": 45}}` | turn the head, wait until it is still, then a frame |
| `POST /aim {"eye": {...}}` | start turning the head and return at once |
| `GET /head`, `POST /head {"eye": {...}}` | read or move the head (servo angles), no image |

A `/vision` frame is 640x400 RGB with depth aligned to it (metres, NaN where missing), plus the intrinsics and the
camera pose in the torso frame. Images are zlib-delta encoded, lossless.
