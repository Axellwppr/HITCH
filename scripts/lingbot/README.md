# LingBot-Depth depth repair service

An HTTP service that repairs the real camera's depth image with
[LingBot-Depth](https://github.com/Robbyant/lingbot-depth). `setup.sh` downloads the official code and weights.

## Requirements

An x86-64 machine with an NVIDIA GPU and a driver that supports CUDA 12.8; `git`, `curl` and
[uv](https://docs.astral.sh/uv/) on `PATH`.

## Install

```bash
LINGBOT_ROOT=~/lingbot bash scripts/lingbot/setup.sh     # ~/lingbot is the default
```

This builds a self-contained directory with the official code, the checked weights, a Python 3.11 environment and the
`serve.sh` launcher. Run it again to refresh it.

## Start

```bash
~/lingbot/serve.sh 0 18890 127.0.0.1
```

The arguments are the GPU, the port and the listen address (defaults 0, 18890 and 127.0.0.1). One process serves one
request at a time.

## Use from the harness

On the machine that runs the harness session, set `WBC_DEPTH_REFINE` to the service URL, for example
`http://127.0.0.1:18890`. If the service runs on another GPU machine, forward the port with
`ssh -L 18890:127.0.0.1:18890 GPU_HOST`, or start it on another listen address.

`WBC_DEPTH_REFINE_MODE` chooses how depth is repaired:

- `full` (default): the model's depth wherever it gives one, replacing measurements.
- `fill`: measurements are kept; the model only fills the holes.

If a call fails, the observation reports the camera as failed and the agent stops, unless `WBC_STOP_ON_FAULT=0`.
Without `WBC_DEPTH_REFINE` depth is used as measured.

## Interface

- `POST /refine` with `rgb`, `depth` (metres) and `mode`; replies with the repaired `depth` and a per-pixel
  `provenance` (0 none, 1 measured, 2 hole filled, 3 measurement replaced).
- `GET /health`: model, weights checksum, GPU and library versions.
