#!/bin/bash
# The LingBot-Depth service on one GPU of this host, from the directory setup.sh installed (this script's own).
# usage: serve.sh [GPU=0] [PORT=18890] [HOST=127.0.0.1]   (caches in $LINGBOT_CACHE, default cache/ next to it)
L=$(cd "$(dirname "$0")" && pwd)
C=${LINGBOT_CACHE:-$L/cache}
mkdir -p "$C"
export CUDA_VISIBLE_DEVICES=${1:-0} PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XDG_CACHE_HOME=$C TORCH_HOME=$C/torch \
  TRITON_CACHE_DIR=$C/triton OMP_NUM_THREADS=4
unset PYTHONPATH LD_LIBRARY_PATH
exec "$L/venv/bin/python" "$L/lingbot_server.py" --repo "$L/lingbot-depth" --weights "$L/weights/lingbot-v05.pt" \
  --port "${2:-18890}" --host "${3:-127.0.0.1}"
