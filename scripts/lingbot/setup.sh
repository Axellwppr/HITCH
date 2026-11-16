#!/bin/bash
# Build the self-contained LingBot-Depth service at $LINGBOT_ROOT (default ~/lingbot):
#   lingbot-depth/   the official code at COMMIT
#   weights/         the v0.5 checkpoint, SHA256 checked
#   python/, venv/   a uv-managed CPython 3.11 and a venv with torch 2.9.1+cu128 and the official xFormers
#                    0.0.33.post2 (builds for CUDA 12.8, which recent GPUs need); the model's attention runs on
#                    xFormers, the upstream default path
#   lingbot_server.py, serve.sh, README.md   from this directory
# Needs git, curl and uv on PATH. Idempotent: a second run only refreshes the scripts and checks the rest.
set -euo pipefail
L=${LINGBOT_ROOT:-$HOME/lingbot}
HERE=$(cd "$(dirname "$0")" && pwd)
COMMIT=f3a237e434ae987bc38281476d6cfb5df3e4d739
SHA=b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23
export UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=$L/python
UV=${UV:-uv}
mkdir -p "$L/weights"
if [ ! -d "$L/lingbot-depth" ]; then
  git clone https://github.com/Robbyant/lingbot-depth.git "$L/lingbot-depth"
fi
(cd "$L/lingbot-depth" && { [ "$(git rev-parse HEAD)" = $COMMIT ] || git checkout -q $COMMIT; })
w=$L/weights/lingbot-v05.pt
if [ ! -f "$w" ]; then
  curl -fL -o "$w.part" https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5/resolve/main/model.pt
  mv "$w.part" "$w"
fi
echo "$SHA  $w" | sha256sum -c -
if [ ! -x "$L/venv/bin/python" ]; then
  $UV python install 3.11
  $UV venv --python 3.11 "$L/venv"
fi
$UV pip install --python "$L/venv/bin/python" --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match torch==2.9.1+cu128 torchvision==0.24.1+cu128 xformers==0.0.33.post2 \
  numpy scipy opencv-python-headless pillow huggingface_hub click matplotlib trimesh
cp "$HERE"/lingbot_server.py "$HERE"/serve.sh "$HERE"/README.md "$L/"
chmod +x "$L/serve.sh"
"$L/venv/bin/python" -c "import torch, xformers, xformers.ops; print('torch', torch.__version__, 'xformers', \
xformers.__version__, 'cuda', torch.version.cuda)"
echo "built $L"
