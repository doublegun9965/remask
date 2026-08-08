#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -f .env.local ]]; then
  cp .env.example .env.local
  echo "Created .env.local. Set MODEL_PATH to the absolute local model directory."
fi

set -a
source .env.local
set +a

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required. Install it with your system package manager." >&2
  exit 1
fi

echo "== Verifying the preinstalled GPU PyTorch =="
python scripts/check_environment.py

if ! command -v uv >/dev/null 2>&1; then
  echo "== Installing uv in the current user account =="
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "== Creating an environment that reuses the ROCm PyTorch build =="
if [[ -x .venv/bin/python ]]; then
  echo "Reusing existing virtual environment: .venv"
else
  uv venv --clear --python "$(command -v python)" --system-site-packages .venv
fi

echo "== Installing project dependencies =="
export UV_DEFAULT_INDEX="${PYPI_INDEX_URL:-${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}}"
echo "Python package index: $UV_DEFAULT_INDEX"
uv pip install --python .venv/bin/python \
  "transformers==4.53.0" \
  "accelerate==1.4.0" \
  "peft==0.15.1" \
  "trl==0.19.1" \
  "datasets==4.0.0" \
  "tiktoken==0.9.0" \
  "wandb>=0.16.0" \
  sentencepiece evaluate scipy tqdm regex scikit-learn \
  python-dotenv "numpy>=1.26.0" pandas s3fs matplotlib

echo "== Verifying the completed environment =="
.venv/bin/python scripts/check_environment.py

echo "Bootstrap complete. Activate with: source .venv/bin/activate"
