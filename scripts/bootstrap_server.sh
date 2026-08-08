#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable. Install a compatible NVIDIA driver first." >&2
  exit 1
fi

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required. Install it with your system package manager." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "== Installing uv in the current user account =="
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "== Creating Python 3.12 environment =="
uv python install 3.12
uv venv --python 3.12 .venv

echo "== Installing project dependencies =="
uv pip install --python .venv/bin/python -e .

echo "== Verifying Python and CUDA =="
.venv/bin/python scripts/check_environment.py

if [[ ! -f .env.local ]]; then
  cp .env.example .env.local
  echo "Created .env.local. Set MODEL_PATH to the absolute local model directory."
fi

echo "Bootstrap complete. Activate with: source .venv/bin/activate"
