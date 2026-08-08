#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: run scripts/bootstrap_server.sh first." >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "ERROR: copy .env.example to .env and set HF_TOKEN." >&2
  exit 1
fi

set -a
source .env
set +a

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "ERROR: set MODEL_PATH in .env." >&2
  exit 1
fi

if [[ "$MODEL_PATH" != /* ]]; then
  echo "ERROR: MODEL_PATH must be an absolute path, got: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "ERROR: model config not found at: $MODEL_PATH/config.json" >&2
  exit 1
fi

export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "Running one-example LLaDA baseline evaluation..."
.venv/bin/python -m eval.eval \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --model_path "$MODEL_PATH" \
  --dataset gsm8k \
  --n_test 1 \
  --batch_size 1 \
  --gen_length 32 \
  --block_length 32 \
  --remasking low_confidence \
  --diffusion_steps 32 \
  --seed 42 \
  --dont_save
