#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: run scripts/bootstrap_server.sh first." >&2
  exit 1
fi

if [[ ! -f .env.local ]]; then
  echo "ERROR: copy .env.example to .env.local and set MODEL_PATH." >&2
  exit 1
fi

set -a
source .env.local
set +a

if [[ -z "${MODEL_PATH:-}" || ! -f "$MODEL_PATH/config.json" ]]; then
  echo "ERROR: MODEL_PATH must point to a local model containing config.json." >&2
  exit 1
fi

export WANDB_MODE="${WANDB_MODE:-disabled}"
unmask_threshold="${UNMASK_THRESHOLD:-0.7}"
remask_threshold="${REMASK_THRESHOLD:-0.5}"

echo "Running offline threshold-remask smoke example..."
echo "Unmask threshold: $unmask_threshold"
echo "Remask threshold: $remask_threshold"
.venv/bin/python -m eval.eval \
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
  --model_path "$MODEL_PATH" \
  --dataset smoke \
  --n_test 1 \
  --batch_size 1 \
  --gen_length 32 \
  --block_length 32 \
  --remasking fastdllm \
  --thres "$unmask_threshold" \
  --remask_threshold "$remask_threshold" \
  --remask_min_age 1 \
  --max_remasks_per_token 2 \
  --remask_max_extra_steps 16 \
  --seed 42 \
  --dont_save
