#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -x .venv/bin/python || ! -f .env.local ]]; then
  echo "ERROR: run scripts/bootstrap_server.sh and configure .env.local first." >&2
  exit 1
fi

set -a
source .env.local
set +a

if [[ -z "${MODEL_PATH:-}" || ! -f "$MODEL_PATH/config.json" ]]; then
  echo "ERROR: MODEL_PATH must point to a local model containing config.json." >&2
  exit 1
fi

dataset_path="${GSM8K_TEST_PATH:-${GSM8K_DATASET_PATH:-${RLDLLM_DATASETS_DIR:-/mnt/datasets}/gsm8k}}"
if [[ ! -e "$dataset_path" ]]; then
  echo "ERROR: GSM8K dataset path does not exist: $dataset_path" >&2
  exit 1
fi

export WANDB_MODE="${WANDB_MODE:-disabled}"
n_test="${N_TEST:-20}"
batch_size="${BATCH_SIZE:-1}"
gen_length="${GEN_LENGTH:-128}"
block_length="${BLOCK_LENGTH:-32}"
unmask_threshold="${UNMASK_THRESHOLD:-0.7}"
remask_threshold="${REMASK_THRESHOLD:-0.5}"
remask_min_age="${REMASK_MIN_AGE:-1}"
max_remasks_per_token="${MAX_REMASKS_PER_TOKEN:-2}"
remask_max_extra_steps="${REMASK_MAX_EXTRA_STEPS:-16}"
output_root="${OUTPUT_DIR:-results}"
run_timestamp="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
run_output_dir="${output_root}/${run_timestamp}_threshold_remask_t${remask_threshold}"

echo "Running GSM8K threshold-remask evaluation..."
echo "Dataset: $dataset_path"
echo "Samples: $n_test"
echo "Batch size: $batch_size"
echo "Generation length: $gen_length"
echo "Unmask threshold: $unmask_threshold"
echo "Remask threshold: $remask_threshold"
echo "Remask minimum age: $remask_min_age"
echo "Maximum remasks per token: $max_remasks_per_token"
echo "Maximum extra steps: $remask_max_extra_steps"
echo "Results directory: $run_output_dir"

eval_args=(
  --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml
  --model_path "$MODEL_PATH"
  --dataset gsm8k
  --batch_size "$batch_size"
  --gen_length "$gen_length"
  --block_length "$block_length"
  --remasking fastdllm
  --thres "$unmask_threshold"
  --remask_threshold "$remask_threshold"
  --remask_min_age "$remask_min_age"
  --max_remasks_per_token "$max_remasks_per_token"
  --remask_max_extra_steps "$remask_max_extra_steps"
  --suffix "threshold_remask_t${remask_threshold}_${run_timestamp}"
  --output_dir "$run_output_dir"
  --seed 42
)

if [[ "$n_test" != "all" ]]; then
  eval_args+=(--n_test "$n_test")
fi

.venv/bin/python -m eval.eval "${eval_args[@]}"
