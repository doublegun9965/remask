# Server setup and baseline smoke test

This branch keeps Apple's algorithm unchanged and adds a reproducible path from
an empty Linux GPU server to a one-example LLaDA evaluation.

## Prerequisites

The server must already provide:

- Linux on x86_64
- a GPU with sufficient memory for LLaDA-8B
- a working GPU-enabled PyTorch installation in the base image (ROCm or CUDA)
- `git` and `curl`
- outbound access to GitHub, PyPI, Astral, Hugging Face, and the dataset host

The bootstrap script installs `uv`, creates a local virtual environment that
reuses the base image's GPU-enabled PyTorch, and installs this project's
remaining Python dependencies. It deliberately does not replace PyTorch or
modify the system GPU driver. CUDA-specific optional packages such as
`bitsandbytes` are not required for the baseline smoke test.
Accelerate, PEFT, and TRL are installed without resolving their `torch`
dependency so that uv cannot shadow the base image's ROCm build with a CUDA
wheel.
The release's unimplemented S3 placeholder is not installed for local runs;
this avoids mixing the cloud image's boto3 with an incompatible botocore.

## First deployment

```bash
git clone --branch baseline-reproduction \
  https://github.com/doublegun9965/remask.git
cd remask
bash scripts/bootstrap_server.sh
```

The bootstrap creates `.env.local` from the committed `.env.example` template.
Edit `.env.local` and set `MODEL_PATH` to the absolute directory containing the
locally downloaded `GSAI-ML/LLaDA-8B-Instruct` files. No API is used for
inference, and a Hugging Face token is not required for a complete local model.
`.env.local` is ignored by Git and must never be committed.
Set `RLDLLM_DATASETS_DIR` to the dataset root; GSM8K is read from its `gsm8k`
subdirectory. Alternatively, set `GSM8K_DATASET_PATH` directly to the actual
downloaded directory.
`PYPI_INDEX_URL` defaults to the Aliyun mirror and can be changed in the same
file. Re-running the bootstrap reuses an existing `.venv` instead of prompting
to replace it. Existing `.env.local` files created by an older revision should
add `PYPI_INDEX_URL=https://mirrors.aliyun.com/pypi/simple`; otherwise the
bootstrap still falls back to that mirror automatically.

```bash
nano .env.local
bash scripts/smoke_eval.sh
```

The smoke run uses one embedded arithmetic question and performs no dataset or
model downloads. The model is loaded from `MODEL_PATH`. Success means one
example completes without training a policy or writing evaluation artifacts.

## Updating the server

```bash
git switch baseline-reproduction
git pull --ff-only
bash scripts/bootstrap_server.sh
bash scripts/smoke_eval.sh
```

## Troubleshooting boundary

- If environment verification reports `GPU available: False`, select a cloud
  image with a PyTorch build compatible with its CUDA or ROCm environment, then rerun
  `scripts/check_environment.py`.
- If an older bootstrap installed a CUDA PyTorch inside `.venv`, run
  `uv pip uninstall --python .venv/bin/python torch` and rerun the bootstrap.
- If an older bootstrap installed the unused S3 stack, run
  `uv pip uninstall --python .venv/bin/python s3fs aiobotocore botocore`.
- Do not put tokens, model weights, checkpoints, or cluster logs in Git.
