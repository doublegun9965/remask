# Server setup and baseline smoke test

This branch keeps Apple's algorithm unchanged and adds a reproducible path from
an empty Linux GPU server to a one-example LLaDA evaluation.

## Prerequisites

The server must already provide:

- Linux on x86_64
- an NVIDIA GPU with sufficient memory for LLaDA-8B
- a working NVIDIA driver (`nvidia-smi` must succeed)
- `git` and `curl`
- outbound access to GitHub, PyPI, Astral, Hugging Face, and the dataset host

The bootstrap script installs Python 3.12, `uv`, a local virtual environment,
and this project's Python dependencies. It deliberately does not modify the
system NVIDIA driver.

## First deployment

```bash
git clone --branch baseline-reproduction \
  https://github.com/doublegun9965/remask.git
cd remask
bash scripts/bootstrap_server.sh
```

Edit `.env` and set `MODEL_PATH` to the absolute directory containing the
locally downloaded `GSAI-ML/LLaDA-8B-Instruct` files. No API is used for
inference, and a Hugging Face token is not required for a complete local model.

```bash
nano .env
bash scripts/smoke_eval.sh
```

The first smoke run may still download GSM8K, so it can take longer than later
runs. The model itself is loaded from `MODEL_PATH`. Success means one example
completes without training a policy or writing evaluation artifacts.

## Updating the server

```bash
git switch baseline-reproduction
git pull --ff-only
bash scripts/bootstrap_server.sh
bash scripts/smoke_eval.sh
```

## Troubleshooting boundary

- If `nvidia-smi` fails, repair the server driver before running the bootstrap.
- If environment verification reports `CUDA available: False`, install a
  PyTorch wheel compatible with the server driver/CUDA environment, then rerun
  `scripts/check_environment.py`.
- Do not put tokens, model weights, checkpoints, or cluster logs in Git.
