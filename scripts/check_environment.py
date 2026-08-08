from __future__ import annotations

import platform
import sys

import accelerate
import torch
import transformers
import trl


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Accelerate: {accelerate.__version__}")
    print(f"TRL: {trl.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if sys.version_info[:2] != (3, 12):
        print("ERROR: this repository requires Python 3.12.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print(
            "ERROR: this PyTorch installation cannot access CUDA. Check the "
            "driver and the installed PyTorch wheel.",
            file=sys.stderr,
        )
        return 1

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        memory_gib = props.total_memory / 1024**3
        print(f"GPU {index}: {props.name} ({memory_gib:.1f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
