from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

import torch


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {package_version('transformers')}")
    print(f"Accelerate: {package_version('accelerate')}")
    print(f"TRL: {package_version('trl')}")
    print(f"ROCm/HIP: {torch.version.hip}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"GPU available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print(
            "ERROR: this PyTorch installation cannot access a GPU. Check the "
            "container device mapping and the installed PyTorch build.",
            file=sys.stderr,
        )
        return 1

    backend = "ROCm" if torch.version.hip else "CUDA"
    print(f"Accelerator backend: {backend}")

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        memory_gib = props.total_memory / 1024**3
        name = props.name or "AMD GPU (product name unavailable in container)"
        print(f"GPU {index}: {name} ({memory_gib:.1f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
