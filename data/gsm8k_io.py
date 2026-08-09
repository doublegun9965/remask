import os
from pathlib import Path

from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from datasets import load_from_disk


_FILE_BUILDERS = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".parquet": "parquet",
}


def gsm8k_path(split: str) -> Path:
    """Resolve a split-specific file or a shared save_to_disk directory."""
    split_override = os.environ.get(f"GSM8K_{split.upper()}_PATH")
    shared_override = os.environ.get("GSM8K_DATASET_PATH")
    if split_override:
        return Path(split_override)
    if shared_override:
        return Path(shared_override)
    root = Path(os.environ.get("RLDLLM_DATASETS_DIR", "/mnt/datasets"))
    return root / "gsm8k"


def load_local_gsm8k_split(path: Path, split: str) -> Dataset:
    """Load one GSM8K split from a HF directory or a raw local data file."""
    if path.is_dir():
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise KeyError(
                    f"GSM8K directory {path} has no {split!r} split; "
                    f"available splits: {list(dataset.keys())}"
                )
            return dataset[split]
        return dataset

    if path.is_file():
        builder = _FILE_BUILDERS.get(path.suffix.lower())
        if builder is None:
            supported = ", ".join(sorted(_FILE_BUILDERS))
            raise ValueError(
                f"Unsupported GSM8K file type {path.suffix!r}. "
                f"Supported types: {supported}"
            )
        return load_dataset(builder, data_files={split: str(path)})[split]

    raise FileNotFoundError(f"GSM8K path does not exist: {path}")


def load_gsm8k_split(split: str) -> Dataset:
    path = gsm8k_path(split)
    if path.exists():
        split_override = os.environ.get(f"GSM8K_{split.upper()}_PATH")
        other_split = "test" if split == "train" else "train"
        if (
            path.is_file()
            and not split_override
            and other_split in path.stem.lower()
        ):
            raise ValueError(
                f"GSM8K_DATASET_PATH points to a {other_split} file ({path}), "
                f"but the {split} split was requested. Set "
                f"GSM8K_{split.upper()}_PATH in .env.local."
            )
        return load_local_gsm8k_split(path, split)
    return load_dataset("openai/gsm8k", "main", split=split)
