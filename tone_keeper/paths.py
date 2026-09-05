from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    override = os.environ.get("TONE_KEEPER_ROOT")
    return Path(override) if override else Path.cwd()


def work_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "data" / "work"


def lora_root(root: Path | None = None) -> Path:
    return (root or project_root()) / "lora"


def pairs_path(root: Path | None = None) -> Path:
    return lora_root(root) / "pairs.jsonl"


def dropped_pairs_path(root: Path | None = None) -> Path:
    return lora_root(root) / "pairs.dropped.jsonl"


def sft_dir(root: Path | None = None) -> Path:
    return lora_root(root) / "sft"


def adapter_dir(root: Path | None = None) -> Path:
    return lora_root(root) / "adapters" / "default"


def logs_dir(root: Path | None = None) -> Path:
    return lora_root(root) / "logs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
