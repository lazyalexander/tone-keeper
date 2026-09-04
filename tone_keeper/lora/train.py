"""LoRA SFT subprocess wrapper. Loss is token CE only."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tone_keeper.config import Config
from tone_keeper.paths import adapter_dir, project_root, sft_dir


def lora_command(cfg: Config, data_dir: Path, adapter_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        cfg.student.model,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_path),
        "--iters",
        str(cfg.lora.iters),
        "--batch-size",
        str(cfg.lora.batch_size),
        "--learning-rate",
        str(cfg.lora.learning_rate),
        "--fine-tune-type",
        cfg.lora.fine_tune_type,
        "--num-layers",
        str(cfg.lora.num_layers),
        "--max-seq-length",
        str(cfg.lora.max_seq_length),
        "--optimizer",
        "adamw",
    ]
    if cfg.lora.mask_prompt:
        cmd.append("--mask-prompt")
    return cmd


def run_sft(cfg: Config, data_dir: Path | None = None, adapter_path: Path | None = None) -> int:
    root = project_root()
    data = data_dir or sft_dir(root)
    adapter = adapter_path or adapter_dir(root)
    adapter.mkdir(parents=True, exist_ok=True)
    train_file = data / "train.jsonl"
    if not train_file.exists() or train_file.stat().st_size == 0:
        raise FileNotFoundError(f"missing SFT train file: {train_file}")
    cmd = lora_command(cfg, data, adapter)
    completed = subprocess.run(cmd, check=False)
    return completed.returncode
