"""Download student and teacher MLX weights into the Hugging Face cache."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tone_keeper.config import load


def main() -> int:
    cfg = load("configs/default.toml")
    from mlx_lm import load as mlx_load

    print(f"loading student: {cfg.student.model}", flush=True)
    mlx_load(cfg.student.model)
    print("ok student", flush=True)
    from mlx_embeddings import load as embed_load

    print(f"loading embedder: {cfg.embed.model}", flush=True)
    embed_load(cfg.embed.model)
    print("ok embedder", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
