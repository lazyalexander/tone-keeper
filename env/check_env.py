"""Verify the pixi env: Python 3.12, MLX Metal, mlx-lm import."""

from __future__ import annotations

import sys
from importlib import metadata


def main() -> int:
    errors: list[str] = []

    if sys.version_info[:2] != (3, 12):
        errors.append(f"python {sys.version.split()[0]} != 3.12")

    try:
        import mlx
        import mlx.core as mx
    except ImportError as exc:
        errors.append(f"mlx import failed: {exc}")
    else:
        try:
            version = metadata.version("mlx")
        except metadata.PackageNotFoundError:
            version = getattr(mlx, "__version__", "?")
        metal = bool(mx.metal.is_available())
        print(f"mlx {version}")
        print(f"metal_available {metal}")
        if not metal:
            errors.append("mlx.core.metal.is_available() is False")

    try:
        import mlx_lm
    except ImportError as exc:
        errors.append(f"mlx_lm import failed: {exc}")
    else:
        try:
            lm_version = metadata.version("mlx-lm")
        except metadata.PackageNotFoundError:
            lm_version = getattr(mlx_lm, "__version__", "?")
        print(f"mlx_lm {lm_version}")

    for name in ("numpy", "scipy", "sklearn", "transformers", "jieba", "huggingface_hub", "mlx_embeddings"):
        try:
            __import__(name)
        except ImportError as exc:
            errors.append(f"{name} import failed: {exc}")

    print(f"executable {sys.executable}")
    if errors:
        print("FAILED")
        for item in errors:
            print(f"  {item}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
