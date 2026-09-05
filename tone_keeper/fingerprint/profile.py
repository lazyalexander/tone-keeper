from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from collections.abc import Callable

from tone_keeper.fingerprint.features import extract_features, feature_names

Extractor = Callable[[str], np.ndarray]


@dataclass
class Profile:
    vocab: list[str]
    mean: list[float]
    std: list[float]
    names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_from_dict(data: dict[str, Any]) -> Profile:
    return Profile(
        vocab=list(data.get("vocab", [])),
        mean=list(data["mean"]),
        std=list(data["std"]),
        names=list(data["names"]),
    )


def build_profile(
    texts: list[str],
    ngram_k: int = 512,
    extract: Extractor | None = None,
    names: list[str] | None = None,
) -> Profile:
    del ngram_k
    if not texts:
        raise ValueError("build_profile requires at least one text")
    fn = extract or extract_features
    matrix = np.vstack([fn(text) for text in texts])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return Profile(
        vocab=[],
        mean=mean.tolist(),
        std=std.tolist(),
        names=list(names) if names is not None else feature_names(),
    )
