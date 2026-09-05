"""Generic style-instrument math. Kits own alphabets and distances they choose.

    f_c(x) = count_c(x) / max(cjk_len(x), 1)
    f(x)   = concat(f_punct(x), f_func(x))
    μ, σ   = mean and std of f(u) over train U
    F      = (μ, σ)

    d_cos(x, F) = 1 - <f(x), μ> / (||f(x)|| ||μ||)
    d_z(x, F)   = || (f(x) - μ) / σ ||_2
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tone_keeper.text.length import cjk_len

Extractor = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class FeatureScheme:
    name: str
    punct: tuple[str, ...]
    func: tuple[str, ...]

    def names(self) -> list[str]:
        return [f"punct:{ch}" for ch in self.punct] + [f"func:{ch}" for ch in self.func]

    def extract(self, text: str) -> np.ndarray:
        length = max(cjk_len(text), 1)
        compact = "".join(ch for ch in text if not ch.isspace())
        punct = np.array([compact.count(ch) / length for ch in self.punct], dtype=np.float64)
        func = np.array([compact.count(ch) / length for ch in self.func], dtype=np.float64)
        return np.concatenate([punct, func])


@dataclass
class Profile:
    vocab: list[str]
    mean: list[float]
    std: list[float]
    names: list[str]
    ruler: str = "v0_punct_func"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_from_dict(data: dict[str, Any]) -> Profile:
    return Profile(
        vocab=list(data.get("vocab", [])),
        mean=list(data["mean"]),
        std=list(data["std"]),
        names=list(data["names"]),
        ruler=str(data.get("ruler") or "v0_punct_func"),
    )


def scheme_from_profile(profile: Profile) -> FeatureScheme:
    punct = tuple(n.split(":", 1)[1] for n in profile.names if n.startswith("punct:"))
    func = tuple(n.split(":", 1)[1] for n in profile.names if n.startswith("func:"))
    if not punct and not func:
        raise ValueError("profile.names has no punct:/func: entries")
    return FeatureScheme(profile.ruler or "from_profile", punct, func)


def build_profile(
    texts: list[str],
    *,
    extract: Extractor,
    names: list[str],
    ruler: str,
) -> Profile:
    if not texts:
        raise ValueError("build_profile requires at least one text")
    matrix = np.vstack([extract(text) for text in texts])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return Profile(
        vocab=[],
        mean=mean.tolist(),
        std=std.tolist(),
        names=list(names),
        ruler=ruler,
    )


def distance_cosine(text: str, profile: Profile, extract: Extractor | None = None) -> float:
    fn = extract or scheme_from_profile(profile).extract
    vector = fn(text)
    mean = np.asarray(profile.mean, dtype=np.float64)
    denom = float(np.linalg.norm(vector) * np.linalg.norm(mean))
    if denom == 0.0:
        return 1.0
    cosine = float(np.dot(vector, mean) / denom)
    return 1.0 - cosine


def distance_zl2(text: str, profile: Profile, extract: Extractor | None = None) -> float:
    fn = extract or scheme_from_profile(profile).extract
    vector = fn(text)
    mean = np.asarray(profile.mean, dtype=np.float64)
    std = np.asarray(profile.std, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    z = (vector - mean) / std
    return float(np.linalg.norm(z))
