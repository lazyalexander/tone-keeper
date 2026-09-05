from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tone_keeper.fingerprint.features import extract_features
from tone_keeper.fingerprint.profile import Profile

Extractor = Callable[[str], np.ndarray]


def style_distance(text: str, profile: Profile, extract: Extractor | None = None) -> float:
    fn = extract or extract_features
    vector = fn(text)
    mean = np.asarray(profile.mean, dtype=np.float64)
    denom = float(np.linalg.norm(vector) * np.linalg.norm(mean))
    if denom == 0.0:
        return 1.0
    cosine = float(np.dot(vector, mean) / denom)
    return 1.0 - cosine


def style_distance_zl2(text: str, profile: Profile, extract: Extractor | None = None) -> float:
    """L2 of z-scored rates vs corpus mean/std. Cosine-to-mean is undefined in z-space."""
    fn = extract or extract_features
    vector = fn(text)
    mean = np.asarray(profile.mean, dtype=np.float64)
    std = np.asarray(profile.std, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    z = (vector - mean) / std
    return float(np.linalg.norm(z))
