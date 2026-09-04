from __future__ import annotations

import numpy as np

from tone_keeper.fingerprint.features import extract_features
from tone_keeper.fingerprint.profile import Profile


def style_distance(text: str, profile: Profile) -> float:
    vector = extract_features(text, profile.vocab)
    mean = np.asarray(profile.mean, dtype=np.float64)
    denom = float(np.linalg.norm(vector) * np.linalg.norm(mean))
    if denom == 0.0:
        return 1.0
    cosine = float(np.dot(vector, mean) / denom)
    return 1.0 - cosine
