"""Re-export distances from rulers.core. Production gates use cosine."""

from collections.abc import Callable

import numpy as np

from rulers.core import Profile, distance_cosine, distance_zl2, scheme_from_profile

Extractor = Callable[[str], np.ndarray]


def style_distance(text: str, profile: Profile, extract: Extractor | None = None) -> float:
    fn = extract or scheme_from_profile(profile).extract
    return distance_cosine(text, profile, fn)


def style_distance_zl2(text: str, profile: Profile, extract: Extractor | None = None) -> float:
    fn = extract or scheme_from_profile(profile).extract
    return distance_zl2(text, profile, fn)
