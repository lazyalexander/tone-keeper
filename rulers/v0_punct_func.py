"""v0: production baseline (tone_keeper.fingerprint, unchanged gates).

16 punctuation rates + 10 oral function chars (的了吗呢吧啊把被就还是不).
Distance is raw cosine to the train mean (std is stored but unused).
"""

from __future__ import annotations

from tone_keeper.fingerprint.distance import style_distance
from tone_keeper.fingerprint.features import PUNCT_FUNC_V1
from tone_keeper.fingerprint.profile import Profile, build_profile


class PunctFuncV0:
    name = "v0_punct_func"

    def build(self, train_texts: list[str]) -> Profile:
        return build_profile(
            train_texts,
            extract=PUNCT_FUNC_V1.extract,
            names=PUNCT_FUNC_V1.names(),
        )

    def distance(self, text: str, profile: Profile) -> float:
        return style_distance(text, profile, extract=PUNCT_FUNC_V1.extract)
