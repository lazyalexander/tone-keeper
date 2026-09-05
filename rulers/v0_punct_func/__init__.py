"""v0 kit: production baseline. 16 punct + 10 oral function chars, d_cos to mean.

Same instrument as tone_keeper.infer.gates. Swap the whole package, not parts.
"""

from __future__ import annotations

from rulers.core import FeatureScheme, Profile, build_profile, distance_cosine

PUNCT = list("，。！？…—、：；,.!?~")
FUNCTION = list("的了吗呢吧啊把被就还是不")
SCHEME = FeatureScheme("punct_func_v1", tuple(PUNCT), tuple(FUNCTION))
PUNCT_FUNC_V1 = SCHEME


class PunctFuncV0:
    name = "v0_punct_func"

    def build(self, train_texts: list[str]) -> Profile:
        return build_profile(
            train_texts,
            extract=SCHEME.extract,
            names=SCHEME.names(),
            ruler=self.name,
        )

    def distance(self, text: str, profile: Profile) -> float:
        return distance_cosine(text, profile, SCHEME.extract)


Ruler = PunctFuncV0
