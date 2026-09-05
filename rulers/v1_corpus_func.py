"""v1: punctuation plus function chars that actually occur in this corpus.

Function inventory is selected from train U (min_count), excluding a small
name-char set so character names are not treated as style.

    d_cos(x, F) = 1 - <f(x), μ> / (||f(x)|| ||μ||)
"""

from __future__ import annotations

from tone_keeper.fingerprint.distance import style_distance
from tone_keeper.fingerprint.features import corpus_func_scheme
from tone_keeper.fingerprint.profile import Profile, build_profile


class CorpusFuncV1:
    name = "v1_corpus_func"

    def __init__(self, min_count: int = 20):
        self.min_count = min_count
        self._scheme = None

    def build(self, train_texts: list[str]) -> Profile:
        self._scheme = corpus_func_scheme(
            train_texts, min_count=self.min_count, with_names=False
        )
        return build_profile(
            train_texts,
            extract=self._scheme.extract,
            names=self._scheme.names(),
        )

    def distance(self, text: str, profile: Profile) -> float:
        if self._scheme is None:
            raise RuntimeError("build() must run before distance()")
        return style_distance(text, profile, extract=self._scheme.extract)
