"""v1 kit: punctuation plus function chars frequent in this train U, d_cos to mean.

Name chars are excluded from F (content leak). Swap the whole package, not parts.
"""

from __future__ import annotations

from rulers.core import FeatureScheme, Profile, build_profile, distance_cosine, scheme_from_profile

PUNCT_PLUS = list("，。！？…—、：；,.!?~“”「」『』")
FUNC_CANDIDATES = list("的了着地是不在之而以与所将已把被就还和都又也却则若因既乃亦复其于吗呢吧啊")
NAME_CHARS: frozenset[str] = frozenset()


def corpus_func_scheme(
    texts: list[str],
    min_count: int = 20,
    exclude: set[str] | None = None,
    name_chars: set[str] | None = None,
    with_names: bool = False,
) -> FeatureScheme:
    names = set(name_chars) if name_chars is not None else set(NAME_CHARS)
    skip = set() if with_names else set(exclude if exclude is not None else names)
    blob = "".join("".join(ch for ch in text if not ch.isspace()) for text in texts)
    func = tuple(ch for ch in FUNC_CANDIDATES if blob.count(ch) >= min_count and ch not in skip)
    extra = tuple(ch for ch in sorted(names) if with_names and blob.count(ch) >= min_count)
    label = "corpus_func_v1_with_names" if with_names else "corpus_func_v1"
    return FeatureScheme(label, tuple(PUNCT_PLUS), func + extra)


class CorpusFuncV1:
    name = "v1_corpus_func"

    def __init__(self, min_count: int = 20):
        self.min_count = min_count
        self._scheme = None

    def build(self, train_texts: list[str]) -> Profile:
        self._scheme = corpus_func_scheme(train_texts, min_count=self.min_count, with_names=False)
        return build_profile(
            train_texts,
            extract=self._scheme.extract,
            names=self._scheme.names(),
            ruler=self.name,
        )

    def distance(self, text: str, profile: Profile) -> float:
        scheme = self._scheme or scheme_from_profile(profile)
        return distance_cosine(text, profile, scheme.extract)


Ruler = CorpusFuncV1
