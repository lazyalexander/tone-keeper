from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tone_keeper.text.length import cjk_len

PUNCT = list("，。！？…—、：；,.!?~")
FUNCTION = list("的了吗呢吧啊把被就还是不")
PUNCT_PLUS = list("，。！？…—、：；,.!?~“”「」『』")
FUNC_CANDIDATES = list("的了着地是不在之而以与所将已把被就还和都又也却则若因既乃亦复其于吗呢吧啊")
# Optional per-corpus excludes. Keep empty in the public tree.
NAME_CHARS: frozenset[str] = frozenset()


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


PUNCT_FUNC_V1 = FeatureScheme("punct_func_v1", tuple(PUNCT), tuple(FUNCTION))


def feature_names(vocab: list[str] | None = None) -> list[str]:
    del vocab
    return PUNCT_FUNC_V1.names()


def extract_features(text: str, vocab: list[str] | None = None) -> np.ndarray:
    del vocab
    return PUNCT_FUNC_V1.extract(text)


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
    name = "corpus_func_v1_with_names" if with_names else "corpus_func_v1"
    return FeatureScheme(name, tuple(PUNCT_PLUS), func + extra)
