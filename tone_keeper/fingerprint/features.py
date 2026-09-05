"""Production (v0) feature scheme. Other kits live under rulers/<name>/."""

from rulers.core import FeatureScheme
from rulers.v0_punct_func import FUNCTION, PUNCT, SCHEME as PUNCT_FUNC_V1


def extract_features(text: str, vocab: list[str] | None = None):
    del vocab
    return PUNCT_FUNC_V1.extract(text)


def feature_names(vocab: list[str] | None = None) -> list[str]:
    del vocab
    return PUNCT_FUNC_V1.names()


__all__ = [
    "FUNCTION",
    "PUNCT",
    "PUNCT_FUNC_V1",
    "FeatureScheme",
    "extract_features",
    "feature_names",
]
