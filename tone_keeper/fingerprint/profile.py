"""Re-export profile types from rulers.core."""

from rulers.core import Profile, profile_from_dict
from rulers.core import build_profile as _build_profile


def build_profile(texts, ngram_k: int = 512, extract=None, names=None, ruler: str = "v0_punct_func"):
    del ngram_k
    if extract is None and names is None:
        from rulers.v0_punct_func import SCHEME

        extract = SCHEME.extract
        names = SCHEME.names()
    elif extract is None or names is None:
        raise ValueError("extract and names must be passed together")
    return _build_profile(texts, extract=extract, names=names, ruler=ruler)


__all__ = ["Profile", "build_profile", "profile_from_dict"]
