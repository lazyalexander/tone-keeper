from __future__ import annotations

import numpy as np

from tone_keeper.text.length import cjk_len

PUNCT = list("，。！？…—、：；,.!?~")
FUNCTION = list("的了吗呢吧啊把被就还是不")
def feature_names(vocab: list[str] | None = None) -> list[str]:
    del vocab
    return [f"punct:{ch}" for ch in PUNCT] + [f"func:{ch}" for ch in FUNCTION]


def extract_features(text: str, vocab: list[str] | None = None) -> np.ndarray:
    del vocab
    length = max(cjk_len(text), 1)
    compact = "".join(ch for ch in text if not ch.isspace())
    punct = np.array([compact.count(ch) / length for ch in PUNCT], dtype=np.float64)
    func = np.array([compact.count(ch) / length for ch in FUNCTION], dtype=np.float64)
    return np.concatenate([punct, func])
