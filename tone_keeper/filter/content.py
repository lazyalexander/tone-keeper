from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tone_keeper.text.length import cjk_len


def char_cosine(left: str, right: str) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
    matrix = vectorizer.fit_transform([left, right])
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


def length_ratio(left: str, right: str) -> float:
    a = cjk_len(left)
    b = cjk_len(right)
    if a == 0 or b == 0:
        return 0.0
    return min(a, b) / max(a, b)


def expand_ratio(a: str, u: str) -> float:
    u_len = cjk_len(u)
    if u_len == 0:
        return float("inf")
    return cjk_len(a) / u_len
