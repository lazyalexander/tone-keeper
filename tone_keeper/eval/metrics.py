"""Pair-level checks. No extra NLP stack.

Content: character n-gram cosine (lexical overlap). BERTScore needs PyTorch; skipped.

Style: Cosine Delta on function-char / punctuation rates vs the user profile
(same family as Burrows' Delta / Cosine Delta; Chinese short text uses chars
not English MFW). pystylometry / faststylometry are English-first or heavy;
not wired as a dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from tone_keeper.filter.content import char_cosine
from tone_keeper.fingerprint.distance import style_distance
from tone_keeper.fingerprint.profile import Profile


@dataclass(frozen=True)
class PairScores:
    content: float
    style_u: float
    style_a: float
    destroyed: bool


def score_pair(u: str, a: str, profile: Profile) -> PairScores:
    style_u = style_distance(u, profile)
    style_a = style_distance(a, profile)
    return PairScores(
        content=char_cosine(a, u),
        style_u=style_u,
        style_a=style_a,
        destroyed=style_a > style_u,
    )


def summarize_destroy(pairs: list[tuple[str, str]], profile: Profile) -> dict[str, float]:
    if not pairs:
        return {"n": 0, "mean_content": 0.0, "destroy_rate": 0.0}
    scores = [score_pair(u, a, profile) for u, a in pairs]
    return {
        "n": float(len(scores)),
        "mean_content": sum(s.content for s in scores) / len(scores),
        "mean_style_u": sum(s.style_u for s in scores) / len(scores),
        "mean_style_a": sum(s.style_a for s in scores) / len(scores),
        "destroy_rate": sum(1.0 for s in scores if s.destroyed) / len(scores),
    }
