from __future__ import annotations

from dataclasses import dataclass

from tone_keeper.config import FilterConfig
from tone_keeper.filter.content import char_cosine
from tone_keeper.filter.embed import Embedder
from tone_keeper.fingerprint.distance import style_distance
from tone_keeper.fingerprint.profile import Profile


@dataclass(frozen=True)
class RewriteResult:
    text: str | None
    rejected: bool
    n_pass: int
    reasons: tuple[str, ...]


def style_relative_pass(candidate: str, source: str, profile: Profile) -> bool:
    return style_distance(candidate, profile) < style_distance(source, profile)


def select_rewrite(
    candidates: list[str],
    source: str,
    profile: Profile,
    cfg: FilterConfig,
    embedder: Embedder,
) -> RewriteResult:
    tagged: list[tuple[str, str, float | None]] = []
    pending: list[str] = []
    for candidate in candidates:
        if not candidate.strip():
            tagged.append(("empty", candidate, None))
        elif char_cosine(candidate, source) >= cfg.tau_copy:
            tagged.append(("copy", candidate, None))
        else:
            tagged.append(("need", candidate, None))
            pending.append(candidate)

    scores = embedder.cosine_many([(c, source) for c in pending]) if pending else []
    score_iter = iter(scores)
    resolved: list[tuple[str, str, float | None]] = []
    for kind, candidate, _ in tagged:
        if kind == "need":
            resolved.append((kind, candidate, next(score_iter)))
        else:
            resolved.append((kind, candidate, None))

    survivors: list[tuple[float, str]] = []
    reasons: list[str] = []
    for kind, candidate, score in resolved:
        if kind == "empty":
            reasons.append("empty")
            continue
        if kind == "copy":
            reasons.append("copy")
            continue
        if score is None or score < cfg.tau_pair:
            reasons.append("content")
            continue
        if not style_relative_pass(candidate, source, profile):
            reasons.append("style")
            continue
        survivors.append((style_distance(candidate, profile), candidate))
    if not survivors:
        return RewriteResult(text=None, rejected=True, n_pass=0, reasons=tuple(reasons))
    survivors.sort(key=lambda item: item[0])
    return RewriteResult(
        text=survivors[0][1],
        rejected=False,
        n_pass=len(survivors),
        reasons=tuple(reasons),
    )
