from __future__ import annotations

from tone_keeper.config import FilterConfig
from tone_keeper.filter.decide import Decision, decide


def filter_pair(
    a: str,
    u: str,
    cfg: FilterConfig | None = None,
    embed_score: float = 1.0,
) -> tuple[bool, str]:
    d = _run(u, a, cfg, embed_score)
    return d.ok, d.reason


def decide_pair(
    u: str,
    a: str,
    cfg: FilterConfig | None = None,
    embed_score: float = 1.0,
) -> Decision:
    return _run(u, a, cfg, embed_score)


def _run(u: str, a: str, cfg: FilterConfig | None, embed_score: float) -> Decision:
    if cfg is None:
        return decide(u, a, embed_score=embed_score)
    return decide(
        u,
        a,
        embed_score=embed_score,
        length_min=cfg.tau_len,
        expand_max=cfg.max_expand,
    )
