"""Evaluate a ruler; verdicts live in the acceptance JSON, not here.

Measurements only:

    tau = Q_{q}({ d(u, F) : u in U_train })
    T1  = |{ u in U_held : d(u, F) <= tau }| / |U_held|
    T2  = |{ x in contrast : d(x, F) <= tau }| / |contrast|
    T3  = |{ (s1,s2) : embed(s1,s2) < content_tau }| / |pairs|

q and content_tau come from rulers/{name}.accept.json.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from rulers.accept import load_criteria
from rulers.protocol import Ruler


class Embedder(Protocol):
    def cosine_many(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    return float(np.quantile(np.asarray(xs, dtype=np.float64), q))


def _rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def eval_ruler(
    ruler: Ruler,
    train_u: list[str],
    held_u: list[str],
    held_pairs: list[tuple[str, str]],
    contrast: list[str],
    confusable: list[tuple[str, str]],
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    criteria = load_criteria(ruler.name)
    q = float(criteria["train_quantile"])
    content_tau = float(criteria["content_tau"])
    profile = ruler.build(train_u)
    d_train = [ruler.distance(u, profile) for u in train_u]
    tau = _quantile(d_train, q)
    d_held = [ruler.distance(u, profile) for u in held_u]
    d_t2 = [ruler.distance(x, profile) for x in contrast]
    t1 = _rate([d <= tau for d in d_held])
    t2 = _rate([d <= tau for d in d_t2])
    destroy_rate = _rate(
        [
            ruler.distance(a, profile) > ruler.distance(u, profile)
            for u, a in held_pairs
        ]
    )
    t3: dict[str, Any] = {"skipped": True}
    if embedder is not None and confusable:
        scores = embedder.cosine_many(confusable)
        fail = [s < content_tau for s in scores]
        t3 = {
            "skipped": False,
            "n": len(confusable),
            "content_fail_rate": round(_rate(fail), 4),
            "mean_embed": round(float(np.mean(scores)), 4),
            "pass_content_gate": round(1.0 - _rate(fail), 4),
        }
    return {
        "ruler": ruler.name,
        "n_train": len(train_u),
        "n_held": len(held_u),
        "n_contrast": len(contrast),
        "tau": round(tau, 4),
        "train_quantile": q,
        "t1_held_pass": round(t1, 4),
        "t2_contrast_pass": round(t2, 4),
        "destroy_rate_held": round(destroy_rate, 4),
        "mean_d_train": round(float(np.mean(d_train)) if d_train else 0.0, 4),
        "mean_d_held": round(float(np.mean(d_held)) if d_held else 0.0, 4),
        "mean_d_contrast": round(float(np.mean(d_t2)) if d_t2 else 0.0, 4),
        "t3": t3,
    }
