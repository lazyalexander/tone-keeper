"""Style-instrument calibration. Does not change inference gates.

F is built from train U only. Held (U, A) tests whether the ruler separates
user text from teacher text. Logs to lora/logs/events.jsonl.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from tone_keeper.data.schema import Pair, Unit
from tone_keeper.fingerprint.distance import style_distance, style_distance_zl2
from tone_keeper.fingerprint.features import (
    PUNCT_FUNC_V1,
    FeatureScheme,
    corpus_func_scheme,
)
from tone_keeper.fingerprint.profile import Profile, build_profile
from tone_keeper.observe import record

DESTROY_MIN = 0.8
Distance = Callable[[str, Profile], float]


def _rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _truncate(text: str, n: int) -> str:
    compact = "".join(ch for ch in text if not ch.isspace())
    return compact[:n]


def evaluate_scheme(
    *,
    scheme: FeatureScheme,
    distance_name: str,
    distance: Distance,
    train_texts: list[str],
    held_pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    profile = build_profile(train_texts, extract=scheme.extract, names=scheme.names())
    d_u = [distance(u, profile) for u, _ in held_pairs]
    d_a = [distance(a, profile) for _, a in held_pairs]
    destroy = [da > du for du, da in zip(d_u, d_a, strict=True)]
    n = len(held_pairs)
    shift = max(1, n // 2)
    shuffled = [
        distance(held_pairs[(i + shift) % n][1], profile) > d_u[i] for i in range(n)
    ]
    trunc = [
        distance(held_pairs[i][1], profile) > distance(_truncate(held_pairs[i][0], 30), profile)
        for i in range(n)
    ]
    destroy_rate = _rate(destroy)
    shuffled_rate = _rate(shuffled)
    return {
        "scheme": scheme.name,
        "distance": distance_name,
        "n_train": len(train_texts),
        "n_held": n,
        "n_features": len(scheme.names()),
        "func_chars": "".join(scheme.func),
        "destroy_rate": round(destroy_rate, 4),
        "shuffled_rate": round(shuffled_rate, 4),
        "trunc30_destroy_rate": round(_rate(trunc), 4),
        "mean_d_u": round(_mean(d_u), 4),
        "mean_d_a": round(_mean(d_a), 4),
        "hard_gate_ok": destroy_rate >= DESTROY_MIN and shuffled_rate <= 0.5 * destroy_rate
        if destroy_rate > 0
        else False,
    }


def run_calibration(
    units: list[Unit],
    pairs: list[Pair],
    train_ids: set[str],
) -> dict[str, Any]:
    train_texts = [u["text"] for u in units if u["id"] in train_ids]
    held_pairs = [
        (p["u"], p["a"])
        for p in pairs
        if p.get("split") == "held" and (p.get("u") or "").strip() and (p.get("a") or "").strip()
    ]
    baseline = PUNCT_FUNC_V1
    corpus = corpus_func_scheme(train_texts, with_names=False)
    leaked = corpus_func_scheme(train_texts, with_names=True)

    rows: list[dict[str, Any]] = []
    for scheme in (baseline, corpus, leaked):
        for dist_name, dist_fn in (
            ("cosine_raw", lambda text, prof, s=scheme: style_distance(text, prof, extract=s.extract)),
            ("zl2", lambda text, prof, s=scheme: style_distance_zl2(text, prof, extract=s.extract)),
        ):
            row = evaluate_scheme(
                scheme=scheme,
                distance_name=dist_name,
                distance=dist_fn,
                train_texts=train_texts,
                held_pairs=held_pairs,
            )
            record("style_calib", row)
            rows.append(row)

    leak = next(r for r in rows if r["scheme"] == leaked.name and r["distance"] == "cosine_raw")
    clean = next(r for r in rows if r["scheme"] == corpus.name and r["distance"] == "cosine_raw")
    summary = {
        "kind": "style_calib_summary",
        "name_leak_delta_destroy": round(leak["destroy_rate"] - clean["destroy_rate"], 4),
        "name_leak_delta_shuffled": round(leak["shuffled_rate"] - clean["shuffled_rate"], 4),
        "recommend_hard_gate": any(r["hard_gate_ok"] for r in rows if r["scheme"] == corpus.name),
        "rows": rows,
    }
    record("style_calib_summary", {k: v for k, v in summary.items() if k != "rows"})
    return summary
