from __future__ import annotations

from pathlib import Path
from typing import Any

from tone_keeper.config import Config, student_prompt
from tone_keeper.data.jsonl import write_jsonl
from tone_keeper.data.schema import Pair
from tone_keeper.filter.decide import Decision, decide


def pair_to_example(pair: Pair, cfg: Config) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": student_prompt(cfg.prompts.student, pair["a"]),
            },
            {"role": "assistant", "content": pair["u"]},
        ]
    }


def apply_decision(pair: Pair, decision: Decision) -> Pair:
    row = dict(pair)
    row["ok"] = decision.ok
    row["reason"] = decision.reason
    row["scores"] = decision.scores()
    row["embed_score"] = decision.embed
    row["filter_log"] = decision.log
    return row  # type: ignore[return-value]


def embed_of(pair: Pair) -> float | None:
    scores = pair.get("scores")
    if isinstance(scores, dict) and scores.get("embed") is not None:
        return float(scores["embed"])
    if pair.get("embed_score") is not None:
        return float(pair["embed_score"])
    return None


def refilter_pairs(pairs: list[Pair], embedder: Any = None) -> list[Pair]:
    from tone_keeper.filter.embed import default_embedder, score_pairs

    if any(embed_of(p) is None for p in pairs):
        pairs = score_pairs(pairs, embedder or default_embedder())
    out: list[Pair] = []
    for pair in pairs:
        d = decide(
            pair.get("u") or "",
            pair.get("a") or "",
            embed_score=float(embed_of(pair) or 0.0),
        )
        out.append(apply_decision(pair, d))
    return out


def export_sft(
    train_pairs: list[Pair],
    valid_pairs: list[Pair],
    out_dir: Path,
    cfg: Config,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    write_jsonl(train_path, (pair_to_example(p, cfg) for p in train_pairs if p["ok"]))
    write_jsonl(valid_path, (pair_to_example(p, cfg) for p in valid_pairs if p["ok"]))
    return train_path, valid_path
