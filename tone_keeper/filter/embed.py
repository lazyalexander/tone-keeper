"""Content identity via embedding cosine. Copy detection stays char n-gram.

Model: Qwen3-Embedding-0.6B (MLX 8bit). Instruction-aware.
Both sides get the same instruction (symmetric paraphrase, not retrieval).
This score does not enter the training gradient.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from tone_keeper.data.schema import Pair

INSTRUCTION = (
    "Represent the propositional content of this text for paraphrase detection. "
    "Ignore register, tone, punctuation, and oral vs formal wording."
)

DEFAULT_MODEL = "mlx-community/Qwen3-Embedding-0.6B-8bit"
DEFAULT_BATCH = 16

_cache: MlxEmbedder | None = None
_cache_key: tuple[str, int] | None = None


def format_input(text: str) -> str:
    return f"Instruct: {INSTRUCTION}\nQuery:{text}"


class Embedder(Protocol):
    def cosine_many(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


class MlxEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = DEFAULT_BATCH):
        from mlx_embeddings import generate, load

        self._generate = generate
        self.model, self.tokenizer = load(model)
        self.batch_size = batch_size

    def cosine_many(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        import mlx.core as mx

        scores = [0.0] * len(pairs)
        pending: list[int] = []
        for i, (left, right) in enumerate(pairs):
            if (left or "").strip() and (right or "").strip():
                pending.append(i)
            else:
                scores[i] = 0.0

        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start : start + self.batch_size]
            texts: list[str] = []
            for i in chunk:
                left, right = pairs[i]
                texts.append(format_input(left))
                texts.append(format_input(right))
            out = self._generate(self.model, self.tokenizer, texts=texts, max_length=512)
            embeds = out.text_embeds
            mx.eval(embeds)
            for j, i in enumerate(chunk):
                scores[i] = float((embeds[2 * j] @ embeds[2 * j + 1]).item())
        return scores


def default_embedder(model: str = DEFAULT_MODEL, batch_size: int = DEFAULT_BATCH) -> MlxEmbedder:
    global _cache, _cache_key
    key = (model, batch_size)
    if _cache is None or _cache_key != key:
        _cache = MlxEmbedder(model=model, batch_size=batch_size)
        _cache_key = key
    return _cache


def score_pairs(pairs: list[Pair], embedder: Embedder) -> list[Pair]:
    raw = [(p.get("u") or "", p.get("a") or "") for p in pairs]
    scores = embedder.cosine_many(raw)
    out: list[Pair] = []
    for pair, score in zip(pairs, scores, strict=True):
        row = dict(pair)
        blob = dict(row.get("scores") or {})
        blob["embed"] = score
        row["scores"] = blob
        row["embed_score"] = score
        out.append(row)  # type: ignore[arg-type]
    return out


def shuffled_negatives(
    pairs: list[Pair],
    embedder: Embedder,
    shift: int | None = None,
) -> list[float]:
    usable = [(p.get("u") or "", p.get("a") or "") for p in pairs if (p.get("u") or "").strip() and (p.get("a") or "").strip()]
    if len(usable) < 2:
        return []
    n = len(usable)
    delta = shift if shift is not None else max(1, n // 2)
    crossed = [(usable[i][0], usable[(i + delta) % n][1]) for i in range(n)]
    return embedder.cosine_many(crossed)


def _embed(pair: Pair) -> float | None:
    scores = pair.get("scores")
    if isinstance(scores, dict) and scores.get("embed") is not None:
        return float(scores["embed"])
    if pair.get("embed_score") is not None:
        return float(pair["embed_score"])
    return None


def _char(pair: Pair) -> float | None:
    scores = pair.get("scores")
    if isinstance(scores, dict) and scores.get("char") is not None:
        return float(scores["char"])
    if pair.get("char_score") is not None:
        return float(pair["char_score"])
    if pair.get("content_score") is not None:
        return float(pair["content_score"])
    return None


def summarize(pairs: list[Pair], negatives: list[float] | None = None) -> dict[str, float | int]:
    embeds = [v for p in pairs if (v := _embed(p)) is not None]
    chars = [v for p in pairs if (v := _char(p)) is not None]
    content_fail_high = sum(
        1
        for p in pairs
        if p.get("reason") == "content" and float(_embed(p) or 0) >= 0.85
    )
    ok_low = sum(1 for p in pairs if p.get("ok") and float(_embed(p) or 0) < 0.70)
    payload: dict[str, float | int] = {
        "n": len(pairs),
        "embed_mean": _mean(embeds),
        "embed_p50": _p50(embeds),
        "char_mean": _mean(chars),
        "char_p50": _p50(chars),
        "content_fail_embed_ge_0.85": content_fail_high,
        "ok_embed_lt_0.70": ok_low,
    }
    if negatives:
        payload["neg_mean"] = _mean(negatives)
        payload["neg_p50"] = _p50(negatives)
    return payload


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _p50(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    return ordered[len(ordered) // 2]
