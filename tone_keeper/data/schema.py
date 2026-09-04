from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class Unit(TypedDict):
    id: str
    source_id: str
    text: str


class Pair(TypedDict):
    id: str
    source_id: str
    u: str
    a: str
    split: Literal["train", "held"]
    ok: bool
    reason: str
    scores: NotRequired[dict]
    char_score: NotRequired[float]
    content_score: NotRequired[float]
    length_ratio: NotRequired[float]
    expand_ratio: NotRequired[float | None]
    cjk_gap: NotRequired[float]
    embed_score: NotRequired[float]
    filter_log: NotRequired[str]
