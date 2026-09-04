from __future__ import annotations


def cjk_len(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def cjk_ratio(text: str) -> float:
    length = cjk_len(text)
    if length == 0:
        return 0.0
    han = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return han / length
