from __future__ import annotations

import re

from tone_keeper.text.length import cjk_len

_BLANK = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[。！？!?；;])")


def split_units(
    text: str,
    min_chars: int = 20,
    max_chars: int = 200,
) -> list[str]:
    """Split into tweet-sized units. Never merge adjacent pieces."""
    units: list[str] = []
    for block in _BLANK.split(text.strip()):
        block = block.strip()
        if not block:
            continue
        for piece in _split_block(block):
            for chunk in _hard_cut(piece, max_chars):
                if cjk_len(chunk) >= min_chars:
                    units.append(chunk)
    return units


def _split_block(block: str) -> list[str]:
    pieces: list[str] = []
    for sentence in _SENTENCE.split(block):
        for line in sentence.split("\n"):
            stripped = line.strip()
            if stripped:
                pieces.append(stripped)
    return pieces


def _hard_cut(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if cjk_len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    buf: list[str] = []
    count = 0
    for char in text:
        buf.append(char)
        if not char.isspace():
            count += 1
        if count >= max_chars:
            chunks.append("".join(buf).strip())
            buf = []
            count = 0
    tail = "".join(buf).strip()
    if tail:
        chunks.append(tail)
    return [chunk for chunk in chunks if chunk]
