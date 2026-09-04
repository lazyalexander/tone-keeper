"""Split a long Markdown document into tweet-sized prose units.

Drops structure (headings, fences, rules, frontmatter). Does not rewrite prose.
Never merges adjacent paragraphs.
"""

from __future__ import annotations

from tone_keeper.text.length import cjk_len
from tone_keeper.text.segment import split_units


def split_markdown(
    text: str,
    min_chars: int = 20,
    max_chars: int = 200,
) -> list[str]:
    body = _strip_structure(text)
    units: list[str] = []
    for para in _paragraphs(body):
        if cjk_len(para) <= max_chars:
            if cjk_len(para) >= min_chars:
                units.append(para)
            continue
        units.extend(split_units(para, min_chars=min_chars, max_chars=max_chars))
    return units


def _strip_structure(text: str) -> str:
    lines = text.splitlines()
    i = 0
    if lines and lines[0].strip() == "---":
        i = 1
        while i < len(lines) and lines[i].strip() != "---":
            i += 1
        i += 1
    kept: list[str] = []
    in_fence = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            i += 1
            continue
        if in_fence:
            i += 1
            continue
        if stripped.startswith("#"):
            i += 1
            continue
        if stripped in {"---", "***", "___", "* * *"}:
            i += 1
            continue
        kept.append(line)
        i += 1
    return "\n".join(kept)


def _paragraphs(text: str) -> list[str]:
    paras: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if buf:
                paras.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(line)
    if buf:
        paras.append("\n".join(buf).strip())
    return [p for p in paras if p]
