from __future__ import annotations

import json
from pathlib import Path

from tone_keeper.config import UnitConfig
from tone_keeper.data.schema import Unit
from tone_keeper.text.segment import split_units
from tone_keeper.tools.md_split import split_markdown


def ingest_path(path: Path, cfg: UnitConfig) -> list[Unit]:
    path = path.resolve()
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".txt", ".jsonl", ".md"}
        )
    units: list[Unit] = []
    for file in files:
        units.extend(_ingest_file(file, cfg))
    return units


def _ingest_file(path: Path, cfg: UnitConfig) -> list[Unit]:
    source_id = path.stem
    texts: list[str] = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text") or row.get("u") or ""
            if text:
                texts.append(str(text))
    elif path.suffix.lower() == ".md":
        texts.append(path.read_text(encoding="utf-8"))
    else:
        texts.append(path.read_text(encoding="utf-8"))

    splitter = split_markdown if path.suffix.lower() == ".md" else split_units
    units: list[Unit] = []
    index = 0
    for text in texts:
        for piece in splitter(text, min_chars=cfg.min_chars, max_chars=cfg.max_chars):
            units.append(
                Unit(
                    id=f"{source_id}-{index:04d}",
                    source_id=source_id,
                    text=piece,
                )
            )
            index += 1
    return units
