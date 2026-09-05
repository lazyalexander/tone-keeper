"""Append-only JSONL events for before/after comparison. Not the training loss."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tone_keeper.paths import logs_dir


def record(kind: str, payload: dict[str, Any], path: Path | None = None) -> Path:
    log_path = path or (logs_dir() / "events.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
        "kind": kind,
        **payload,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return log_path
