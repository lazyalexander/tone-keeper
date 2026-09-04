from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Post:
    source: str
    user_id: str
    post_id: str
    kind: str
    text: str
    created_at: datetime | None
    url: str | None
    raw: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        created = row["created_at"]
        if isinstance(created, datetime) and created.tzinfo is not None:
            row["created_at"] = created.astimezone(timezone.utc).replace(tzinfo=None)
        return row
