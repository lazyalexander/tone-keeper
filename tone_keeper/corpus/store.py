from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import duckdb

from tone_keeper.corpus.models import Post

_SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    source TEXT NOT NULL,
    user_id TEXT NOT NULL,
    post_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TIMESTAMP,
    text TEXT NOT NULL,
    url TEXT,
    raw JSON,
    fetched_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (source, post_id)
);
"""


class CorpusStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(path))
        self._con.execute(_SCHEMA)

    def close(self) -> None:
        self._con.close()

    def upsert(self, posts: Iterable[Post]) -> int:
        rows = [post.to_row() for post in posts if post.text.strip()]
        if not rows:
            return 0
        self._con.executemany(
            """
            INSERT INTO posts (source, user_id, post_id, kind, created_at, text, url, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source, post_id) DO UPDATE SET
                user_id = excluded.user_id,
                kind = excluded.kind,
                created_at = excluded.created_at,
                text = excluded.text,
                url = excluded.url,
                raw = excluded.raw,
                fetched_at = now()
            """,
            [
                (
                    r["source"],
                    r["user_id"],
                    r["post_id"],
                    r["kind"],
                    r["created_at"],
                    r["text"],
                    r["url"],
                    json.dumps(r["raw"], ensure_ascii=False),
                )
                for r in rows
            ],
        )
        return len(rows)

    def export_jsonl(self, path: Path, source: str | None = None, user_id: str | None = None) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        clauses = ["length(trim(text)) > 0"]
        params: list[str] = []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        where = " AND ".join(clauses)
        rel = self._con.execute(
            f"SELECT text FROM posts WHERE {where} ORDER BY created_at NULLS LAST, post_id",
            params,
        )
        n = 0
        with path.open("w", encoding="utf-8") as handle:
            for (text,) in rel.fetchall():
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n += 1
        return n

    def count(self, source: str | None = None) -> int:
        if source:
            row = self._con.execute("SELECT count(*) FROM posts WHERE source = ?", [source]).fetchone()
        else:
            row = self._con.execute("SELECT count(*) FROM posts").fetchone()
        return int(row[0]) if row else 0
