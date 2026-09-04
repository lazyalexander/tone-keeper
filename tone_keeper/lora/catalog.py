"""Generation catalog. One round = one user, enforced upstream. No source, no user_id.

pairs holds every unit in the round, including rejects. reason != 'ok' is the reject set.
Policy (thresholds, models) lives on gen_runs.policy, not as pair columns.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from tone_keeper.data.schema import Pair

_GEN_RUNS = """
CREATE TABLE IF NOT EXISTS gen_runs (
    round INTEGER PRIMARY KEY,
    git_sha TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    policy JSON NOT NULL
)
"""

_PAIRS = """
CREATE TABLE IF NOT EXISTS pairs (
    round INTEGER NOT NULL,
    unit_id TEXT NOT NULL,
    split TEXT NOT NULL,
    u TEXT NOT NULL,
    a TEXT NOT NULL,
    reason TEXT NOT NULL,
    scores JSON NOT NULL,
    filter_log TEXT NOT NULL,
    ok BOOLEAN GENERATED ALWAYS AS (reason = 'ok') VIRTUAL,
    PRIMARY KEY (round, unit_id),
    FOREIGN KEY (round) REFERENCES gen_runs(round)
)
"""

_COMMENTS = [
    "COMMENT ON TABLE gen_runs IS 'One generation round. One user per round, enforced upstream. Policy is a JSON document.'",
    "COMMENT ON TABLE pairs IS 'All pairs in the round, including rejects. Rejects are reason != ''ok''. No source_id, no user_id.'",
    "COMMENT ON COLUMN pairs.reason IS 'ok | empty | language | expanded | length | content | error:*'",
    "COMMENT ON COLUMN pairs.scores IS 'Per-pair observations. Keys may change with the filter; policy for this round is gen_runs.policy.'",
    "COMMENT ON COLUMN pairs.ok IS 'Generated: reason = ''ok''. Not stored.'",
    "COMMENT ON COLUMN gen_runs.policy IS 'Filter thresholds and model ids for this round.'",
]

_PAIRS_COLS = {
    "round",
    "unit_id",
    "split",
    "u",
    "a",
    "reason",
    "scores",
    "filter_log",
    "ok",
}
_RUNS_COLS = {"round", "git_sha", "created_at", "policy"}


def catalog_path(root: Path | None = None) -> Path:
    from tone_keeper.paths import project_root

    return (root or project_root()) / "lora" / "catalog.duckdb"


def git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "none"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "none"


def default_policy(*, teacher_model: str | None = None, embed_model: str | None = None) -> dict[str, Any]:
    from tone_keeper.filter.decide import (
        EMBED_COSINE_MIN,
        EXPAND_MAX,
        LANGUAGE_CJK_GAP_MAX,
        LENGTH_RATIO_MIN,
    )
    from tone_keeper.filter.embed import INSTRUCTION

    return {
        "embed_min": EMBED_COSINE_MIN,
        "length_min": LENGTH_RATIO_MIN,
        "expand_max": EXPAND_MAX,
        "language_gap_max": LANGUAGE_CJK_GAP_MAX,
        "teacher_model": teacher_model,
        "embed_model": embed_model,
        "embed_instruction": INSTRUCTION,
    }


def write_pairs(
    pairs: list[Pair],
    round: int,
    path: Path | None = None,
    policy: dict[str, Any] | None = None,
    teacher_model: str | None = None,
    embed_model: str | None = None,
) -> Path:
    payload = policy if policy is not None else default_policy(
        teacher_model=teacher_model,
        embed_model=embed_model,
    )
    db = path or catalog_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    con, tmp = _connect_write(db)
    try:
        _ensure_schema(con)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        con.execute("DELETE FROM pairs WHERE round = ?", [round])
        con.execute("DELETE FROM gen_runs WHERE round = ?", [round])
        con.execute(
            "INSERT INTO gen_runs (round, git_sha, created_at, policy) VALUES (?, ?, ?, ?::JSON)",
            [round, git_sha(), now, json.dumps(payload, ensure_ascii=False)],
        )
        con.executemany(
            """
            INSERT INTO pairs (
                round, unit_id, split, u, a, reason, scores, filter_log
            ) VALUES (?, ?, ?, ?, ?, ?, ?::JSON, ?)
            """,
            [
                (
                    round,
                    p["id"],
                    p["split"],
                    p["u"],
                    p["a"],
                    p["reason"],
                    json.dumps(_scores_from_pair(p), ensure_ascii=False),
                    p.get("filter_log") or p["reason"],
                )
                for p in pairs
            ],
        )
    finally:
        con.close()
    if tmp is not None:
        tmp.replace(db)
        tmp.with_name(tmp.name + ".wal").unlink(missing_ok=True)
    return db


def _scores_from_pair(pair: Pair) -> dict[str, Any]:
    existing = pair.get("scores")
    if isinstance(existing, str):
        existing = json.loads(existing)
    if isinstance(existing, dict) and existing:
        return existing
    return {
        "embed": pair.get("embed_score"),
        "char": pair.get("char_score", pair.get("content_score")),
        "length": pair.get("length_ratio"),
        "expand": pair.get("expand_ratio"),
        "cjk_gap": pair.get("cjk_gap"),
    }


def _connect_write(db: Path) -> tuple[duckdb.DuckDBPyConnection, Path | None]:
    try:
        return duckdb.connect(str(db)), None
    except duckdb.IOException:
        tmp = db.with_name(db.name + ".tmp")
        if tmp.exists():
            tmp.unlink()
        if db.exists():
            shutil.copy2(db, tmp)
        return duckdb.connect(str(tmp)), tmp


def _table_cols(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}


def _ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
    stale = False
    if "pairs" in tables and _table_cols(con, "pairs") != _PAIRS_COLS:
        stale = True
    if "gen_runs" in tables and _table_cols(con, "gen_runs") != _RUNS_COLS:
        stale = True
    if stale:
        con.execute("DROP VIEW IF EXISTS rejected")
        con.execute("DROP VIEW IF EXISTS filter_summary")
        con.execute("DROP TABLE IF EXISTS pairs")
        con.execute("DROP TABLE IF EXISTS gen_runs")
    con.execute(_GEN_RUNS)
    con.execute(_PAIRS)
    for stmt in _COMMENTS:
        con.execute(stmt)


def peek(path: Path | None = None, sql: str | None = None) -> str:
    db = path or catalog_path()
    try:
        con = duckdb.connect(str(db), read_only=True)
    except duckdb.IOException as exc:
        raise RuntimeError(f"catalog locked (close Harlequin): {db}") from exc
    try:
        if sql:
            rel = con.execute(sql)
            cols = [d[0] for d in rel.description]
            rows = rel.fetchall()
            lines = ["\t".join(cols)]
            for row in rows[:50]:
                lines.append("\t".join("" if v is None else str(v) for v in row))
            if len(rows) > 50:
                lines.append(f"... {len(rows)} rows")
            return "\n".join(lines)
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        parts = [f"db={db}", f"tables={tables}"]
        if "gen_runs" in tables:
            runs = con.execute(
                "SELECT round, git_sha, created_at, policy FROM gen_runs ORDER BY round"
            ).fetchall()
            parts.append("runs=" + json.dumps(runs, default=str, ensure_ascii=False))
        if "pairs" in tables:
            n = con.execute("SELECT count(*) FROM pairs").fetchone()[0]
            ok = con.execute("SELECT count(*) FROM pairs WHERE ok").fetchone()[0]
            reasons = con.execute(
                """
                SELECT round, split, reason, count(*) AS n,
                       round(avg((scores->>'embed')::DOUBLE), 3) AS avg_embed,
                       round(avg((scores->>'char')::DOUBLE), 3) AS avg_char
                FROM pairs
                GROUP BY 1, 2, 3
                ORDER BY 1, 2, 4 DESC
                """
            ).fetchall()
            parts.append(f"pairs={n} ok={ok}")
            parts.append("reasons=" + json.dumps(reasons, ensure_ascii=False))
        return "\n".join(parts)
    finally:
        con.close()
