import json
from pathlib import Path

import duckdb

from tone_keeper.data.schema import Pair
from tone_keeper.filter.decide import decide
from tone_keeper.lora.catalog import write_pairs
from tone_keeper.lora.export import apply_decision


def _pair(i: str, split: str = "train", embed: float = 0.95) -> Pair:
    u = "今天早上上班路上又堵成这样真的让人无语了啊"
    a = "今天早上上班路上又堵成这样确实让人无语了啊。"
    base = Pair(
        id=i,
        source_id="should-not-be-stored",
        u=u,
        a=a,
        split=split,  # type: ignore[arg-type]
        ok=True,
        reason="ok",
    )
    return apply_decision(base, decide(u, a, embed_score=embed))


def _scores(value) -> dict:
    if isinstance(value, dict):
        return value
    return json.loads(value)


def test_schema_two_tables_no_reject_table(tmp_path: Path):
    db = write_pairs(
        [_pair("u-1")],
        round=1,
        path=tmp_path / "c.duckdb",
        teacher_model="t",
        embed_model="e",
    )
    con = duckdb.connect(str(db), read_only=True)
    try:
        pair_cols = {r[1] for r in con.execute("PRAGMA table_info('pairs')").fetchall()}
        run_cols = {r[1] for r in con.execute("PRAGMA table_info('gen_runs')").fetchall()}
        tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
        policy = _scores(con.execute("SELECT policy FROM gen_runs").fetchone()[0])
        ok = con.execute("SELECT ok, reason FROM pairs").fetchone()
    finally:
        con.close()
    assert tables == {"gen_runs", "pairs"}
    assert pair_cols == {
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
    assert "source_id" not in pair_cols
    assert "user_id" not in pair_cols
    assert "embed_score" not in pair_cols
    assert "char_score" not in pair_cols
    assert run_cols == {"round", "git_sha", "created_at", "policy"}
    assert policy["embed_min"] == 0.8
    assert policy["teacher_model"] == "t"
    assert policy["embed_model"] == "e"
    assert ok == (True, "ok")


def test_ok_is_generated_from_reason(tmp_path: Path):
    db = write_pairs([_pair("fail", embed=0.1)], round=1, path=tmp_path / "c.duckdb")
    con = duckdb.connect(str(db), read_only=True)
    try:
        reason, ok = con.execute("SELECT reason, ok FROM pairs").fetchone()
        n_reject = con.execute("SELECT count(*) FROM pairs WHERE NOT ok").fetchone()[0]
    finally:
        con.close()
    assert reason == "content"
    assert ok is False
    assert n_reject == 1


def test_overwrite_round_keeps_other_rounds(tmp_path: Path):
    db = tmp_path / "c.duckdb"
    write_pairs([_pair("a")], round=1, path=db)
    write_pairs([_pair("b"), _pair("c")], round=2, path=db)
    write_pairs([_pair("d")], round=1, path=db)
    con = duckdb.connect(str(db), read_only=True)
    try:
        r1 = con.execute("SELECT unit_id FROM pairs WHERE round = 1").fetchall()
        r2 = con.execute("SELECT count(*) FROM pairs WHERE round = 2").fetchone()[0]
        log, scores = con.execute(
            "SELECT filter_log, scores FROM pairs WHERE unit_id = 'd'"
        ).fetchone()
    finally:
        con.close()
    assert r1 == [("d",)]
    assert r2 == 2
    assert "embed=" in log
    assert _scores(scores)["embed"] == 0.95
    assert _scores(scores)["char"] > 0.5


def test_migrates_old_schema(tmp_path: Path):
    db = tmp_path / "c.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE pairs (
            unit_id TEXT PRIMARY KEY,
            source_id TEXT,
            split TEXT,
            u TEXT,
            a TEXT,
            ok BOOLEAN,
            reason TEXT
        )
        """
    )
    con.execute("INSERT INTO pairs VALUES ('old', 'src', 'train', 'u', 'a', true, 'ok')")
    con.close()
    write_pairs([_pair("new")], round=1, path=db, teacher_model="t", embed_model="e")
    con = duckdb.connect(str(db), read_only=True)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info('pairs')").fetchall()}
        ids = [r[0] for r in con.execute("SELECT unit_id FROM pairs").fetchall()]
    finally:
        con.close()
    assert "source_id" not in cols
    assert "scores" in cols
    assert ids == ["new"]


def test_reads_legacy_flat_scores(tmp_path: Path):
    pair = _pair("legacy")
    del pair["scores"]
    pair["embed_score"] = 0.91
    pair["char_score"] = 0.42
    pair["length_ratio"] = 0.8
    db = write_pairs([pair], round=1, path=tmp_path / "c.duckdb")
    con = duckdb.connect(str(db), read_only=True)
    try:
        scores = _scores(con.execute("SELECT scores FROM pairs").fetchone()[0])
    finally:
        con.close()
    assert scores["embed"] == 0.91
    assert scores["char"] == 0.42
    assert scores["length"] == 0.8
