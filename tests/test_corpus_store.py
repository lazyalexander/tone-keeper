from datetime import datetime, timezone
from pathlib import Path

from tone_keeper.corpus.models import Post
from tone_keeper.corpus.store import CorpusStore
from tone_keeper.corpus.text import html_to_text


def test_html_to_text_strips_tags():
    assert html_to_text("<p>今天<strong>真的</strong>无语了</p>") == "今天真的无语了"


def test_upsert_dedup_and_export(tmp_path: Path):
    store = CorpusStore(tmp_path / "c.duckdb")
    post = Post(
        source="x",
        user_id="1",
        post_id="10",
        kind="tweet",
        text="今天早上上班路上又堵成这样真的让人无语了啊",
        created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        url="https://x.com/i/web/status/10",
        raw={"id": "10"},
    )
    assert store.upsert([post]) == 1
    again = Post(
        source="x",
        user_id="1",
        post_id="10",
        kind="tweet",
        text="更新后的文本足够长可以覆盖原来的那一条",
        created_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
        url="https://x.com/i/web/status/10",
        raw={"id": "10"},
    )
    store.upsert([again])
    assert store.count("x") == 1
    out = tmp_path / "out.jsonl"
    assert store.export_jsonl(out) == 1
    assert "更新后的文本" in out.read_text(encoding="utf-8")
    store.close()
