from pathlib import Path

from rulers.v0_punct_func import PUNCT_FUNC_V1
from rulers.v1_corpus_func import corpus_func_scheme
from tone_keeper.observe import record


def test_observe_appends_jsonl(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    record("t", {"x": 1}, path=path)
    record("t", {"x": 2}, path=path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"kind": "t"' in lines[0]


def test_corpus_scheme_drops_names():
    names = {"甲", "乙"}
    texts = ["甲乙的了着地是不在之而其于所以将已经过。" * 3]
    clean = corpus_func_scheme(texts, min_count=1, with_names=False, name_chars=names)
    leaked = corpus_func_scheme(texts, min_count=1, with_names=True, name_chars=names)
    assert "甲" not in clean.func
    assert "甲" in leaked.func or "乙" in leaked.func
    assert PUNCT_FUNC_V1.name == "punct_func_v1"
