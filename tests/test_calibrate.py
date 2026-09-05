from pathlib import Path

from tone_keeper.data.schema import Pair, Unit
from tone_keeper.fingerprint.calibrate import evaluate_scheme, run_calibration
from tone_keeper.fingerprint.distance import style_distance
from tone_keeper.fingerprint.features import PUNCT_FUNC_V1, corpus_func_scheme
from tone_keeper.observe import record


def test_observe_appends_jsonl(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    record("t", {"x": 1}, path=path)
    record("t", {"x": 2}, path=path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"kind": "t"' in lines[0]


def test_evaluate_scheme_destroy_on_toy():
    train = [
        "今天早上又堵了啊，真的让人无语了呢吧。",
        "晚饭就随便对付一下吧，明天再说了啊。",
        "这周的事情叠在一起了，有点想摆了。",
    ]
    held = [
        (
            "刚喝完咖啡整个人还是很困啊。",
            "根据有关规定应当依法予以处理并及时报告。",
        )
    ]
    row = evaluate_scheme(
        scheme=PUNCT_FUNC_V1,
        distance_name="cosine_raw",
        distance=lambda text, prof: style_distance(text, prof, extract=PUNCT_FUNC_V1.extract),
        train_texts=train,
        held_pairs=held,
    )
    assert row["n_held"] == 1
    assert row["n_features"] == len(PUNCT_FUNC_V1.names())
    assert 0.0 <= row["destroy_rate"] <= 1.0


def test_corpus_scheme_drops_names():
    names = {"甲", "乙"}
    texts = ["甲乙的了着地是不在之而其于所以将已经过。" * 3]
    clean = corpus_func_scheme(texts, min_count=1, with_names=False, name_chars=names)
    leaked = corpus_func_scheme(texts, min_count=1, with_names=True, name_chars=names)
    assert "甲" not in clean.func
    assert "甲" in leaked.func or "乙" in leaked.func


def test_run_calibration_writes_rows(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TONE_KEEPER_ROOT", str(tmp_path))
    units = [
        Unit(id="a-0", source_id="a", text="今天早上又堵了啊真的让人无语了呢。"),
        Unit(id="a-1", source_id="a", text="晚饭就随便对付一下吧明天再说了。"),
        Unit(id="a-2", source_id="a", text="根据有关规定应当依法予以处理。"),
    ]
    pairs = [
        Pair(
            id="a-2",
            source_id="a",
            u=units[2]["text"],
            a="现将有关情况函告如下请予以配合落实。",
            split="held",
            ok=True,
            reason="ok",
        )
    ]
    summary = run_calibration(units, pairs, train_ids={"a-0", "a-1"})
    assert len(summary["rows"]) == 6
    assert "recommend_hard_gate" in summary
