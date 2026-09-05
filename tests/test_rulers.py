from pathlib import Path

import pytest

from rulers import RULERS
from rulers.accept import accept_path, build_document, load_criteria, require_accept_file
from rulers.fixtures import T2_CONTRAST, T3_CONFUSABLE
from rulers.harness import eval_ruler
from rulers.v0_punct_func import PunctFuncV0
from rulers.v1_corpus_func import CorpusFuncV1


class StubEmbed:
    def cosine_many(self, pairs):
        return [0.15 for _ in pairs]


def test_t3_confusable_fails_content_not_requiring_style():
    train = [
        "今天早上又堵了啊，真的让人无语了呢吧。",
        "晚饭就随便对付一下吧，明天再说了啊。",
        "这周的事情叠在一起了，有点想摆了。",
        "刚喝完咖啡整个人还是很困啊不想动。",
    ]
    held = train[:2]
    report = eval_ruler(
        PunctFuncV0(),
        train_u=train,
        held_u=held,
        held_pairs=[(train[0], "根据有关规定应当依法予以处理。")],
        contrast=T2_CONTRAST[:5],
        confusable=T3_CONFUSABLE[:4],
        embedder=StubEmbed(),
    )
    assert report["t3"]["content_fail_rate"] == 1.0
    doc = build_document("v0_punct_func", report)
    assert doc["verdict"]["t3"] is True
    assert "t1_held_pass" in report
    assert "t2_contrast_pass" in report


def test_v1_builds_from_train_only():
    train = ["着地的了是不在之而其于所以将已经过。" * 4]
    ruler = CorpusFuncV1(min_count=1)
    report = eval_ruler(
        ruler,
        train_u=train,
        held_u=train,
        held_pairs=[],
        contrast=["根据有关规定应当依法予以处理并及时报告。"],
        confusable=[],
        embedder=None,
    )
    assert report["ruler"] == "v1_corpus_func"
    assert report["t3"]["skipped"] is True
    doc = build_document("v1_corpus_func", report)
    assert doc["verdict"]["t3"] is None


def test_every_registered_ruler_has_accept_json():
    for name in RULERS:
        path = require_accept_file(name)
        assert path == accept_path(name)
        criteria = load_criteria(name)
        assert "t1_held_pass" in criteria
        assert "t2_contrast_pass" in criteria
        assert "t3_content_fail_rate" in criteria


def test_missing_accept_json_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("rulers.accept._RULERS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="missing"):
        require_accept_file("v9_missing")
