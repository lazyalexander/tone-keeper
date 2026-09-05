import json
from pathlib import Path

from tone_keeper.config import load
from tone_keeper.pipeline import prepare


class HighEmbed:
    def cosine_many(self, pairs):
        return [0.9 if left.strip() and right.strip() else 0.0 for left, right in pairs]


def test_fake_prepare_writes_sft(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load(Path(__file__).resolve().parents[1] / "configs" / "default.toml")
    raw = Path(__file__).resolve().parent / "fixtures" / "raw"
    stats = prepare(raw, cfg, destroyer_kind="fake", root=tmp_path, embedder=HighEmbed())
    assert stats["units"] >= 8
    assert stats["train_ok"] >= 1
    train = tmp_path / "lora" / "sft" / "train.jsonl"
    assert train.exists()
    assert train.read_text(encoding="utf-8").strip()
    profile = tmp_path / "data" / "work" / "profile.json"
    assert profile.exists()
    assert json.loads(profile.read_text(encoding="utf-8"))["ruler"] == "v0_punct_func"
    assert stats["ruler"] == "v0_punct_func"
    assert stats["round"] == 1
    assert (tmp_path / "lora" / "catalog.duckdb").exists()
    assert (tmp_path / "lora" / "pairs.jsonl").exists()
    assert not (tmp_path / "data" / "work" / "pairs.jsonl").exists()
