import json
from pathlib import Path

from tone_keeper.config import load
from tone_keeper.data.schema import Pair
from tone_keeper.lora.export import export_sft, pair_to_example


def test_messages_format(tmp_path: Path):
    cfg = load("configs/default.toml")
    pair = Pair(
        id="x-0",
        source_id="x",
        u="今天早上上班路上又堵成这样真的让人无语了啊",
        a="今天早上上班路上又堵成这样确实令人无语了啊。",
        split="train",
        ok=True,
        reason="ok",
    )
    example = pair_to_example(pair, cfg)
    assert example["messages"][0]["role"] == "user"
    assert pair["a"] in example["messages"][0]["content"]
    assert example["messages"][1] == {"role": "assistant", "content": pair["u"]}
    train, valid = export_sft([pair], [], tmp_path, cfg)
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["messages"][1]["content"] == pair["u"]
    assert valid.read_text(encoding="utf-8") == ""


def test_skips_failed_pairs(tmp_path: Path):
    cfg = load("configs/default.toml")
    bad = Pair(
        id="x-1",
        source_id="x",
        u="u",
        a="a",
        split="train",
        ok=False,
        reason="content",
    )
    export_sft([bad], [], tmp_path, cfg)
    assert (tmp_path / "train.jsonl").read_text(encoding="utf-8") == ""
