from tone_keeper.corpus.bilibili import _to_post


def _item(mid: int, dyn_type: str, text: str, orig_text: str | None = None) -> dict:
    item = {
        "id_str": "99",
        "type": dyn_type,
        "modules": {
            "module_author": {"mid": mid, "pub_ts": 1700000000},
            "module_dynamic": {"desc": {"text": text}},
        },
    }
    if orig_text is not None:
        item["orig"] = {"modules": {"module_dynamic": {"desc": {"text": orig_text}}}}
    return item


def test_drops_other_author():
    assert _to_post("1", _item(2, "DYNAMIC_TYPE_WORD", "别人的动态")) is None


def test_keeps_own_text_not_forwarded_orig():
    post = _to_post("1", _item(1, "DYNAMIC_TYPE_FORWARD", "我补一句", orig_text="别人写的长文"))
    assert post is not None
    assert post.text == "我补一句"
    assert "别人写的长文" not in post.text
    assert post.kind == "forward_caption"
