from tone_keeper.filter.decide import decide


U = "今天早上上班路上又堵成这样真的让人无语了啊"


def test_near_paraphrase_ok():
    d = decide(U, "今天早上上班路上又堵成这样确实让人无语了啊。", embed_score=0.95)
    assert d.ok
    assert d.reason == "ok"
    assert "embed=" in d.log
    assert "char=" in d.log


def test_empty():
    d = decide(U, "  ", embed_score=0.99)
    assert not d.ok
    assert d.reason == "empty"
    assert d.expand_ratio is None


def test_language():
    d = decide(
        U,
        "The committee shall convene pursuant to article four of the charter.",
        embed_score=0.99,
    )
    assert not d.ok
    assert d.reason == "language"
    assert "cjk_gap=" in d.log
    assert "fail" in d.log


def test_expanded():
    d = decide(U, U + "根据实际情况进行进一步说明。" * 8, embed_score=0.99)
    assert not d.ok
    assert d.reason == "expanded"


def test_length():
    d = decide(U, "无语。", embed_score=0.99)
    assert not d.ok
    assert d.reason == "length"


def test_content_fail_is_embed_not_char():
    a = "晚饭就随便对付一下吧明天再好好吃上一顿，实在懒得再出门了。"
    d = decide(U, a, embed_score=0.50)
    assert not d.ok
    assert d.reason == "content"
    assert d.embed == 0.50
    assert "embed=0.500 >= 0.800 fail" in d.log
    assert "char=" in d.log


def test_low_char_high_embed_passes():
    d = decide(U, "完全换词但命题相同的书面改写。", embed_score=0.94)
    assert d.ok
    assert d.reason == "ok"


def test_log_shows_thresholds():
    d = decide(U, U, embed_score=0.96)
    assert ">= 0.800" in d.log
    assert ">= 0.400" in d.log
    assert "<= 2.500" in d.log
    assert "<= 0.500" in d.log
