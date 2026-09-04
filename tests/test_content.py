from tone_keeper.config import FilterConfig
from tone_keeper.filter.content import char_cosine, length_ratio
from tone_keeper.filter.pairs import filter_pair


def test_near_paraphrase_passes_cosine():
    u = "今天早上上班路上又堵成这样真的让人无语了啊"
    a = "今天早上上班路上又堵成这样确实让人无语了啊。"
    assert char_cosine(a, u) > 0.65


def test_unrelated_is_low():
    u = "今天早上上班路上又堵成这样真的让人无语了啊"
    a = "The committee shall convene pursuant to article four."
    assert char_cosine(a, u) < 0.4


def test_filter_rejects_summary_and_expansion():
    cfg = FilterConfig(tau_pair=0.72, tau_len=0.5, max_expand=1.8)
    u = "今天早上上班路上又堵成这样真的让人无语了啊"
    ok, reason = filter_pair("无语。", u, cfg)
    assert not ok
    assert reason in {"length", "expanded", "content"}
    long_a = u + "根据实际情况进行进一步说明。" * 6
    ok, reason = filter_pair(long_a, u, cfg)
    assert not ok
    assert reason == "expanded"


def test_length_ratio_symmetric():
    assert length_ratio("abcd", "abcdefgh") == 0.5
