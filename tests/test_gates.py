from tone_keeper.config import FilterConfig
from tone_keeper.fingerprint.profile import build_profile
from tone_keeper.infer.gates import select_rewrite


class HighEmbed:
    def cosine_many(self, pairs):
        return [0.9 for _ in pairs]

ORAL = [
    "今天早上上班路上又堵成这样真的让人无语了啊",
    "刚喝完一杯咖啡整个人还是很困根本不想动了",
    "这周的deadline全叠在一起我有点想直接摆了",
    "晚饭就随便对付一下吧明天再好好吃上一顿",
]


def test_copying_source_fails_relative_style():
    profile = build_profile(ORAL)
    source = "今天早上上班路上又堵成这样确实令人无语了啊。"
    cfg = FilterConfig()
    result = select_rewrite([source], source, profile, cfg, embedder=HighEmbed())
    assert result.rejected
    assert result.n_pass == 0
    assert "copy" in result.reasons


def test_oral_candidate_can_pass_both_gates():
    profile = build_profile(ORAL)
    source = "今天早上上班路上又堵成这样确实令人无语了啊。"
    candidate = ORAL[0]
    cfg = FilterConfig()
    result = select_rewrite([source, candidate], source, profile, cfg, embedder=HighEmbed())
    assert not result.rejected
    assert result.text == candidate
    assert result.n_pass == 1


def test_low_embed_is_content_reject():
    profile = build_profile(ORAL)
    source = "今天早上上班路上又堵成这样确实令人无语了啊。"
    cfg = FilterConfig()

    class LowEmbed:
        def cosine_many(self, pairs):
            return [0.2 for _ in pairs]

    result = select_rewrite([ORAL[0]], source, profile, cfg, embedder=LowEmbed())
    assert result.rejected
    assert "content" in result.reasons


def test_all_fail_does_not_return_best_rejected():
    profile = build_profile(ORAL)
    source = "根据有关规定应当依法予以处理并及时报告"
    cfg = FilterConfig()
    result = select_rewrite([source, source + "。"], source, profile, cfg, embedder=HighEmbed())
    assert result.rejected
    assert result.text is None
