from tone_keeper.eval.metrics import score_pair, summarize_destroy
from tone_keeper.fingerprint.profile import build_profile
from tone_keeper.teacher.destroy import FakeDestroyer


def test_fake_destroy_moves_style_and_keeps_content():
    u = "今天早上上班路上又堵成这样真的让人无语了啊"
    oral = [
        u,
        "刚喝完一杯咖啡整个人还是很困根本不想动了",
        "晚饭就随便对付一下吧明天再好好吃上一顿饭",
    ]
    profile = build_profile(oral)
    a = FakeDestroyer().rewrite(u)
    scores = score_pair(u, a, profile)
    assert scores.content > 0.5
    summary = summarize_destroy([(u, a)], profile)
    assert summary["n"] == 1.0
