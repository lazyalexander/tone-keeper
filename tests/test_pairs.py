from tone_keeper.config import load
from tone_keeper.filter.pairs import filter_pair
from tone_keeper.teacher.destroy import FakeDestroyer


def test_fake_destroyer_keeps_pair_filterable():
    cfg = load("configs/default.toml")
    u = "今天早上上班路上又堵成这样真的让人无语了啊"
    a = FakeDestroyer().rewrite(u)
    assert a != u
    assert "啊" in u
    ok, reason = filter_pair(a, u, cfg.filter)
    assert ok, reason
