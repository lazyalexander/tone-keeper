from tone_keeper.data.schema import Pair
from tone_keeper.filter.embed import score_pairs, shuffled_negatives, summarize


class StubEmbedder:
    def cosine_many(self, pairs):
        out = []
        for left, right in pairs:
            if not left.strip() or not right.strip():
                out.append(0.0)
            elif left == right:
                out.append(1.0)
            else:
                out.append(0.5)
        return out


def _pair(i: str, u: str, a: str, ok: bool = True, reason: str = "ok") -> Pair:
    return Pair(
        id=i,
        source_id="x",
        u=u,
        a=a,
        split="train",
        ok=ok,
        reason=reason,
        char_score=0.2,
    )


def test_score_pairs_writes_embed_score():
    pairs = [
        _pair("1", "hello", "hello"),
        _pair("2", "hello", "world", ok=False, reason="content"),
        _pair("3", "x", "", ok=False, reason="empty"),
    ]
    scored = score_pairs(pairs, StubEmbedder())
    assert [p["embed_score"] for p in scored] == [1.0, 0.5, 0.0]
    assert [p["scores"]["embed"] for p in scored] == [1.0, 0.5, 0.0]


def test_shuffled_negatives_crosses_pairs():
    pairs = [
        _pair("1", "aaa", "AAA"),
        _pair("2", "bbb", "BBB"),
    ]
    negs = shuffled_negatives(pairs, StubEmbedder(), shift=1)
    assert negs == [0.5, 0.5]


def test_summarize_counts_disagreement():
    pairs = [
        _pair("1", "a", "a", ok=False, reason="content"),
        _pair("2", "b", "b", ok=True, reason="ok"),
    ]
    scored = score_pairs(pairs, StubEmbedder())
    report = summarize(scored, negatives=[0.2])
    assert report["content_fail_embed_ge_0.85"] == 1
    assert report["ok_embed_lt_0.70"] == 0
    assert report["neg_p50"] == 0.2
