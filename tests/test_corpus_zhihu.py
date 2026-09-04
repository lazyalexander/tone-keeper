from tone_keeper.corpus.zhihu import _to_post


def test_drops_other_author_answer():
    item = {
        "id": 1,
        "content": "<p>别人的回答</p>",
        "author": {"url_token": "other"},
        "created_time": 1700000000,
        "question": {"id": 2},
    }
    assert _to_post("me", "answers", item) is None


def test_keeps_own_answer_html_stripped():
    item = {
        "id": 3,
        "content": "<p>我自己写的回答内容足够长</p>",
        "author": {"url_token": "me"},
        "created_time": 1700000000,
        "question": {"id": 9},
    }
    post = _to_post("me", "answers", item)
    assert post is not None
    assert post.text == "我自己写的回答内容足够长"
    assert post.kind == "answer"
    assert "question/9/answer/3" in (post.url or "")
