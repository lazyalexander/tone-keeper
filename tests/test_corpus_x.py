from tone_keeper.corpus.x import _to_post


def test_drops_retweet_and_other_author():
    rt = {
        "id": "1",
        "text": "RT @x: hello",
        "author_id": "9",
        "referenced_tweets": [{"type": "retweeted", "id": "2"}],
    }
    assert _to_post("9", rt) is None
    other = {"id": "3", "text": "not mine", "author_id": "8"}
    assert _to_post("9", other) is None


def test_keeps_own_tweet_and_reply():
    tweet = {"id": "4", "text": "own post long enough", "author_id": "9", "created_at": "2024-01-01T00:00:00.000Z"}
    post = _to_post("9", tweet)
    assert post is not None
    assert post.kind == "tweet"
    reply = {
        "id": "5",
        "text": "own reply",
        "author_id": "9",
        "in_reply_to_user_id": "1",
        "referenced_tweets": [{"type": "replied_to", "id": "0"}],
    }
    post = _to_post("9", reply)
    assert post is not None
    assert post.kind == "reply"
