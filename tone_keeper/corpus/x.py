"""X (Twitter) via official API v2. Needs TONE_KEEPER_X_BEARER."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

from tone_keeper.corpus.http import RateClient, env
from tone_keeper.corpus.models import Post

API = "https://api.x.com/2"


def fetch_posts(user_id: str, client: RateClient | None = None) -> Iterator[Post]:
    token = env("TONE_KEEPER_X_BEARER")
    if not token:
        raise RuntimeError("set TONE_KEEPER_X_BEARER to an X API v2 bearer token")
    own = client or RateClient(
        {
            "Authorization": f"Bearer {token}",
            "User-Agent": "tone-keeper-corpus",
        },
        pause_s=1.0,
    )
    params: dict[str, str | int] = {
        "max_results": 100,
        "exclude": "retweets",
        "tweet.fields": "created_at,author_id,lang,referenced_tweets,in_reply_to_user_id,conversation_id",
    }
    url = f"{API}/users/{user_id}/tweets"
    try:
        while True:
            payload = own.get(url, params=params).json()
            for item in payload.get("data") or []:
                post = _to_post(user_id, item)
                if post is not None:
                    yield post
            meta = payload.get("meta") or {}
            token_next = meta.get("next_token")
            if not token_next:
                break
            params["pagination_token"] = token_next
    finally:
        if client is None:
            own.close()


def _to_post(user_id: str, item: dict) -> Post | None:
    text = (item.get("text") or "").strip()
    if not text:
        return None
    author = str(item.get("author_id") or user_id)
    if author != str(user_id):
        return None
    kinds = {ref.get("type") for ref in item.get("referenced_tweets") or []}
    if "retweeted" in kinds:
        return None
    kind = "reply" if "replied_to" in kinds or item.get("in_reply_to_user_id") else "tweet"
    created = item.get("created_at")
    created_at = None
    if created:
        created_at = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(timezone.utc)
    post_id = str(item["id"])
    return Post(
        source="x",
        user_id=str(user_id),
        post_id=post_id,
        kind=kind,
        text=text,
        created_at=created_at,
        url=f"https://x.com/i/web/status/{post_id}",
        raw=item,
    )
