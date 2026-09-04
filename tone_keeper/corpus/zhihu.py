"""Zhihu member API v4 (web). Not an official third-party SDK.

Needs TONE_KEEPER_ZHIHU_COOKIE (logged-in Cookie header).
Fetches that member's own answers / articles / pins. Other authors are dropped.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

from tone_keeper.corpus.http import DEFAULT_UA, RateClient, env
from tone_keeper.corpus.models import Post
from tone_keeper.corpus.text import html_to_text

KINDS = ("answers", "articles", "pins")


def fetch_posts(user_token: str, kinds: tuple[str, ...] = KINDS, client: RateClient | None = None) -> Iterator[Post]:
    cookie = env("TONE_KEEPER_ZHIHU_COOKIE")
    if not cookie:
        raise RuntimeError("set TONE_KEEPER_ZHIHU_COOKIE to a logged-in Zhihu Cookie header")
    own = client or RateClient(
        {
            "User-Agent": DEFAULT_UA,
            "Cookie": cookie,
            "Referer": f"https://www.zhihu.com/people/{user_token}",
        },
        pause_s=1.0,
    )
    try:
        for kind in kinds:
            yield from _paginate(own, user_token, kind)
    finally:
        if client is None:
            own.close()


def _paginate(client: RateClient, user_token: str, kind: str) -> Iterator[Post]:
    offset = 0
    limit = 20
    while True:
        url = f"https://www.zhihu.com/api/v4/members/{user_token}/{kind}"
        params = {
            "limit": limit,
            "offset": offset,
            "include": "data[*].content,created_time,updated_time,created,excerpt_title,question",
        }
        payload = client.get(url, params=params).json()
        items = payload.get("data") or []
        if not items:
            break
        for item in items:
            post = _to_post(user_token, kind, item)
            if post is not None:
                yield post
        paging = payload.get("paging") or {}
        if paging.get("is_end", len(items) < limit):
            break
        offset += len(items)


def _to_post(user_token: str, kind: str, item: dict) -> Post | None:
    author = item.get("author") or {}
    url_token = author.get("url_token")
    if url_token and str(url_token) != str(user_token):
        return None
    html = item.get("content") or item.get("excerpt") or item.get("title") or ""
    text = html_to_text(str(html))
    if not text:
        return None
    created_raw = item.get("created_time") or item.get("created") or item.get("updated_time")
    created_at = None
    if isinstance(created_raw, (int, float)):
        created_at = datetime.fromtimestamp(int(created_raw), tz=timezone.utc)
    post_id = str(item.get("id") or item.get("token") or "")
    if not post_id:
        return None
    url = item.get("url")
    if kind == "answers" and item.get("question"):
        qid = item["question"].get("id")
        url = f"https://www.zhihu.com/question/{qid}/answer/{post_id}"
    elif kind == "articles":
        url = f"https://zhuanlan.zhihu.com/p/{post_id}"
    return Post(
        source="zhihu",
        user_id=str(user_token),
        post_id=f"{kind}:{post_id}",
        kind={"answers": "answer", "articles": "article", "pins": "pin"}[kind],
        text=text,
        created_at=created_at,
        url=url,
        raw={"id": item.get("id"), "type": item.get("type"), "kind": kind},
    )
