"""Bilibili user-space dynamics via public polymer API.

Default: that user's own 动态 text. Forwards of other people are dropped
except the user's added caption.

This is not "comments under their videos" (those are other authors).
There is no stable official endpoint for "all comments this mid ever wrote".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator

from tone_keeper.corpus.http import DEFAULT_UA, RateClient, env
from tone_keeper.corpus.models import Post


def fetch_posts(mid: str, client: RateClient | None = None) -> Iterator[Post]:
    cookie = env("TONE_KEEPER_BILI_COOKIE") or ""
    headers = {
        "User-Agent": DEFAULT_UA,
        "Referer": f"https://space.bilibili.com/{mid}/dynamic",
    }
    if cookie:
        headers["Cookie"] = cookie
    own = client or RateClient(headers, pause_s=0.8)
    offset = ""
    try:
        while True:
            params = {"host_mid": mid, "timezone_offset": "-480", "features": "itemOpusStyle"}
            if offset:
                params["offset"] = offset
            payload = own.get(
                "https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space",
                params=params,
            ).json()
            if payload.get("code") not in (0, None):
                raise RuntimeError(f"bilibili api error: {payload.get('message') or payload}")
            data = payload.get("data") or {}
            for item in data.get("items") or []:
                post = _to_post(mid, item)
                if post is not None:
                    yield post
            if not data.get("has_more"):
                break
            offset = str(data.get("offset") or "")
            if not offset:
                break
    finally:
        if client is None:
            own.close()


def _to_post(mid: str, item: dict[str, Any]) -> Post | None:
    modules = item.get("modules") or {}
    author = (modules.get("module_author") or {})
    author_mid = str(author.get("mid") or "")
    if author_mid and author_mid != str(mid):
        return None
    dyn_type = item.get("type") or ""
    if dyn_type == "DYNAMIC_TYPE_FORWARD":
        text = _major_text(modules)  # user's caption only, not orig
        kind = "forward_caption"
    else:
        text = _major_text(modules)
        kind = "dynamic"
    text = (text or "").strip()
    if not text:
        return None
    post_id = str(item.get("id_str") or item.get("id") or "")
    if not post_id:
        return None
    ts = author.get("pub_ts")
    created_at = datetime.fromtimestamp(int(ts), tz=timezone.utc) if ts else None
    return Post(
        source="bilibili",
        user_id=str(mid),
        post_id=post_id,
        kind=kind,
        text=text,
        created_at=created_at,
        url=f"https://t.bilibili.com/{post_id}",
        raw={"type": dyn_type, "id": post_id},
    )


def _major_text(modules: dict[str, Any]) -> str:
    dynamic = modules.get("module_dynamic") or {}
    desc = dynamic.get("desc") or {}
    if isinstance(desc, dict) and desc.get("text"):
        return str(desc["text"])
    major = dynamic.get("major") or {}
    opus = major.get("opus") or {}
    if opus.get("summary"):
        summary = opus["summary"]
        if isinstance(summary, dict):
            return str(summary.get("text") or "")
        return str(summary)
    archive = major.get("archive") or {}
    if archive.get("title"):
        return str(archive.get("title") or "")
    return ""
