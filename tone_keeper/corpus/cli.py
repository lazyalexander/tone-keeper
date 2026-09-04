from __future__ import annotations

import argparse
from pathlib import Path

from tone_keeper.corpus.store import CorpusStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tone_keeper.corpus",
        description="Fetch a user's own posts into DuckDB. Drops other authors.",
    )
    parser.add_argument("--db", default="data/corpus.duckdb")
    sub = parser.add_subparsers(dest="cmd", required=True)

    px = sub.add_parser("fetch-x", help="X API v2 user tweets. Env: TONE_KEEPER_X_BEARER")
    px.add_argument("--user-id", required=True)

    pz = sub.add_parser("fetch-zhihu", help="Zhihu member v4. Env: TONE_KEEPER_ZHIHU_COOKIE")
    pz.add_argument("--user-id", required=True, help="url_token, not numeric hash")
    pz.add_argument("--kinds", default="answers,articles,pins")

    pb = sub.add_parser("fetch-bilibili", help="Bilibili space dynamics. Optional TONE_KEEPER_BILI_COOKIE")
    pb.add_argument("--user-id", required=True, help="mid / UID")

    pe = sub.add_parser("export", help="Write posts.text as jsonl for tone_keeper ingest")
    pe.add_argument("--out", default="data/raw/from_web.jsonl")
    pe.add_argument("--source", default=None)
    pe.add_argument("--user-id", default=None)

    args = parser.parse_args(argv)
    store = CorpusStore(Path(args.db))
    try:
        if args.cmd == "fetch-x":
            from tone_keeper.corpus.x import fetch_posts

            n = store.upsert(fetch_posts(args.user_id))
            print({"source": "x", "upserted": n, "total": store.count("x")})
            return 0
        if args.cmd == "fetch-zhihu":
            from tone_keeper.corpus.zhihu import fetch_posts

            kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
            n = store.upsert(fetch_posts(args.user_id, kinds=kinds))
            print({"source": "zhihu", "upserted": n, "total": store.count("zhihu")})
            return 0
        if args.cmd == "fetch-bilibili":
            from tone_keeper.corpus.bilibili import fetch_posts

            n = store.upsert(fetch_posts(args.user_id))
            print({"source": "bilibili", "upserted": n, "total": store.count("bilibili")})
            return 0
        if args.cmd == "export":
            n = store.export_jsonl(Path(args.out), source=args.source, user_id=args.user_id)
            print({"exported": n, "out": args.out})
            return 0
    finally:
        store.close()
    return 2
