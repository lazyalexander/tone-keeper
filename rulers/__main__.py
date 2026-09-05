"""Run T1/T2/T3 and write per-ruler acceptance JSON.

    pixi run python -m rulers
    pixi run python -m rulers --ruler v0_punct_func

Each ruler kit must ship rulers/{name}/accept.json. Full run documents go to
lora/logs/rulers/{name}.json (gitignored).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rulers import RULERS
from rulers.accept import build_document, require_accept_file, write_document
from rulers.fixtures import T2_CONTRAST, T3_CONFUSABLE
from rulers.harness import eval_ruler
from tone_keeper.data.jsonl import read_jsonl
from tone_keeper.observe import record
from tone_keeper.pipeline import load_pairs


def _load_extra_jsonl(path: Path) -> list[str]:
    if not path.exists():
        return []
    rows = read_jsonl(path)
    out: list[str] = []
    for row in rows:
        text = row.get("text") or row.get("u") or ""
        if text.strip():
            out.append(text)
    return out


def _load_pairs_jsonl(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    rows = read_jsonl(path)
    out: list[tuple[str, str]] = []
    for row in rows:
        left, right = row.get("s1") or row.get("a") or "", row.get("s2") or row.get("b") or ""
        if left.strip() and right.strip():
            out.append((left, right))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rulers")
    parser.add_argument("--units", default="data/work/units.jsonl")
    parser.add_argument("--pairs", default="lora/pairs.jsonl")
    parser.add_argument("--splits", default="data/work/splits.json")
    parser.add_argument("--ruler", action="append", dest="ruler_names")
    parser.add_argument("--t2", default="rulers/data/t2.jsonl")
    parser.add_argument("--t3", default="rulers/data/t3.jsonl")
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args(argv)

    units = read_jsonl(Path(args.units))
    pairs = load_pairs(Path(args.pairs))
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    train_ids = set(splits["train_ids"])
    held_ids = set(splits["held_ids"])
    train_u = [u["text"] for u in units if u["id"] in train_ids]
    held_u = [u["text"] for u in units if u["id"] in held_ids]
    held_pairs = [
        (p["u"], p["a"])
        for p in pairs
        if p.get("split") == "held" and (p.get("u") or "").strip() and (p.get("a") or "").strip()
    ]
    contrast = T2_CONTRAST + _load_extra_jsonl(Path(args.t2))
    confusable = T3_CONFUSABLE + _load_pairs_jsonl(Path(args.t3))

    embedder = None
    if not args.skip_embed:
        from tone_keeper.filter.embed import MlxEmbedder

        embedder = MlxEmbedder()

    names = args.ruler_names or list(RULERS)
    docs = []
    for name in names:
        require_accept_file(name)
        run = eval_ruler(
            RULERS[name](),
            train_u=train_u,
            held_u=held_u,
            held_pairs=held_pairs,
            contrast=contrast,
            confusable=confusable,
            embedder=embedder,
        )
        doc = build_document(name, run)
        out = write_document(doc)
        record("ruler_eval", {"ruler": name, "path": str(out), "verdict": doc["verdict"]})
        docs.append(doc)
    print(json.dumps(docs, ensure_ascii=False))
    return 0 if all(d["verdict"]["passed"] for d in docs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
