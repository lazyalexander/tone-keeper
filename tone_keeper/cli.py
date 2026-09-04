from __future__ import annotations

import argparse
import json
from pathlib import Path

from tone_keeper.config import Config, load
from tone_keeper.envfile import load_dotenv
from tone_keeper.fingerprint.profile import profile_from_dict
from tone_keeper.paths import adapter_dir, sft_dir, work_dir
from tone_keeper.pipeline import ingest, load_pairs, prepare, split_and_write
from tone_keeper.pipeline import build_and_write_profile
from tone_keeper.data.jsonl import read_jsonl, write_jsonl


def _cfg(path: str | None) -> Config:
    return load(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tone_keeper",
        description="Local personal register transfer. SFT only; gates are not the loss.",
    )
    parser.add_argument("--config", default="configs/default.toml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Split raw files into tweet-sized units (never merge).")
    p_ingest.add_argument("--input", required=True)
    p_ingest.add_argument("--out", default="data/work/units.jsonl")

    p_profile = sub.add_parser("profile", help="Build fingerprint profile from units. Eval only.")
    p_profile.add_argument("--units", default="data/work/units.jsonl")
    p_profile.add_argument("--out", default="data/work/profile.json")

    p_prep = sub.add_parser(
        "prepare",
        help="Ingest, split, destroy style, filter, export SFT jsonl. valid CE is not checkpoint selection.",
    )
    p_prep.add_argument("--input", required=True)
    p_prep.add_argument("--destroyer", choices=["deepseek", "fake", "mlx"], default="deepseek")
    p_prep.add_argument("--workers", type=int, default=8)
    p_prep.add_argument("--round", type=int, default=1)

    p_pack = sub.add_parser("pack-lora", help="Refilter existing pairs and write lora/sft + catalog.duckdb.")
    p_pack.add_argument("--pairs", default="lora/pairs.jsonl")
    p_pack.add_argument("--round", type=int, default=1)

    p_train = sub.add_parser("train", help="LoRA SFT via mlx-lm. Writes lora/adapters.")
    p_train.add_argument("--data", default=None)
    p_train.add_argument("--adapter", default=None)

    p_rebuild = sub.add_parser(
        "rebuild-catalog",
        help="Drop and recreate lora/catalog.duckdb from pairs.jsonl. Does not retrain or re-embed.",
    )
    p_rebuild.add_argument("--pairs", default="lora/pairs.jsonl")
    p_rebuild.add_argument("--round", type=int, default=1)

    p_embed = sub.add_parser(
        "score-embed",
        help="Score existing pairs with embedding cosine. Does not change ok/reason.",
    )
    p_embed.add_argument("--pairs", default="lora/pairs.jsonl")
    p_embed.add_argument("--round", type=int, default=1)

    p_peek = sub.add_parser("peek", help="Inspect lora/catalog.duckdb (counts, or --sql).")
    p_peek.add_argument("--sql", default=None)
    p_peek.add_argument("--db", default=None)

    p_rewrite = sub.add_parser("rewrite", help="Sample N candidates, content hard gate + relative style gate.")
    p_rewrite.add_argument("--text", required=True)
    p_rewrite.add_argument("--profile", default="data/work/profile.json")
    p_rewrite.add_argument("--adapter", default=None)

    p_eval = sub.add_parser("eval-heldout", help="Held-out Pass@1 using gates, not train CE.")
    p_eval.add_argument("--pairs", default="lora/pairs.jsonl")
    p_eval.add_argument("--profile", default="data/work/profile.json")
    p_eval.add_argument("--adapter", default=None)

    p_ed = sub.add_parser("eval-destroy", help="Destroy quality: content cosine + Cosine Delta vs user profile.")
    p_ed.add_argument("--pairs", default="lora/pairs.jsonl")
    p_ed.add_argument("--profile", default="data/work/profile.json")

    args = parser.parse_args(argv)
    load_dotenv()
    cfg = _cfg(args.config)

    if args.cmd == "ingest":
        units = ingest(Path(args.input), cfg, Path(args.out))
        split_and_write(units, cfg, work_dir() / "splits.json")
        print(json.dumps({"units": len(units)}, ensure_ascii=False))
        return 0

    if args.cmd == "profile":
        units = read_jsonl(Path(args.units))
        profile = build_and_write_profile(units, Path(args.out))  # type: ignore[arg-type]
        print(json.dumps({"units": len(units), "dim": len(profile.mean)}, ensure_ascii=False))
        return 0

    if args.cmd == "prepare":
        stats = prepare(
            Path(args.input),
            cfg,
            destroyer_kind=args.destroyer,
            workers=args.workers,
            round=args.round,
        )
        print(json.dumps(stats, ensure_ascii=False))
        return 0

    if args.cmd == "pack-lora":
        from tone_keeper.lora.catalog import write_pairs
        from tone_keeper.lora.export import export_sft, refilter_pairs

        pairs_path = Path(args.pairs)
        pairs = refilter_pairs(load_pairs(pairs_path))
        write_jsonl(pairs_path, pairs)
        dropped = [p for p in pairs if not p["ok"]]
        write_jsonl(pairs_path.with_name("pairs.dropped.jsonl"), dropped)
        train = [p for p in pairs if p["split"] == "train"]
        held = [p for p in pairs if p["split"] == "held"]
        export_sft(train, held, sft_dir(), cfg)
        db = write_pairs(
            pairs,
            round=args.round,
            teacher_model=cfg.teacher.model,
            embed_model=cfg.embed.model,
        )
        print(
            json.dumps(
                {
                    "round": args.round,
                    "train_ok": sum(1 for p in train if p["ok"]),
                    "held_ok": sum(1 for p in held if p["ok"]),
                    "dropped": sum(1 for p in pairs if not p["ok"]),
                    "sft": str(sft_dir()),
                    "db": str(db),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.cmd == "rebuild-catalog":
        from tone_keeper.lora.catalog import write_pairs

        pairs_path = Path(args.pairs)
        pairs = load_pairs(pairs_path)
        db = write_pairs(
            pairs,
            round=args.round,
            teacher_model=cfg.teacher.model,
            embed_model=cfg.embed.model,
        )
        print(
            json.dumps(
                {
                    "round": args.round,
                    "n": len(pairs),
                    "ok": sum(1 for p in pairs if p.get("ok")),
                    "db": str(db),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.cmd == "score-embed":
        from tone_keeper.filter.embed import MlxEmbedder, score_pairs, shuffled_negatives, summarize
        from tone_keeper.lora.catalog import write_pairs

        pairs_path = Path(args.pairs)
        pairs = load_pairs(pairs_path)
        embedder = MlxEmbedder(model=cfg.embed.model, batch_size=cfg.embed.batch_size)
        pairs = score_pairs(pairs, embedder)
        write_jsonl(pairs_path, pairs)
        negs = shuffled_negatives(pairs, embedder)
        db = write_pairs(
            pairs,
            round=args.round,
            teacher_model=cfg.teacher.model,
            embed_model=cfg.embed.model,
        )
        report = summarize(pairs, negs)
        report["db"] = str(db)
        print(json.dumps(report, ensure_ascii=False))
        return 0

    if args.cmd == "peek":
        from tone_keeper.lora.catalog import catalog_path, peek

        db = Path(args.db) if args.db else catalog_path()
        print(peek(db, sql=args.sql))
        return 0

    if args.cmd == "train":
        from tone_keeper.lora.train import run_sft

        data = Path(args.data) if args.data else sft_dir()
        adapter = Path(args.adapter) if args.adapter else adapter_dir()
        return run_sft(cfg, data_dir=data, adapter_path=adapter)

    if args.cmd == "rewrite":
        from tone_keeper.infer.rewrite import rewrite

        profile = profile_from_dict(json.loads(Path(args.profile).read_text(encoding="utf-8")))
        adapter = Path(args.adapter) if args.adapter else adapter_dir()
        result = rewrite(args.text, profile, cfg, adapter_path=adapter)
        payload = {
            "rejected": result.rejected,
            "n_pass": result.n_pass,
            "text": result.text,
            "reasons": list(result.reasons),
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 1 if result.rejected else 0

    if args.cmd == "eval-heldout":
        from tone_keeper.eval.heldout import eval_heldout
        from tone_keeper.infer.generate import MlxSampler

        profile = profile_from_dict(json.loads(Path(args.profile).read_text(encoding="utf-8")))
        pairs = load_pairs(Path(args.pairs))
        adapter = Path(args.adapter) if args.adapter else adapter_dir()
        sampler = MlxSampler(cfg, adapter_path=adapter)
        report = eval_heldout(pairs, profile, cfg, sampler=sampler)
        print(
            json.dumps(
                {"n": report.n, "pass_at_1": report.pass_at_1, "n_accepted": report.n_accepted},
                ensure_ascii=False,
            )
        )
        return 0

    if args.cmd == "eval-destroy":
        from tone_keeper.eval.metrics import summarize_destroy

        profile = profile_from_dict(json.loads(Path(args.profile).read_text(encoding="utf-8")))
        pairs = load_pairs(Path(args.pairs))
        ok_pairs = [(p["u"], p["a"]) for p in pairs if p.get("ok")]
        print(json.dumps(summarize_destroy(ok_pairs, profile), ensure_ascii=False))
        return 0

    return 2
