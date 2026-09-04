from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from tone_keeper.config import Config
from tone_keeper.data.ingest import ingest_path
from tone_keeper.data.jsonl import read_jsonl, write_jsonl
from tone_keeper.data.schema import Pair, Unit
from tone_keeper.data.split import split_units_set
from tone_keeper.filter.decide import decide
from tone_keeper.filter.embed import Embedder, default_embedder, score_pairs
from tone_keeper.fingerprint.profile import Profile, build_profile
from tone_keeper.lora.catalog import catalog_path, write_pairs
from tone_keeper.lora.export import apply_decision, export_sft
from tone_keeper.paths import dropped_pairs_path, ensure_dir, pairs_path, sft_dir, work_dir
from tone_keeper.teacher.destroy import Destroyer, make_destroyer

_LOG = logging.getLogger("tone_keeper.filter")


def ingest(input_path: Path, cfg: Config, out_path: Path) -> list[Unit]:
    units = ingest_path(input_path, cfg.unit)
    write_jsonl(out_path, units)
    return units


def split_and_write(units: list[Unit], cfg: Config, out_path: Path) -> dict:
    train, held = split_units_set(
        units,
        ratio=cfg.unit.heldout_ratio,
        seed=cfg.unit.seed,
        by_source=True,
    )
    payload = {
        "train_ids": [u["id"] for u in train],
        "held_ids": [u["id"] for u in held],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_and_write_profile(units: list[Unit], out_path: Path) -> Profile:
    profile = build_profile([u["text"] for u in units])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile.to_dict(), ensure_ascii=False), encoding="utf-8")
    return profile


def _destroy_one(
    unit: Unit,
    split_name: Literal["train", "held"],
    destroyer: Destroyer,
    cfg: Config,
) -> Pair:
    del cfg
    a = ""
    try:
        a = destroyer.rewrite(unit["text"])
        reason = "pending"
        log = "pending"
    except Exception as exc:
        reason = f"error:{type(exc).__name__}"
        log = f"{reason}: {exc}"
    return Pair(
        id=unit["id"],
        source_id=unit["source_id"],
        u=unit["text"],
        a=a,
        split=split_name,
        ok=False,
        reason=reason,
        filter_log=log,
    )


def decide_pairs(pairs: list[Pair], embedder: Embedder) -> list[Pair]:
    pending = [p for p in pairs if p["reason"] == "pending"]
    scored = {p["id"]: p for p in score_pairs(pending, embedder)}
    out: list[Pair] = []
    for pair in pairs:
        if pair["reason"] != "pending":
            out.append(pair)
            continue
        row = scored[pair["id"]]
        decision = decide(
            row["u"],
            row["a"],
            embed_score=float(row.get("embed_score") or 0.0),
        )
        _LOG.debug("%s %s", row["id"], decision.log)
        out.append(apply_decision(row, decision))
    return out


def destroy_units(
    units: list[Unit],
    split_name: Literal["train", "held"],
    destroyer: Destroyer,
    cfg: Config,
    workers: int = 8,
) -> list[Pair]:
    if workers <= 1 or len(units) <= 1:
        return [_destroy_one(unit, split_name, destroyer, cfg) for unit in units]
    pairs: list[Pair] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_destroy_one, unit, split_name, destroyer, cfg) for unit in units
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            pairs.append(fut.result())
            if i % 25 == 0 or i == len(units):
                print(f"destroy {split_name} {i}/{len(units)}", flush=True)
    order = {unit["id"]: n for n, unit in enumerate(units)}
    pairs.sort(key=lambda p: order.get(p["id"], 0))
    return pairs


def prepare(
    input_path: Path,
    cfg: Config,
    destroyer_kind: str = "deepseek",
    root: Path | None = None,
    workers: int = 8,
    round: int = 1,
    embedder: Embedder | None = None,
) -> dict[str, int | str]:
    work = ensure_dir(work_dir(root))
    units_path = work / "units.jsonl"
    units = ingest(input_path, cfg, units_path)
    split_payload = split_and_write(units, cfg, work / "splits.json")
    train_ids = set(split_payload["train_ids"])
    held_ids = set(split_payload["held_ids"])
    train_units = [u for u in units if u["id"] in train_ids]
    held_units = [u for u in units if u["id"] in held_ids]

    build_and_write_profile(units, work / "profile.json")
    destroyer = make_destroyer(destroyer_kind, cfg)
    train_pairs = destroy_units(train_units, "train", destroyer, cfg, workers=workers)
    held_pairs = destroy_units(held_units, "held", destroyer, cfg, workers=workers)
    scorer = embedder or default_embedder(cfg.embed.model, cfg.embed.batch_size)
    all_pairs = decide_pairs(train_pairs + held_pairs, scorer)
    train_pairs = [p for p in all_pairs if p["split"] == "train"]
    held_pairs = [p for p in all_pairs if p["split"] == "held"]
    write_jsonl(pairs_path(root), all_pairs)

    dropped = [p for p in all_pairs if not p["ok"]]
    write_jsonl(dropped_pairs_path(root), dropped)

    export_sft(train_pairs, held_pairs, sft_dir(root), cfg)
    db = write_pairs(
        all_pairs,
        round=round,
        path=catalog_path(root),
        teacher_model=cfg.teacher.model,
        embed_model=cfg.embed.model,
    )
    return {
        "units": len(units),
        "train": len(train_units),
        "held": len(held_units),
        "train_ok": sum(1 for p in train_pairs if p["ok"]),
        "held_ok": sum(1 for p in held_pairs if p["ok"]),
        "dropped": len(dropped),
        "round": round,
        "db": str(db),
    }


def load_pairs(path: Path) -> list[Pair]:
    return read_jsonl(path)  # type: ignore[return-value]
