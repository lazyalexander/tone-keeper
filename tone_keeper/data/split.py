from __future__ import annotations

import random

from tone_keeper.data.schema import Unit


def split_units_set(
    units: list[Unit],
    ratio: float,
    seed: int,
    by_source: bool = True,
) -> tuple[list[Unit], list[Unit]]:
    if not units:
        return [], []
    sources = sorted({unit["source_id"] for unit in units})
    if by_source and len(sources) > 1:
        return _split_by_source(units, sources, ratio, seed)
    return _split_by_unit(units, ratio, seed)


def _split_by_source(
    units: list[Unit],
    sources: list[str],
    ratio: float,
    seed: int,
) -> tuple[list[Unit], list[Unit]]:
    rng = random.Random(seed)
    shuffled = list(sources)
    rng.shuffle(shuffled)
    n_held = max(1, int(round(len(shuffled) * ratio)))
    if n_held >= len(shuffled):
        n_held = len(shuffled) - 1
    held_sources = set(shuffled[:n_held])
    train = [unit for unit in units if unit["source_id"] not in held_sources]
    held = [unit for unit in units if unit["source_id"] in held_sources]
    return train, held


def _split_by_unit(
    units: list[Unit],
    ratio: float,
    seed: int,
) -> tuple[list[Unit], list[Unit]]:
    rng = random.Random(seed)
    order = list(range(len(units)))
    rng.shuffle(order)
    n_held = int(round(len(units) * ratio))
    n_held = max(0, min(n_held, len(units) - 1))
    held_idx = set(order[:n_held])
    train = [units[i] for i in range(len(units)) if i not in held_idx]
    held = [units[i] for i in range(len(units)) if i in held_idx]
    return train, held
