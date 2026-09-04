from tone_keeper.data.schema import Unit
from tone_keeper.data.split import split_units_set


def _unit(i: int, source: str) -> Unit:
    return Unit(id=f"{source}-{i}", source_id=source, text=f"text-{source}-{i}-padding-ok")


def test_same_source_stays_on_one_side():
    units = [_unit(i, "a") for i in range(4)] + [_unit(i, "b") for i in range(4)]
    train, held = split_units_set(units, ratio=0.5, seed=0, by_source=True)
    train_src = {u["source_id"] for u in train}
    held_src = {u["source_id"] for u in held}
    assert train_src.isdisjoint(held_src)
    assert {u["id"] for u in train}.isdisjoint({u["id"] for u in held})
    assert len(train) + len(held) == len(units)


def test_single_source_falls_back_to_unit_split():
    units = [_unit(i, "only") for i in range(10)]
    train, held = split_units_set(units, ratio=0.2, seed=1, by_source=True)
    assert len(held) == 2
    assert len(train) == 8
    assert {u["id"] for u in train}.isdisjoint({u["id"] for u in held})
