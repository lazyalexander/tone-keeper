# rulers

Experimental style instruments. Not imported by training or production gates.

Each version is a module plus a sibling **acceptance JSON**:

- `rulers/vN_name.py` — `build(train_U) -> F`, `distance(text, F) -> float` (lower = closer)
- `rulers/vN_name.accept.json` — **required** contract (criteria only). Missing file is an error.
- `rulers/accept.schema.json` — shape of the document

A run stamps `criteria + run + verdict` to `lora/logs/rulers/{name}.json` (gitignored) and appends `lora/logs/events.jsonl` (`kind=ruler_eval`).

Do not promote a ruler into `tone_keeper.infer.gates` until `verdict.hard_style_gate_ok` is true. T3 is a content-instrument check, not a style-gate license.

## Protocols

Formulas (tau is fit on train U only; held U must not enter F):

```
tau = Q_0.8({ d(u, F) : u in U_train })

T1 = |{ u in U_held : d(u, F) <= tau }| / |U_held|
     target >= 0.8
     (false-negative check on the user's own text)

T2 = |{ x in contrast : d(x, F) <= tau }| / |contrast|
     target <= 0.3
     (false-positive check on other-register / generic AI text)

T3: for confusable pairs (similar syntax, unrelated propositions)
     content_fail_rate = |{ (s1,s2) : embed(s1,s2) < 0.80 }| / |pairs|
     target >= 0.8
     Style may match on both sides. Do not require F to reject T3.
```

Do not require T1 = 1.0 on 20–200 character units. That forces a vacuous ball and T2 collapses.

`destroy_rate_held` is extra diagnostics: fraction of held pairs with `d(A,F) > d(U,F)`. It is not T1. Teacher paraphrases of U are the wrong negative for authorship.

## Versions

| module | what |
|---|---|
| `v0_punct_func` | Production baseline: 16 punct + 10 oral function chars, raw cosine to mean |
| `v1_corpus_func` | Punct + function chars frequent in *this* train U, name chars dropped |

Add a new version as `rulers/vN_*.py` **and** `rulers/vN_*.accept.json`, then register it in `RULERS`.

## Data

Built-in T2/T3 fixtures ship in `rulers/fixtures.py` (not the user's corpus). Optional extras:

- `rulers/data/t2.jsonl` — `{"text": "..."}` other authors / AI
- `rulers/data/t3.jsonl` — `{"s1": "...", "s2": "..."}` confusable pairs

`rulers/data/` is gitignored.

## Run

```
pixi run python -m rulers
pixi run python -m rulers --ruler v0_punct_func --skip-embed
```
