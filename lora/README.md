# lora/

LoRA 生产目录。代码在 `tone_keeper/lora/`，这一轮的造对、SFT、adapter、catalog 都在这里。

| 路径 | 内容 |
|---|---|
| `pairs.jsonl` | 全量 (U, A)，含 scores 和 filter_log |
| `pairs.dropped.jsonl` | `reason != ok` 的子集，方便扫一眼 |
| `sft/train.jsonl` `sft/valid.jsonl` | mlx-lm 训练输入 |
| `adapters/` | 训出的 adapter |
| `catalog.duckdb` | 按 round 记。无 source、无 user_id |

`data/work/` 只留 ingest：units / splits / profile。

两张表：

- `gen_runs(round, git_sha, created_at, policy JSON)` — 一轮的政策（阈值、teacher、embed）
- `pairs(round, unit_id, split, u, a, reason, scores JSON, filter_log, ok)` — 含拒绝。`ok` 由 `reason = 'ok'` 生成，不存盘

拒绝就是 `WHERE NOT ok`，没有第二张表。

```
pixi run python -m tone_keeper peek
pixi run python -m tone_keeper peek --sql "SELECT reason, count(*) n, round(avg((scores->>'embed')::DOUBLE),3) e FROM pairs GROUP BY 1"
pixi run python -m tone_keeper peek --sql "SELECT unit_id, reason, scores, filter_log FROM pairs WHERE NOT ok"
pixi run python -m tone_keeper peek --sql "SELECT policy FROM gen_runs"
pixi run python -m tone_keeper rebuild-catalog --round 1
```

Harlequin：`harlequin lora/catalog.duckdb`。关掉旧会话再打开。
