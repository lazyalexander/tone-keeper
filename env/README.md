# env/

本目录是环境配置，不是训练代码。

| 文件 | 职责 |
|---|---|
| `../pixi.toml` | pixi 声明。必须在仓库根，`pixi install` 从根目录解析 |
| `../pixi.lock` | 解析后的钉死版本。随 `pixi install` 生成 |
| `policy.toml` | Python 版本、平台、禁止依赖 |
| `check_env.py` | 导入与 Metal 可用性检查 |

## 安装

在仓库根：

```
pixi install
pixi run check-env
```

进入 shell：`pixi shell`

系统 Homebrew Python 是 3.14，不用它。训练和推理只走 `pixi run python`。

```
pixi run test
```

## 训练框架命令

语料放进目录（`.txt` / `.jsonl` 的 `text` 字段）。单元是微博级短文本：只切不拼。

```
pixi run python -m tone_keeper prepare --input tests/fixtures/raw --destroyer fake
pixi run python -m tone_keeper train
pixi run python -m tone_keeper rewrite --text "……"
```

`--destroyer mlx` 才下载 14B 教师。`train` 才下载学生 4B。`valid.jsonl` 的 CE 不是选 checkpoint 的依据；用 `eval-heldout`。

```
pixi run download-models
```

## 语料采集

只拉该用户自己写的文本，进 DuckDB。环境变量：

- `TONE_KEEPER_X_BEARER` — X API v2
- `TONE_KEEPER_ZHIHU_COOKIE` — 知乎登录 Cookie
- `TONE_KEEPER_BILI_COOKIE` — B 站 Cookie（动态接口建议带）

```
pixi run python -m tone_keeper.corpus fetch-x --user-id <id>
pixi run python -m tone_keeper.corpus fetch-zhihu --user-id <url_token>
pixi run python -m tone_keeper.corpus fetch-bilibili --user-id <mid>
pixi run python -m tone_keeper.corpus export --out data/raw/from_web.jsonl
```
