"""Load per-ruler acceptance criteria and stamp a run document.

Every registered ruler MUST ship a sibling file:

    rulers/{ruler.name}.accept.json

That file is the contract (criteria only). Evaluation writes the full
document (criteria + run + verdict) to lora/logs/rulers/{name}.json.

    tau = Q_{train_quantile}({ d(u, F) : u in U_train })
    T1  = |{ u in U_held : d(u, F) <= tau }| / |U_held|
    T2  = |{ x in contrast : d(x, F) <= tau }| / |contrast|
    T3  = |{ (s1,s2) : embed(s1,s2) < content_tau }| / |pairs|

hard_style_gate_ok := T1 ok AND T2 ok.
T3 is a content-instrument check; it does not authorize the style gate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tone_keeper.paths import logs_dir

PROTOCOL = "t1_t2_t3_v1"
_RULERS_DIR = Path(__file__).resolve().parent


def accept_path(ruler_name: str) -> Path:
    return _RULERS_DIR / f"{ruler_name}.accept.json"


def require_accept_file(ruler_name: str) -> Path:
    path = accept_path(ruler_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"ruler {ruler_name!r} is missing {path.name}; "
            "every version must ship a sibling .accept.json"
        )
    return path


def load_criteria(ruler_name: str) -> dict[str, Any]:
    path = require_accept_file(ruler_name)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("ruler") != ruler_name:
        raise ValueError(f"{path.name} ruler field {doc.get('ruler')!r} != {ruler_name!r}")
    if doc.get("protocol") != PROTOCOL:
        raise ValueError(f"{path.name} protocol {doc.get('protocol')!r} != {PROTOCOL!r}")
    return doc["criteria"]


def verdict_from_run(criteria: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    t1_ok = run["t1_held_pass"] >= criteria["t1_held_pass"]["min"]
    t2_ok = run["t2_contrast_pass"] <= criteria["t2_contrast_pass"]["max"]
    t3 = run.get("t3") or {}
    if t3.get("skipped"):
        t3_ok: bool | None = None
    else:
        t3_ok = t3["content_fail_rate"] >= criteria["t3_content_fail_rate"]["min"]
    hard = bool(t1_ok and t2_ok)
    return {
        "t1": t1_ok,
        "t2": t2_ok,
        "t3": t3_ok,
        "hard_style_gate_ok": hard,
        "passed": hard,
    }


def build_document(ruler_name: str, run: dict[str, Any]) -> dict[str, Any]:
    criteria = load_criteria(ruler_name)
    run_out = dict(run)
    run_out["ts"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    return {
        "ruler": ruler_name,
        "protocol": PROTOCOL,
        "criteria": criteria,
        "run": run_out,
        "verdict": verdict_from_run(criteria, run_out),
    }


def write_document(doc: dict[str, Any], path: Path | None = None) -> Path:
    out = path or (logs_dir() / "rulers" / f"{doc['ruler']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out
