"""造对过滤（U vs A）。只看两条文本。不看来源、不看用户。

推理门（ŷ vs A）不在这里，在 tone_keeper/infer/gates.py。
阈值只改本文件顶部。每条判定带分数和 log，写入 DuckDB。

content 门是 embedding 余弦（命题是否同一），不是字符 n-gram。
char n-gram 只记日志；近拷贝拒绝在推理侧 tau_copy。
"""

from __future__ import annotations

from dataclasses import dataclass

from tone_keeper.filter.content import char_cosine, expand_ratio, length_ratio
from tone_keeper.text.length import cjk_ratio

# --- 阈值：改这里 ---
EMBED_COSINE_MIN = 0.80
LENGTH_RATIO_MIN = 0.40
EXPAND_MAX = 2.5
LANGUAGE_CJK_GAP_MAX = 0.5


@dataclass(frozen=True)
class Decision:
    ok: bool
    reason: str
    content: float
    embed: float
    length_ratio: float
    expand_ratio: float | None
    cjk_gap: float
    log: str

    def scores(self) -> dict[str, float | None]:
        return {
            "embed": self.embed,
            "char": self.content,
            "length": self.length_ratio,
            "expand": self.expand_ratio,
            "cjk_gap": self.cjk_gap,
        }


def _fmt(value: float) -> str:
    if value == float("inf") or value == float("-inf"):
        return "inf"
    return f"{value:.3f}"


def _mark(passed: bool) -> str:
    return "pass" if passed else "fail"


def decide(
    u: str,
    a: str,
    *,
    embed_score: float,
    embed_min: float = EMBED_COSINE_MIN,
    length_min: float = LENGTH_RATIO_MIN,
    expand_max: float = EXPAND_MAX,
    language_gap_max: float = LANGUAGE_CJK_GAP_MAX,
) -> Decision:
    u = (u or "").strip()
    a = (a or "").strip()
    content = char_cosine(a, u) if u and a else 0.0
    length = length_ratio(a, u) if u and a else 0.0
    expand = expand_ratio(a, u) if u and a else float("inf")
    cjk_gap = abs(cjk_ratio(a) - cjk_ratio(u)) if u and a else 1.0
    embed = float(embed_score) if u and a else 0.0

    pass_lang = cjk_gap <= language_gap_max
    pass_expand = expand <= expand_max
    pass_length = length >= length_min
    pass_embed = embed >= embed_min

    if not u or not a:
        reason = "empty"
    elif not pass_lang:
        reason = "language"
    elif not pass_expand:
        reason = "expanded"
    elif not pass_length:
        reason = "length"
    elif not pass_embed:
        reason = "content"
    else:
        reason = "ok"

    log = (
        f"{reason}"
        f" | embed={_fmt(embed)} >= {embed_min:.3f} {_mark(pass_embed)}"
        f" | char={_fmt(content)}"
        f" | length={_fmt(length)} >= {length_min:.3f} {_mark(pass_length)}"
        f" | expand={_fmt(expand)} <= {expand_max:.3f} {_mark(pass_expand)}"
        f" | cjk_gap={_fmt(cjk_gap)} <= {language_gap_max:.3f} {_mark(pass_lang)}"
    )
    return Decision(
        ok=reason == "ok",
        reason=reason,
        content=content,
        embed=embed,
        length_ratio=length,
        expand_ratio=None if expand == float("inf") else expand,
        cjk_gap=cjk_gap,
        log=log,
    )
