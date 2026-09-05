from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UnitConfig:
    min_chars: int = 20
    max_chars: int = 200
    heldout_ratio: float = 0.2
    seed: int = 0


@dataclass(frozen=True)
class FilterConfig:
    tau_pair: float = 0.80
    tau_len: float = 0.40
    max_expand: float = 2.5
    tau_copy: float = 0.97


@dataclass(frozen=True)
class InferConfig:
    n_samples: int = 8
    temperature: float = 0.9
    max_tokens: int = 256


@dataclass(frozen=True)
class StudentConfig:
    model: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    adapter_path: str = "lora/adapters/default"


@dataclass(frozen=True)
class TeacherConfig:
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    max_tokens: int = 256
    temperature: float = 0.3


@dataclass(frozen=True)
class LoraConfig:
    iters: int = 400
    batch_size: int = 2
    learning_rate: float = 1e-5
    fine_tune_type: str = "lora"
    num_layers: int = 16
    mask_prompt: bool = True
    max_seq_length: int = 1024


@dataclass(frozen=True)
class StyleConfig:
    ruler: str = "v0_punct_func"


@dataclass(frozen=True)
class EmbedConfig:
    model: str = "mlx-community/Qwen3-Embedding-0.6B-8bit"
    batch_size: int = 16


@dataclass(frozen=True)
class PromptConfig:
    student: str = (
        "将下面的文字转写成我的文风。保持原意。不要润色，不要补充，不要解释，不要扩写成更长的段落。\n\n{A}"
    )
    teacher: str = (
        "把下面这段改写成常见的大模型书面语：完整、中性、去掉个人口吻和标点习惯。\n"
        "必须保持原意和大致篇幅：不要增删事实，不要改立场，不要写成更长的文章，不要解释。\n"
        "只输出改写后的正文。\n\n{U}"
    )


@dataclass(frozen=True)
class Config:
    unit: UnitConfig
    filter: FilterConfig
    infer: InferConfig
    student: StudentConfig
    teacher: TeacherConfig
    lora: LoraConfig
    prompts: PromptConfig
    embed: EmbedConfig
    style: StyleConfig


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"[{name}] must be a table")
    return value


def load(path: str | Path | None = None) -> Config:
    config_path = Path(path) if path else Path("configs/default.toml")
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    return Config(
        unit=UnitConfig(**_section(raw, "unit")),
        filter=FilterConfig(**_section(raw, "filter")),
        infer=InferConfig(**_section(raw, "infer")),
        student=StudentConfig(**_section(raw, "student")),
        teacher=TeacherConfig(**_section(raw, "teacher")),
        lora=LoraConfig(**_section(raw, "lora")),
        prompts=PromptConfig(**_section(raw, "prompts")),
        embed=EmbedConfig(**_section(raw, "embed")),
        style=StyleConfig(**_section(raw, "style")),
    )


def student_prompt(template: str, a: str) -> str:
    return template.replace("{A}", a)


def teacher_prompt(template: str, u: str) -> str:
    return template.replace("{U}", u)
