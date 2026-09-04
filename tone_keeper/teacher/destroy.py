from __future__ import annotations

import os
import time
from typing import Protocol

import httpx

from tone_keeper.config import Config, teacher_prompt


class Destroyer(Protocol):
    def rewrite(self, u: str) -> str: ...


class FakeDestroyer:
    """Test double only. Prefixes a written-register opener. Not production."""

    def rewrite(self, u: str) -> str:
        body = u.strip()
        if body and body[-1] not in "。！？":
            body += "。"
        return f"综合来看，{body}"


class DeepSeekDestroyer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def rewrite(self, u: str) -> str:
        key = os.environ.get("TONE_KEEPER_DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("set TONE_KEEPER_DEEPSEEK_API_KEY")
        url = self.cfg.teacher.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.teacher.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你只输出改写后的正文。不要解释，不要列表，不要使用markdown。",
                },
                {"role": "user", "content": teacher_prompt(self.cfg.prompts.teacher, u)},
            ],
            "temperature": self.cfg.teacher.temperature,
            "max_tokens": self.cfg.teacher.max_tokens,
            "thinking": {"type": "disabled"},
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        body = payload
        for attempt in range(4):
            try:
                response = httpx.post(url, json=body, headers=headers, timeout=60.0)
                if response.status_code == 400 and "thinking" in body:
                    body = dict(payload)
                    body.pop("thinking", None)
                    continue
                response.raise_for_status()
                text = (response.json()["choices"][0]["message"].get("content") or "").strip()
                if not text:
                    raise RuntimeError("deepseek returned empty content")
                return text
            except (httpx.HTTPError, RuntimeError, KeyError, IndexError) as exc:
                last_error = exc
                time.sleep(1.5 ** attempt)
        raise RuntimeError(f"deepseek rewrite failed: {last_error}") from last_error


class MlxDestroyer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load

        self._model, self._tokenizer = load(self.cfg.teacher.model)

    def rewrite(self, u: str) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        self._load()
        prompt = teacher_prompt(self.cfg.prompts.teacher, u)
        messages = [{"role": "user", "content": prompt}]
        rendered = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        sampler = make_sampler(temp=self.cfg.teacher.temperature)
        return generate(
            self._model,
            self._tokenizer,
            prompt=rendered,
            max_tokens=self.cfg.teacher.max_tokens,
            sampler=sampler,
            verbose=False,
        ).strip()


def make_destroyer(kind: str, cfg: Config) -> Destroyer:
    if kind == "fake":
        return FakeDestroyer()
    if kind == "deepseek":
        return DeepSeekDestroyer(cfg)
    if kind == "mlx":
        return MlxDestroyer(cfg)
    raise ValueError(f"unknown destroyer: {kind}")
