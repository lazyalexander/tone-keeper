from __future__ import annotations

from pathlib import Path

from tone_keeper.config import Config, student_prompt


class MlxSampler:
    def __init__(self, cfg: Config, adapter_path: Path | None = None):
        self.cfg = cfg
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load

        kwargs = {}
        if self.adapter_path is not None:
            kwargs["adapter_path"] = str(self.adapter_path)
        self._model, self._tokenizer = load(self.cfg.student.model, **kwargs)

    def sample(self, source: str, n: int | None = None) -> list[str]:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        self._load()
        count = n if n is not None else self.cfg.infer.n_samples
        prompt = student_prompt(self.cfg.prompts.student, source)
        messages = [{"role": "user", "content": prompt}]
        rendered = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        sampler = make_sampler(temp=self.cfg.infer.temperature)
        outputs: list[str] = []
        for _ in range(count):
            text = generate(
                self._model,
                self._tokenizer,
                prompt=rendered,
                max_tokens=self.cfg.infer.max_tokens,
                sampler=sampler,
                verbose=False,
            )
            outputs.append(text.strip())
        return outputs
