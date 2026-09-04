from __future__ import annotations

from pathlib import Path

from tone_keeper.config import Config
from tone_keeper.filter.embed import default_embedder
from tone_keeper.fingerprint.profile import Profile
from tone_keeper.infer.gates import RewriteResult, select_rewrite
from tone_keeper.infer.generate import MlxSampler


def rewrite(
    source: str,
    profile: Profile,
    cfg: Config,
    adapter_path: Path | None = None,
    sampler: MlxSampler | None = None,
) -> RewriteResult:
    engine = sampler or MlxSampler(cfg, adapter_path=adapter_path)
    candidates = engine.sample(source)
    embedder = default_embedder(cfg.embed.model, cfg.embed.batch_size)
    return select_rewrite(candidates, source, profile, cfg.filter, embedder=embedder)
