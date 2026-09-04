from __future__ import annotations

from dataclasses import dataclass

from tone_keeper.config import Config
from tone_keeper.data.schema import Pair
from tone_keeper.fingerprint.profile import Profile
from tone_keeper.infer.generate import MlxSampler
from tone_keeper.infer.rewrite import rewrite


@dataclass
class HeldoutReport:
    n: int
    pass_at_1: float
    n_accepted: int


def eval_heldout(
    pairs: list[Pair],
    profile: Profile,
    cfg: Config,
    sampler: MlxSampler | None = None,
) -> HeldoutReport:
    held = [pair for pair in pairs if pair["split"] == "held" and pair["ok"]]
    if not held:
        return HeldoutReport(n=0, pass_at_1=0.0, n_accepted=0)
    accepted = 0
    for pair in held:
        result = rewrite(pair["a"], profile, cfg, sampler=sampler)
        if not result.rejected:
            accepted += 1
    return HeldoutReport(
        n=len(held),
        pass_at_1=accepted / len(held),
        n_accepted=accepted,
    )
