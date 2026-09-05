"""Ruler interface for style-instrument experiments.

A ruler is NOT the training loss. It scores how close a text is to a corpus
profile F built from train-side user text U.

    f(x)  = concatenated punctuation and function-char rates
            f_c(x) = count_c(x) / max(cjk_len(x), 1)
    μ, σ  = mean and std of f(u) over train U
    F     = (μ, σ)

Default distance (raw cosine to the mean), lower = closer to the corpus:

    d_cos(x, F) = 1 - <f(x), μ> / (||f(x)|| ||μ||)

Optional z-scored L2 (Burrows-like typicality; cosine-to-mean is undefined
after centering):

    d_z(x, F) = || (f(x) - μ) / σ ||_2
"""

from __future__ import annotations

from typing import Any, Protocol


class Ruler(Protocol):
    name: str

    def build(self, train_texts: list[str]) -> Any:
        """Fit F from train-side user texts only. Held U must not enter F."""

    def distance(self, text: str, profile: Any) -> float:
        """Return d(text, F). Lower means closer to the corpus profile."""
