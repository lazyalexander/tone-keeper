from __future__ import annotations

import importlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _kit_names() -> list[str]:
    names: list[str] = []
    for child in sorted(_ROOT.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue
        if (child / "__init__.py").is_file() and (child / "accept.json").is_file():
            names.append(child.name)
    return names


def _load_cls(name: str):
    mod = importlib.import_module(f"rulers.{name}")
    cls = getattr(mod, "Ruler", None)
    if cls is None:
        raise ImportError(f"rulers.{name} must export Ruler")
    if getattr(cls, "name", None) != name:
        raise ImportError(f"rulers.{name} Ruler.name={getattr(cls, 'name', None)!r} != {name!r}")
    return cls


def load_ruler(name: str):
    known = _kit_names()
    if name not in known:
        raise KeyError(f"unknown style.ruler={name!r}; known: {', '.join(known)}")
    return _load_cls(name)()


def assert_profile_matches_config(profile, ruler_name: str) -> None:
    got = getattr(profile, "ruler", None) or "v0_punct_func"
    if got != ruler_name:
        raise ValueError(
            f"profile.ruler={got!r} != config style.ruler={ruler_name!r}; "
            "rebuild the profile with the configured ruler"
        )


RULERS = {name: _load_cls(name) for name in _kit_names()}
PunctFuncV0 = RULERS["v0_punct_func"]
CorpusFuncV1 = RULERS["v1_corpus_func"]

__all__ = [
    "RULERS",
    "PunctFuncV0",
    "CorpusFuncV1",
    "load_ruler",
    "assert_profile_matches_config",
]
