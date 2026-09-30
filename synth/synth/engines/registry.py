"""Реестр движков: имя из пресета → класс.

Импорт ленивый, и это принципиально: xtts тянет torch, chatterbox тоже.
Они не должны ломать `python main.py presets` на машине, где стоит
только piper.
"""
from __future__ import annotations

import importlib

from .base import SynthEngine

_ENGINES: dict[str, tuple[str, str]] = {
    "piper": ("synth.engines.piper", "PiperEngine"),
    "xtts": ("synth.engines.xtts", "XttsEngine"),
    "chatterbox": ("synth.engines.chatterbox", "ChatterboxEngine"),
}


def available() -> list[str]:
    return sorted(_ENGINES)


def create(name: str, params: dict) -> SynthEngine:
    if name not in _ENGINES:
        raise KeyError(
            f"неизвестный движок {name!r}; доступны: {', '.join(available())}"
        )
    module_name, class_name = _ENGINES[name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(params)
