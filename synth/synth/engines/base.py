"""Интерфейс движка синтеза.

Весь остальной код проекта знает только этот файл. Если смена модели
синтеза потребует правок в pipeline, channel или manifest — значит
абстракция дырявая, и чинить её надо здесь, а не там.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Utterance:
    """Одна синтезированная фраза — общий формат для всех движков."""

    audio: np.ndarray  # моно, float32, значения в [-1, 1]
    sr: int

    def __post_init__(self) -> None:
        if self.audio.ndim != 1:
            raise ValueError(f"ожидался моно-сигнал, получен массив {self.audio.shape}")
        if self.audio.dtype != np.float32:
            raise ValueError(f"ожидался float32, получен {self.audio.dtype}")
        if self.sr <= 0:
            raise ValueError(f"некорректная частота дискретизации: {self.sr}")

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sr


class SynthEngine(ABC):
    """Движок синтеза речи.

    Экземпляр создаётся под конкретный пресет: настройки приходят в
    конструктор один раз и дальше не меняются. Поэтому «пресет» и
    «настроенный движок» — одно и то же, и перепутать настройки между
    двумя прогонами нельзя даже случайно.
    """

    name: str = "base"

    def __init__(self, params: dict):
        self.params = dict(params)
        self._loaded = False

    def ensure_loaded(self) -> None:
        """Ленивая загрузка весов: один раз за прогон, а не на каждую фразу."""
        if not self._loaded:
            self.load()
            self._loaded = True

    @abstractmethod
    def load(self) -> None:
        """Подгружает веса. Тяжёлая операция."""

    @abstractmethod
    def say(self, text: str) -> Utterance:
        """Синтезирует одну фразу. Текст на вход приходит уже нормализованным."""

    def describe(self) -> dict:
        """Что записать в манифест, чтобы прогон можно было воспроизвести."""
        return {"engine": self.name, "params": self.params}
