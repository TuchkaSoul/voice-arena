"""Piper — слабая атака раунда 0.

VITS в ONNX: работает на CPU быстрее реального времени, голос весит
около 60 МБ, никакого torch. Звучит заметно «роботно» — и это ровно то,
что нужно детектору как лёгкая цель на старте.

Голоса качаются отдельно и в репозиторий не кладутся, см. README.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import SynthEngine, Utterance


def _chunk_to_float(chunk) -> tuple[np.ndarray, int]:
    """piper 1.3+ отдаёт объекты AudioChunk, более ранние — сырые int16."""
    if hasattr(chunk, "audio_float_array"):
        audio = np.asarray(chunk.audio_float_array, dtype=np.float32)
        return audio, int(chunk.sample_rate)
    if hasattr(chunk, "audio_int16_bytes"):
        pcm = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0, int(chunk.sample_rate)
    raise TypeError(f"неизвестный формат чанка piper: {type(chunk)!r}")


class PiperEngine(SynthEngine):
    name = "piper"

    # Ручки, которые пресет может крутить. Всё, чего нет в пресете,
    # остаётся на значениях по умолчанию самого piper.
    _CONFIG_KEYS = (
        "speaker_id",       # для многодикторных голосов
        "length_scale",     # темп: больше — медленнее
        "noise_scale",      # разброс тембра
        "noise_w_scale",    # разброс длительностей фонем
        "volume",
        "normalize_audio",
    )

    def __init__(self, params: dict):
        super().__init__(params)
        self._voice = None

    def load(self) -> None:
        try:
            from piper import PiperVoice
        except ImportError as e:  # pragma: no cover - зависит от окружения
            raise RuntimeError(
                "не установлен piper-tts. Поставь: pip install piper-tts"
            ) from e

        model = Path(self.params["model"])
        if not model.exists():
            raise FileNotFoundError(
                f"нет файла голоса: {model}. Скачай его по инструкции из synth/README.md"
            )
        if not model.with_suffix(model.suffix + ".json").exists():
            raise FileNotFoundError(
                f"рядом с {model.name} нет файла {model.name}.json — "
                "piper без него не знает частоту дискретизации и фонемы"
            )
        self._voice = PiperVoice.load(
            str(model), use_cuda=bool(self.params.get("use_cuda", False))
        )

    def _syn_config(self):
        """Собирает SynthesisConfig, если версия piper его поддерживает."""
        try:
            from piper import SynthesisConfig
        except ImportError:
            return None  # piper 1.2 и раньше: тонкой настройки нет
        kwargs = {k: self.params[k] for k in self._CONFIG_KEYS if k in self.params}
        return SynthesisConfig(**kwargs) if kwargs else None

    def say(self, text: str) -> Utterance:
        self.ensure_loaded()

        if not hasattr(self._voice, "synthesize"):  # pragma: no cover
            # совсем старый API: один поток int16-байтов
            pcm = b"".join(self._voice.synthesize_stream_raw(text))
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            return Utterance(audio=audio, sr=int(self._voice.config.sample_rate))

        config = self._syn_config()
        stream = (
            self._voice.synthesize(text, syn_config=config)
            if config is not None
            else self._voice.synthesize(text)
        )

        parts: list[np.ndarray] = []
        sr: int | None = None
        for chunk in stream:
            audio, chunk_sr = _chunk_to_float(chunk)
            parts.append(audio)
            if sr is None:
                sr = chunk_sr

        if not parts or sr is None:
            raise RuntimeError(f"piper вернул пустой звук для текста: {text!r}")

        return Utterance(audio=np.concatenate(parts).astype(np.float32), sr=sr)
