"""XTTS-v2 — сильная атака: синтез с клонированием голоса по образцу.

Обёртка вокруг наработки Антона (TTS.api + speaker_wav). Отличия от
исходного скрипта:

  * отдаёт массив, а не файл — кодеки накладывает ступень channel,
    одна и та же для синтеза и для живых записей;
  * COQUI_TOS_AGREED выставляется до импорта, иначе в батч-прогоне
    библиотека молча зависнет на вопросе про лицензию;
  * speaker_wav приходит из пресета, а не зашит в код.

Лицензия: XTTS-v2 под CPML — некоммерческая. Для учебного проекта это
нормально, но в записке её надо перечислить.

Этика: speaker_wav должен быть записью голоса кого-то из команды.
Голоса посторонних людей (дикторов, актёров, публичных персон) не
клонируем — это правило из README проекта.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .base import SynthEngine, Utterance


class XttsEngine(SynthEngine):
    name = "xtts"

    DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(self, params: dict):
        super().__init__(params)
        self._tts = None
        self._speaker_wav: list[str] = []

    def load(self) -> None:
        # Должно стоять до импорта TTS: иначе интерактивный вопрос про CPML
        # повесит прогон на 200 фраз без единого сообщения в консоль.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        # transformers подхватывает TensorFlow, если тот виден в окружении
        # (у нас venv с --system-site-packages, и он виден). Это лишние
        # секунды на каждом запуске и гора предупреждений в логе прогона.
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_TORCH", "1")

        try:
            from TTS.api import TTS
        except ImportError as e:  # pragma: no cover - зависит от окружения
            # Различаем «не поставлен» и «поставлен, но не заводится».
            # Второе случается регулярно: coqui-tts требует transformers 4.x,
            # а что угодно в окружении может подтянуть пятую ветку — и тогда
            # импорт падает глубоко внутри, на отсутствующей функции.
            if getattr(e, "name", "") in ("TTS", "TTS.api"):
                raise RuntimeError(
                    "не установлен coqui-tts. Поставь: pip install coqui-tts\n"
                    "(оригинальный пакет TTS от Coqui заброшен, живой форк — coqui-tts)"
                ) from e
            raise RuntimeError(
                f"coqui-tts установлен, но не импортируется: {e}\n"
                "Обычно это несовместимая версия transformers — нужна ветка 4.x:\n"
                "    pip install 'transformers<5'"
            ) from e

        self._speaker_wav = self._resolve_speaker_wav()

        self._tts = TTS(self.params.get("model", self.DEFAULT_MODEL))
        device = self.params.get("device", "cpu")
        if device == "auto":
            device = self._autodetect_device()
        self._tts.to(device)

    @staticmethod
    def _autodetect_device() -> str:
        try:
            import torch
        except ImportError:  # pragma: no cover
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_speaker_wav(self) -> list[str]:
        """speaker_wav: строка или список. Чем больше образцов, тем стабильнее тембр."""
        raw = self.params.get("speaker_wav")
        if not raw:
            raise ValueError(
                "в пресете не задан speaker_wav — XTTS без образца голоса не работает"
            )
        paths = [raw] if isinstance(raw, str) else list(raw)
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"нет образцов голоса: {', '.join(missing)}")
        return [str(Path(p)) for p in paths]

    def say(self, text: str) -> Utterance:
        self.ensure_loaded()

        wav = self._tts.tts(
            text=text,
            speaker_wav=self._speaker_wav,
            language=self.params.get("language", "ru"),
            **self.params.get("generate_params", {}),
        )

        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError(f"XTTS вернул пустой звук для текста: {text!r}")

        return Utterance(audio=audio, sr=self._output_sr())

    def _output_sr(self) -> int:
        synthesizer = getattr(self._tts, "synthesizer", None)
        sr = getattr(synthesizer, "output_sample_rate", None)
        return int(sr) if sr else 24000  # XTTS-v2 по умолчанию 24 кГц
