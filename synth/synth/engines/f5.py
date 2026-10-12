"""F5-TTS_RUSSIAN with a local checkpoint, Russian vocabulary and Vocos.

Reference audio must belong to the team. We require its transcript explicitly
so a corpus run never silently downloads a second, large ASR model.
Weights: Misha24-10/F5-TTS_RUSSIAN, CC BY-NC 4.0.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .base import SynthEngine, Utterance


class F5Engine(SynthEngine):
    name = "f5"

    def __init__(self, params: dict):
        super().__init__(params)
        self._tts = None
        self._ref_file = ""
        self._ref_text = ""
        self._vocab: set[str] = set()

    def _file(self, key: str) -> Path:
        value = self.params.get(key)
        if not value:
            raise ValueError(f"F5: в пресете не задан {key}")
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"F5: нет файла {key}: {path}")
        return path

    def load(self) -> None:
        checkpoint = self._file("checkpoint")
        vocab = self._file("vocab")
        reference = self._file("speaker_wav")
        transcript = self._file("ref_text_file")
        self._ref_text = transcript.read_text(encoding="utf-8-sig").strip()
        if not self._ref_text:
            raise ValueError("F5: расшифровка образца голоса пуста")
        self._vocab = set(vocab.read_text(encoding="utf-8").splitlines())
        self._validate_text(self._ref_text)
        self._ref_file = str(reference)
        vocoder = self.params.get("vocoder_path")
        if not vocoder:
            raise ValueError("F5: в пресете не задан vocoder_path")
        for filename in ("config.yaml", "pytorch_model.bin"):
            path = Path(vocoder) / filename
            if not path.is_file():
                raise FileNotFoundError(f"F5: нет файла вокодера: {path}")

        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_TORCH", "1")
        try:
            from f5_tts.api import F5TTS
        except ImportError as e:  # pragma: no cover - installation dependent
            raise RuntimeError(
                f"F5-TTS не импортируется: {e}. "
                "Установи requirements-f5.txt и f5-tts==1.1.22 --no-deps "
                "в отдельное окружение (см. README)."
            ) from e

        device = self.params.get("device", "auto")
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tts = F5TTS(
            model=self.params.get("model", "F5TTS_v1_Base"),
            ckpt_file=str(checkpoint),
            vocab_file=str(vocab),
            vocoder_local_path=str(vocoder),
            device=device,
        )

    def _validate_text(self, text: str) -> None:
        unknown = sorted(set(text) - self._vocab)
        if unknown:
            raise ValueError(f"F5: символы отсутствуют в русском словаре: {unknown!r}")

    def say(self, text: str) -> Utterance:
        self.ensure_loaded()
        self._validate_text(text)
        wav, sr, _ = self._tts.infer(
            ref_file=self._ref_file,
            ref_text=self._ref_text,
            gen_text=text,
            **self.params.get("generate_params", {}),
        )
        audio = np.asarray(wav, dtype=np.float32)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("F5: пустой или некорректный результат синтеза")
        return Utterance(audio=np.clip(audio, -1.0, 1.0), sr=int(sr))
