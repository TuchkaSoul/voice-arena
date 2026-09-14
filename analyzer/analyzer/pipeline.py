"""Сборка пайплайна: аудио → VAD → AASIST + ECAPA → агрегированный результат."""
import time
import numpy as np

from analyzer.audio import load_audio, SileroVAD, SR
from analyzer.models import AASIST, ECAPA


class Analyzer:
    def __init__(
        self,
        vad_path: str,
        aasist_path: str,
        ecapa_path: str,
    ):
        self.vad = SileroVAD(vad_path)
        self.aasist = AASIST(aasist_path)
        self.ecapa = ECAPA(ecapa_path)

    def analyze(self, path: str, spoof_threshold: float = 0.5) -> dict:
        t0 = time.perf_counter()

        audio = load_audio(path)
        segments = self.vad.speech_segments(audio)

        spoof_scores: list[float] = []
        embeddings: list[np.ndarray] = []

        for start, end in segments:
            chunk = audio[int(start * SR):int(end * SR)]
            if len(chunk) < 512:  # слишком короткий кусок — пропускаем
                continue
            spoof_scores.append(self.aasist.predict(chunk))
            embeddings.append(self.ecapa.embed(chunk))

        if spoof_scores:
            spoof_score = float(np.mean(spoof_scores))
            embedding = np.mean(embeddings, axis=0)
        else:
            spoof_score = 0.0
            embedding = None

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "file": path,
            "duration": len(audio) / SR,
            "num_segments": len(segments),
            "spoof_score": spoof_score,
            "verdict": "spoof" if spoof_score > spoof_threshold else "bonafide",
            "gender": "unknown",  # TODO: обучить классификатор поверх эмбеддингов
            "embedding": embedding,
            "latency_ms": latency_ms,
        }