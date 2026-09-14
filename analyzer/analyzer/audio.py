"""Загрузка аудио и детекция речи через Silero VAD (ONNX)."""
import numpy as np
import librosa
import onnxruntime as ort

SR = 16000
VAD_WINDOW = 512  # сэмплов при 16 кГц — требование Silero
VAD_CONTEXT = 64  # скользящий контекст для v5


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Читает файл, ресемплит в моно 16 кГц, возвращает float32."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


class SileroVAD:
    """Обёртка над silero_vad.onnx (v5, с контекстом 64 сэмпла)."""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.context = np.zeros(VAD_CONTEXT, dtype=np.float32)

    def _init_state(self) -> np.ndarray:
        return np.zeros((2, 1, 128), dtype=np.float32)

    def _run_chunk(self, chunk: np.ndarray, state: np.ndarray):
        x = np.concatenate([self.context, chunk])[np.newaxis, :].astype(np.float32)
        sr = np.array(SR, dtype=np.int64)
        prob, state = self.session.run(
            None, {"input": x, "state": state, "sr": sr}
        )
        self.context = chunk[-VAD_CONTEXT:].copy()
        return float(prob[0][0]), state

    def speech_segments(
        self,
        audio: np.ndarray,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
    ) -> list[tuple[float, float]]:
        self.context = np.zeros(VAD_CONTEXT, dtype=np.float32)

        num_windows = len(audio) // VAD_WINDOW
        if num_windows == 0:
            return []

        probs = np.zeros(num_windows, dtype=np.float32)
        state = self._init_state()
        for i in range(num_windows):
            chunk = audio[i * VAD_WINDOW:(i + 1) * VAD_WINDOW]
            probs[i], state = self._run_chunk(chunk, state)

        is_speech = probs > threshold
        segments: list[tuple[float, float]] = []
        start_idx: int | None = None

        for i, flag in enumerate(is_speech):
            if flag and start_idx is None:
                start_idx = i
            elif not flag and start_idx is not None:
                self._maybe_append(segments, start_idx, i, min_speech_ms)
                start_idx = None
        if start_idx is not None:
            self._maybe_append(segments, start_idx, num_windows, min_speech_ms)

        return segments

    @staticmethod
    def _maybe_append(segments, start_idx, end_idx, min_speech_ms):
        start_sec = start_idx * VAD_WINDOW / SR
        end_sec = end_idx * VAD_WINDOW / SR
        if (end_sec - start_sec) * 1000 >= min_speech_ms:
            segments.append((start_sec, end_sec))