import shutil
import subprocess
import numpy as np
import onnxruntime as ort
import soundfile as sf
import soxr

ort.set_default_logger_severity(3)

SR = 16000
VAD_WINDOW = 512
VAD_CONTEXT = 64


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """
    Быстрое чтение через libsndfile + прямой C-ресемплинг через soxr (HQ).
    Если контейнер/кодек не поддерживается libsndfile — прямое декодирование через ffmpeg.
    """
    try:
        y, file_sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if file_sr != sr:
            y = soxr.resample(y, file_sr, sr, quality="HQ")
        return np.asarray(y, dtype=np.float32)
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(f"не удалось прочитать {path}; ffmpeg не найден")
        result = subprocess.run(
            [
                ffmpeg, "-v", "error", "-i", path,
                "-ac", "1", "-ar", str(sr),
                "-af", "aresample=resampler=soxr",
                "-f", "f32le", "pipe:1",
            ],
            capture_output=True,
            check=True,
        )
        return np.frombuffer(result.stdout, dtype="<f4").copy()


class SileroVAD:
    def __init__(self, model_path: str):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        # Для окон по 512 сэмплов CPU работает быстрее GPU за счёт отсутствия PCIe-копирований
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_buf = np.zeros((1, VAD_CONTEXT + VAD_WINDOW), dtype=np.float32)
        self._sr_tensor = np.array(SR, dtype=np.int64)

    def speech_segments(
        self, audio: np.ndarray, threshold: float = 0.5, min_speech_ms: int = 250
    ) -> list[tuple[float, float]]:
        num_windows = len(audio) // VAD_WINDOW
        if num_windows == 0:
            return []

        self._input_buf.fill(0.0)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        probs = np.empty(num_windows, dtype=np.float32)

        for i in range(num_windows):
            chunk = audio[i * VAD_WINDOW : (i + 1) * VAD_WINDOW]
            self._input_buf[0, VAD_CONTEXT:] = chunk
            prob, state = self.session.run(
                None, {"input": self._input_buf, "state": state, "sr": self._sr_tensor}
            )
            self._input_buf[0, :VAD_CONTEXT] = chunk[-VAD_CONTEXT:]
            probs[i] = prob[0, 0]

        is_speech = probs > threshold
        segments = []
        start_idx = None

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
    def _maybe_append(segments: list, start_idx: int, end_idx: int, min_speech_ms: int):
        start_sec = start_idx * VAD_WINDOW / SR
        end_sec = end_idx * VAD_WINDOW / SR
        if (end_sec - start_sec) * 1000 >= min_speech_ms:
            segments.append((start_sec, end_sec))