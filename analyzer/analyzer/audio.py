import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa

ort.set_default_logger_severity(3)

SR = 16000
VAD_WINDOW = 512
VAD_CONTEXT = 64

def load_audio(path: str, sr: int = SR) -> np.ndarray:
    try:
        y, file_sr = sf.read(path, dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)
        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    except Exception:
        y, _ = librosa.load(path, sr=sr, mono=True)
    return y

class SileroVAD:
    def __init__(self, model_path: str):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        providers = [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
            "CPUExecutionProvider"
        ]
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
        self.context = np.zeros(VAD_CONTEXT, dtype=np.float32)

    def _init_state(self) -> np.ndarray:
        return np.zeros((2, 1, 128), dtype=np.float32)

    def _run_chunk(self, chunk: np.ndarray, state: np.ndarray):
        x = np.concatenate([self.context, chunk])[np.newaxis, :].astype(np.float32)
        sr_tensor = np.array(SR, dtype=np.int64)
        prob, state = self.session.run(None, {"input": x, "state": state, "sr": sr_tensor})
        self.context = chunk[-VAD_CONTEXT:].copy()
        return float(prob[0][0]), state

    def speech_segments(
        self, audio: np.ndarray, threshold: float = 0.5, min_speech_ms: int = 250
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