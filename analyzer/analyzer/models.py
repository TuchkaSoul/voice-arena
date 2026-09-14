import numpy as np
import onnxruntime as ort
import librosa

SR = 16000


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class AASIST:
    """Детектор синтеза на Spectra-AASIST. Вход: фиксированное окно 64600."""

    WINDOW = 64600  # 4.0375 с при 16 кГц — стандарт ASVspoof

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name  # "wav"
        outputs = [o.name for o in self.session.get_outputs()]
        if "logits" not in outputs:
            raise RuntimeError(f"Нет выхода 'logits'. Доступны: {outputs}")

    def predict(self, audio: np.ndarray) -> float:
        chunks = self._chunk(audio)
        scores = [self._predict_chunk(c) for c in chunks]
        return float(np.mean(scores))

    def _chunk(self, audio: np.ndarray) -> list[np.ndarray]:
        n = len(audio)
        if n == 0:
            return [np.zeros(self.WINDOW, dtype=np.float32)]
        if n <= self.WINDOW:
            return [np.pad(audio, (0, self.WINDOW - n))]

        chunks = []
        step = self.WINDOW // 2
        for start in range(0, n - self.WINDOW + 1, step):
            chunks.append(audio[start:start + self.WINDOW])
        if (n - self.WINDOW) % step != 0:
            chunks.append(audio[-self.WINDOW:])
        return chunks

    def _predict_chunk(self, chunk: np.ndarray) -> float:
        x = chunk[np.newaxis, :].astype(np.float32)
        logits = np.asarray(
            self.session.run(["logits"], {self.input_name: x})[0]
        ).flatten()
        probs = _softmax(logits)
        # порядок классов этой модели: [spoof, bonafide]
        return float(probs[0])


class ECAPA:
    """Экстрактор эмбеддингов голоса (VoxCeleb ECAPA512)."""

    SR = 16000
    N_MELS = 80

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.expects_fbank = (
            len(shape) == 3
            and isinstance(shape[-1], int)
            and shape[-1] == self.N_MELS
        )

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """audio: mono float32 @ 16 кГц. Возвращает вектор (D,)."""
        if self.expects_fbank:
            feat = self._log_fbank(audio)
            x = feat[np.newaxis, :, :].astype(np.float32)
        else:
            x = audio[np.newaxis, :].astype(np.float32)

        emb = self.session.run(None, {self.input_name: x})[0]
        return np.asarray(emb, dtype=np.float32).flatten()

    def _log_fbank(self, audio: np.ndarray) -> np.ndarray:
        """80-мерный log-mel filterbank, совместимый с weSpeaker/SpeechBrain."""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.SR,
            n_fft=400,
            hop_length=160,
            win_length=400,
            n_mels=self.N_MELS,
            fmin=20,
            fmax=7600,
        )
        log_mel = np.log(mel + 1e-6)
        return log_mel.T  # (T, 80)