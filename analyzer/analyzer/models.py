import os
import sys
import urllib.request
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

SR = 16000
ort.set_default_logger_severity(3)

SILERO_VAD_URLS = [
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
]


def _download_file(urls: list, path: str, name: str) -> str:
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[*] Downloading {name} to '{path}'...", file=sys.stderr)

    for url in urls:
        try:
            tmp_path = path + ".part"
            urllib.request.urlretrieve(url, tmp_path)
            os.replace(tmp_path, path)
            return path
        except Exception as e:
            print(f"[!] Failed {url}: {e}", file=sys.stderr)
    raise RuntimeError(f"Failed to download {name}")


def ensure_silero_vad(path: str) -> str:
    return _download_file(SILERO_VAD_URLS, path, "Silero VAD")


def ensure_wav2vec2(local_dir: str, model_id: str) -> str:
    if os.path.isdir(local_dir) and any(os.scandir(local_dir)):
        return local_dir
    os.makedirs(local_dir, exist_ok=True)
    print(f"[*] Downloading Wav2Vec2 '{model_id}' to '{local_dir}'...", file=sys.stderr)
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    fe.save_pretrained(local_dir)
    model.save_pretrained(local_dir)
    return local_dir


class Wav2Vec2AntiSpoof:
    def __init__(self, model_path: str, device: str | torch.device | None = None):
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.processor = AutoFeatureExtractor.from_pretrained(model_path)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_path, torch_dtype=dtype
        ).to(device=self.device, dtype=dtype)
        self.model.eval()
        self.spoof_idx = 1

    @torch.inference_mode()
    def predict(self, audio: np.ndarray) -> float:
        chunk_samples = SR * 5
        chunks = []
        weights = []

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            valid_len = len(chunk)
            if valid_len < SR:
                chunk = np.pad(chunk, (0, SR - valid_len), "constant")
            chunks.append(chunk)
            weights.append(valid_len)

        if not chunks:
            return 0.0

        inputs = self.processor(chunks, sampling_rate=SR, return_tensors="pt", padding=True)
        inputs = {
            k: v.to(self.device, dtype=self.model.dtype) if v.dtype.is_floating_point else v.to(self.device)
            for k, v in inputs.items()
        }

        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        scores = probs[:, self.spoof_idx].cpu().tolist()

        return float(np.average(scores, weights=weights)) if scores else 0.0