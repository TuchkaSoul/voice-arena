import os
import sys
import urllib.request
import numpy as np
import torch
import onnxruntime as ort
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

SR = 16000
ort.set_default_logger_severity(3)

SILERO_VAD_URLS = [
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
]
ECAPA_URLS = [
    "https://huggingface.co/Wespeaker/wespeaker-voxceleb-ECAPA512/resolve/main/voxceleb_ECAPA512.onnx"
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

def ensure_silero_vad(path: str) -> str: return _download_file(SILERO_VAD_URLS, path, "Silero VAD")
def ensure_ecapa(path: str) -> str: return _download_file(ECAPA_URLS, path, "ECAPA")

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
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoFeatureExtractor.from_pretrained(model_path)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForAudioClassification.from_pretrained(model_path, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self.spoof_idx = 1 

    def predict(self, audio: np.ndarray) -> float:
        chunk_samples = SR * 5  
        chunks = []
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) < SR:
                chunk = np.pad(chunk, (0, SR - len(chunk)), 'constant')
            chunks.append(chunk)
            
        if not chunks: return 0.0

        inputs = self.processor(chunks, sampling_rate=SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device, dtype=torch.float16) if v.dtype == torch.float32 else v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            scores = probs[:, self.spoof_idx].cpu().tolist()
            
        return float(np.mean(scores)) if scores else 0.0

class ECAPA:
    SR = 16000
    N_MELS = 80

    def __init__(self, model_path: str):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        providers = [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
            "CPUExecutionProvider"
        ]
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.expects_fbank = (len(shape) == 3 and isinstance(shape[-1], int) and shape[-1] == self.N_MELS)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        if self.expects_fbank:
            feat = self._log_fbank(audio)
            x = feat[np.newaxis, :, :].astype(np.float32)
        else:
            x = audio[np.newaxis, :].astype(np.float32)
        emb = self.session.run(None, {self.input_name: x})[0]
        return np.asarray(emb, dtype=np.float32).flatten()

    def _log_fbank(self, audio: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.SR, n_fft=400, hop_length=160, win_length=400, n_mels=self.N_MELS, fmin=20, fmax=7600
        )
        return np.log(mel + 1e-6).T