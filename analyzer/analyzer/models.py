import numpy as np
import torch
import onnxruntime as ort
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

SR = 16000
ort.set_default_logger_severity(3)

class Wav2Vec2AntiSpoof:
    """Deepfake detection using FP16 optimized Wav2Vec2 classification."""
    
    def __init__(self, model_id: str = "garystafford/wav2vec2-deepfake-voice-detector"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        self.spoof_idx = 1 

    def predict(self, audio: np.ndarray) -> float:
        chunk_samples = SR * 5  
        scores = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) < SR:
                chunk = np.pad(chunk, (0, SR - len(chunk)), 'constant')
                
            inputs = self.processor(chunk, sampling_rate=SR, return_tensors="pt", padding=True)
            inputs = {
                k: v.to(self.device, dtype=torch.float16) if v.dtype == torch.float32 else v.to(self.device) 
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            scores.append(float(probs[0, self.spoof_idx].item()))
            
        return float(np.mean(scores)) if scores else 0.0

class ECAPA:
    """Speaker embedding extractor (VoxCeleb ECAPA512)."""
    
    SR = 16000
    N_MELS = 80

    def __init__(self, model_path: str):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
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
            y=audio, sr=self.SR, n_fft=400, hop_length=160, win_length=400, 
            n_mels=self.N_MELS, fmin=20, fmax=7600
        )
        return np.log(mel + 1e-6).T