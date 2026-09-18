import time
import numpy as np
import torch
from analyzer.audio import load_audio, SileroVAD, SR
from analyzer.models import Wav2Vec2AntiSpoof

class Analyzer:
    def __init__(self, vad_path: str, model_id: str = "garystafford/wav2vec2-deepfake-voice-detector"):
        self.vad = SileroVAD(vad_path)
        self.ast_w2v2 = Wav2Vec2AntiSpoof(model_id=model_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def _check_spectral_flatness(self, audio_np: np.ndarray) -> float:
        tensor = torch.from_numpy(audio_np).float().unsqueeze(0).to(self.device)
        X = torch.stft(tensor, n_fft=1024, hop_length=256, window=torch.hann_window(1024, device=self.device), return_complex=True)
        mag = torch.abs(X)
        
        start_bin = int(512 * (6000 / (SR / 2))) 
        high_freq_mag = mag[:, start_bin:, :]
        
        geometric_mean = torch.exp(torch.mean(torch.log(high_freq_mag + 1e-10), dim=1))
        arithmetic_mean = torch.mean(high_freq_mag, dim=1)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        energy_threshold = torch.max(mag) * 1e-3
        mask = (torch.sum(mag, dim=1) > energy_threshold).float()
        
        if torch.sum(mask) > 0:
            return torch.median(flatness[mask > 0]).item()
        return 0.0

    def analyze(self, path: str, spoof_threshold: float = 0.5) -> dict:
        t0 = time.perf_counter()

        audio = load_audio(path)
        segments = self.vad.speech_segments(audio)

        if not segments:
            return {
                "file": path, "duration": len(audio) / SR, "num_segments": 0, 
                "spoof_score": 0.0, "verdict": "bonafide", "embedding": None, 
                "dsp_reason": "No speech detected", "latency_ms": (time.perf_counter() - t0) * 1000
            }

        valid_chunks = [audio[int(start * SR):int(end * SR)] for start, end in segments if (end - start) * SR >= 512]

        if valid_chunks:
            speech_audio = np.concatenate(valid_chunks)
            
            max_samples = SR * 15
            if len(speech_audio) > max_samples:
                speech_audio = speech_audio[:max_samples]
            
            flatness = self._check_spectral_flatness(speech_audio)
            
            if flatness > 0.55:
                final_spoof_score = 0.99
                dsp_reason = f"Аномалия ВЧ (Сглаживание): {flatness:.2f}"
            elif flatness < 0.20:
                final_spoof_score = 0.99
                dsp_reason = f"Аномалия ВЧ (Артефакты): {flatness:.2f}"
            else:
                w2v2_score = self.ast_w2v2.predict(speech_audio)
                
                if w2v2_score >= 0.985:
                    final_spoof_score = w2v2_score
                    dsp_reason = f"Абсолютная уверенность ML: {w2v2_score:.2f}"
                elif 0.35 < flatness < 0.48:
                    final_spoof_score = min(w2v2_score, spoof_threshold - 0.1)
                    dsp_reason = f"Физика микрофона (W: {w2v2_score:.2f}, F: {flatness:.2f})"
                else:
                    final_spoof_score = w2v2_score
                    dsp_reason = f"ML-Анализ (W: {w2v2_score:.2f}, F: {flatness:.2f})"
        else:
            final_spoof_score = 0.0
            dsp_reason = "Insufficient data"

        verdict = "spoof" if final_spoof_score > spoof_threshold else "bonafide"

        return {
            "file": path, 
            "duration": len(audio) / SR, 
            "num_segments": len(segments), 
            "spoof_score": final_spoof_score, 
            "verdict": verdict, 
            "embedding": None, 
            "dsp_reason": dsp_reason, 
            "latency_ms": (time.perf_counter() - t0) * 1000
        }