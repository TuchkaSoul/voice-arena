import time
import numpy as np
import torch
from analyzer.audio import load_audio, SileroVAD, SR
from analyzer.models import Wav2Vec2AntiSpoof, ensure_silero_vad, ensure_wav2vec2, ensure_ecapa

class Analyzer:
    def __init__(self, vad_path: str, model_path: str, model_id: str, ecapa_path: str):
        actual_vad_path = ensure_silero_vad(vad_path)
        actual_w2v2_path = ensure_wav2vec2(model_path, model_id)
        ensure_ecapa(ecapa_path)

        self.vad = SileroVAD(actual_vad_path)
        self.ast_w2v2 = Wav2Vec2AntiSpoof(model_path=actual_w2v2_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def _check_advanced_spectrum(self, audio_np: np.ndarray) -> tuple[float, float, float]:
        tensor = torch.from_numpy(audio_np).float().unsqueeze(0).to(self.device)
        window = torch.hann_window(1024, device=self.device)
        X = torch.stft(tensor, n_fft=1024, hop_length=256, window=window, return_complex=True)
        mag = torch.abs(X)
        
        start_bin = int(512 * (6000 / (SR / 2))) 
        hf_mag = mag[:, start_bin:, :]
        g_mean = torch.exp(torch.mean(torch.log(hf_mag + 1e-10), dim=1))
        a_mean = torch.mean(hf_mag, dim=1)
        flatness = g_mean / (a_mean + 1e-10)
        
        energy_thresh = torch.max(mag) * 1e-3
        mask = (torch.sum(mag, dim=1) > energy_thresh).float()
        median_flatness = float(torch.median(flatness[mask > 0]).item()) if torch.sum(mask) > 0 else 0.0

        cum_energy = torch.cumsum(mag, dim=1)
        total_energy = cum_energy[:, -1:, :]
        rolloff_mask = cum_energy >= (0.85 * total_energy)
        rolloff_bins = torch.argmax(rolloff_mask.float(), dim=1)
        median_rolloff = float(torch.median(rolloff_bins.float()).item() * (SR / 1024))

        freq_indices = torch.arange(mag.shape[1], device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        mean_freq = torch.sum(freq_indices * mag, dim=1) / (torch.sum(mag, dim=1) + 1e-10)
        var_freq = torch.sum(((freq_indices - mean_freq.unsqueeze(1)) ** 2) * mag, dim=1) / (torch.sum(mag, dim=1) + 1e-10)
        median_var = float(torch.median(var_freq).item())

        del tensor, window, X, mag, hf_mag, cum_energy, total_energy, rolloff_mask
        
        return median_flatness, median_rolloff, median_var

    def analyze(self, path: str, spoof_threshold: float = 0.5) -> dict:
        t0 = time.perf_counter()

        audio = load_audio(path)
        segments = self.vad.speech_segments(audio)

        if not segments:
            return self._build_result(path, audio, segments, 0.0, "Нет речи", spoof_threshold, t0)

        valid_chunks = [audio[int(s * SR):int(e * SR)] for s, e in segments if (e - s) * SR >= 512]

        if not valid_chunks:
            return self._build_result(path, audio, segments, 0.0, "Недостаточно данных", spoof_threshold, t0)

        speech_audio = np.concatenate(valid_chunks)
        if len(speech_audio) > SR * 15:
            speech_audio = speech_audio[:SR * 15]
        
        flatness, rolloff, spec_var = self._check_advanced_spectrum(speech_audio)
        w2v2_score = self.ast_w2v2.predict(speech_audio)

        if w2v2_score >= 0.985:
            final_spoof_score = w2v2_score
            dsp_reason = f"Абсолютная уверенность ML (W: {w2v2_score:.2f})"
        elif 0.30 <= flatness <= 0.48:
            final_spoof_score = min(w2v2_score, spoof_threshold - 0.1)
            dsp_reason = f"Физика микрофона (W: {w2v2_score:.2f}, F: {flatness:.2f}, R: {rolloff:.0f}Hz)"
        elif flatness > 0.55:
            final_spoof_score = 0.99
            dsp_reason = f"Аномалия ВЧ (Сглаживание): (F: {flatness:.2f}, R: {rolloff:.0f}Hz)"
        elif flatness < 0.20:
            final_spoof_score = 0.99
            dsp_reason = f"Аномалия ВЧ (Артефакты): (F: {flatness:.2f}, Var: {spec_var:.1f})"
        else:
            final_spoof_score = w2v2_score
            dsp_reason = f"ML-Анализ (W: {w2v2_score:.2f}, F: {flatness:.2f})"

        del speech_audio, valid_chunks, audio

        result = self._build_result(path, None, segments, final_spoof_score, dsp_reason, spoof_threshold, t0)
        result.update({
            "model_score": w2v2_score,
            "flatness": flatness,
            "rolloff_hz": rolloff,
            "spectrum_variance": spec_var,
        })
        return result

    def _build_result(self, path: str, audio: np.ndarray, segments: list, spoof_score: float, reason: str, threshold: float, t0: float) -> dict:
        return {
            "file": path, 
            "duration": sum([e - s for s, e in segments]) if segments else 0.0,
            "num_segments": len(segments), 
            "spoof_score": spoof_score, 
            "verdict": "spoof" if spoof_score > threshold else "bonafide", 
            "embedding": None, 
            "dsp_reason": reason, 
            "latency_ms": (time.perf_counter() - t0) * 1000
        }
