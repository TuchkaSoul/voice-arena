import os
import time
import numpy as np
import torch
from analyzer.audio import load_audio, SileroVAD, SR
from analyzer.models import Wav2Vec2AntiSpoof, ensure_silero_vad, ensure_wav2vec2

LOSSY_EXTS = {".ogg", ".opus"}
IQR_FLATNESS_MIN = 0.08
FORMANT_BAND_LOW_HZ = 300.0
FORMANT_BAND_HIGH_HZ = 3400.0
FORMANT_FLATNESS_MIN = 0.02
FORMANT_FLATNESS_MAX = 0.65


class Analyzer:
    def __init__(
        self,
        vad_path: str = "models/silero_vad.onnx",
        model_path: str = "models/wav2vec2-deepfake-voice-detector",
        model_id: str = "garystafford/wav2vec2-deepfake-voice-detector",
        device: str = "auto",
    ):
        if device == "auto" or device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        actual_vad_path = ensure_silero_vad(vad_path)
        actual_w2v2_path = ensure_wav2vec2(model_path, model_id)

        self.vad = SileroVAD(actual_vad_path)
        self.ast_w2v2 = Wav2Vec2AntiSpoof(model_path=actual_w2v2_path, device=self.device)

        self._stft_window = None
        self._freq_indices = None

    def _get_stft_buffers(self, n_bins: int = 513) -> tuple[torch.Tensor, torch.Tensor]:
        if self._stft_window is None or self._stft_window.device != self.device:
            self._stft_window = torch.hann_window(1024, device=self.device)
        if (
            self._freq_indices is None
            or self._freq_indices.device != self.device
            or self._freq_indices.shape[1] != n_bins
        ):
            self._freq_indices = (
                torch.arange(n_bins, device=self.device, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(2)
            )
        return self._stft_window, self._freq_indices

    @torch.inference_mode()
    def _check_advanced_spectrum(self, audio_np: np.ndarray) -> dict[str, float]:
        tensor = torch.from_numpy(audio_np).float().unsqueeze(0).to(self.device)
        window, freq_indices = self._get_stft_buffers(513)

        X = torch.stft(tensor, n_fft=1024, hop_length=256, window=window, return_complex=True)
        mag = torch.abs(X)

        bin_hz = SR / 1024.0
        start_bin = int(512 * (6000 / (SR / 2)))
        hf_mag = mag[:, start_bin:, :]
        g_mean = torch.exp(torch.mean(torch.log(hf_mag + 1e-10), dim=1))
        a_mean = torch.mean(hf_mag, dim=1)
        flatness = g_mean / (a_mean + 1e-10)

        # Полоса голосовых формант 300-3400 Гц
        formant_lo = int(FORMANT_BAND_LOW_HZ / bin_hz)
        formant_hi = int(FORMANT_BAND_HIGH_HZ / bin_hz)
        formant_mag = mag[:, formant_lo:formant_hi, :]
        f_g_mean = torch.exp(torch.mean(torch.log(formant_mag + 1e-10), dim=1))
        f_a_mean = torch.mean(formant_mag, dim=1)
        formant_flatness = f_g_mean / (f_a_mean + 1e-10)

        # Отношение энергии верхних формант F2/F3 (1400-4000 Гц) к нижней базе (80-1400 Гц)
        low_band_e = torch.sum(mag[:, int(80 / bin_hz) : int(1400 / bin_hz), :], dim=1)
        upper_formant_e = torch.sum(mag[:, int(1400 / bin_hz) : int(4000 / bin_hz), :], dim=1)

        cum_energy = torch.cumsum(mag, dim=1)
        total_energy = cum_energy[:, -1:, :]
        mag_sum = total_energy.squeeze(1)

        rolloff_mask = cum_energy >= (0.85 * total_energy)
        rolloff_bins = torch.argmax(rolloff_mask.to(torch.uint8), dim=1).float() * bin_hz

        mean_freq = torch.sum(freq_indices * mag, dim=1) / (mag_sum + 1e-10)
        var_freq = torch.sum(((freq_indices - mean_freq.unsqueeze(1)) ** 2) * mag, dim=1) / (mag_sum + 1e-10)

        energy_thresh = torch.max(mag) * 1e-3
        active_frames = mag_sum > energy_thresh
        zero = torch.tensor(0.0, device=self.device)

        flat_valid = flatness[active_frames]
        if flat_valid.numel() > 1:
            med_flat = torch.median(flat_valid)
            iqr_flat = torch.quantile(flat_valid, 0.75) - torch.quantile(flat_valid, 0.25)
        elif flat_valid.numel() == 1:
            med_flat = flat_valid[0]
            iqr_flat = zero
        else:
            med_flat = zero
            iqr_flat = zero

        formant_valid = formant_flatness[active_frames]
        med_formant = torch.median(formant_valid) if formant_valid.numel() > 0 else zero

        ubr_frames = (upper_formant_e / (low_band_e + 1e-10))[active_frames]
        med_ubr = torch.median(ubr_frames) if ubr_frames.numel() > 0 else zero

        # Кепстральная выраженность основного тона (CPP: 80..440 Гц)
        cepstrum = torch.fft.irfft(torch.log(mag + 1e-7), n=1024, dim=1)
        pitch_cep = cepstrum[:, 36:201, :]
        cpp = torch.max(pitch_cep, dim=1).values - torch.mean(pitch_cep, dim=1)
        cpp_active = cpp[active_frames]
        med_cpp = torch.median(cpp_active) if cpp_active.numel() > 0 else zero

        med_roll = torch.median(rolloff_bins)
        med_var = torch.median(var_freq)

        vals = torch.stack([
            med_flat, med_roll, med_var, iqr_flat, med_formant, med_ubr, med_cpp
        ]).cpu().tolist()

        return {
            "flatness": float(vals[0]),
            "rolloff": float(vals[1]),
            "spec_var": float(vals[2]),
            "iqr_flat": float(vals[3]),
            "formant_flat": float(vals[4]),
            "upper_band_ratio": float(vals[5]),
            "cpp": float(vals[6]),
        }

    def analyze(self, path: str, spoof_threshold: float = 0.5) -> dict:
        t0 = time.perf_counter()

        audio = load_audio(path)
        segments = self.vad.speech_segments(audio)

        if not segments:
            return self._build_result(path, segments, 0.0, "Нет речи", spoof_threshold, t0)

        max_samples = SR * 15
        valid_chunks = []
        collected = 0
        for s, e in segments:
            start_idx, end_idx = int(s * SR), int(e * SR)
            seg_len = end_idx - start_idx
            if seg_len >= 512:
                need = max_samples - collected
                chunk = audio[start_idx : start_idx + min(seg_len, need)]
                valid_chunks.append(chunk)
                collected += len(chunk)
                if collected >= max_samples:
                    break

        if not valid_chunks:
            return self._build_result(path, segments, 0.0, "Недостаточно данных", spoof_threshold, t0)

        speech_audio = np.concatenate(valid_chunks)

        dsp = self._check_advanced_spectrum(speech_audio)
        w2v2_score = self.ast_w2v2.predict(speech_audio)

        flatness = dsp["flatness"]
        rolloff = dsp["rolloff"]
        spec_var = dsp["spec_var"]
        iqr_flat = dsp["iqr_flat"]
        formant_flat = dsp["formant_flat"]
        ubr = dsp["upper_band_ratio"]

        ext = os.path.splitext(path)[1].lower()
        lossy_compressed = ext in LOSSY_EXTS

        # 1. Абсолютная уверенность нейросети
        if w2v2_score >= 0.985:
            final_spoof_score = w2v2_score
            dsp_reason = "Абсолютный приоритет ML"

        # 2. Деградация спектрального среза и верхних формант F2/F3 (перехват глухого .ogg синтеза)
        elif rolloff < 1400.0 or (ubr < 0.04 and w2v2_score > 0.25):
            final_spoof_score = max(w2v2_score, 0.95)
            dsp_reason = "Деградация среза/формант"

        # 3. Коридор естественной плоскостности ВЧ (0.30..0.48)
        elif 0.30 <= flatness <= 0.48:
            vocal_tract_alive = (
                iqr_flat >= IQR_FLATNESS_MIN
                and FORMANT_FLATNESS_MIN <= formant_flat <= FORMANT_FLATNESS_MAX
            )
            shield_allowed = vocal_tract_alive and not lossy_compressed
            is_f5_anomaly = rolloff < 2200.0 and spec_var > 3800.0

            if w2v2_score > 0.97 and is_f5_anomaly:
                final_spoof_score = w2v2_score
                dsp_reason = "Приоритет ML (F5-синтез)"
            elif not shield_allowed and w2v2_score >= spoof_threshold:
                final_spoof_score = w2v2_score
                dsp_reason = "Лазейка .ogg/ВЧ закрыта"
            else:
                final_spoof_score = min(w2v2_score, spoof_threshold - 0.1)
                dsp_reason = "Физика микрофона (Щит ВЧ)"

        # 4. Явные ВЧ-аномалии вокодеров
        elif flatness > 0.55:
            final_spoof_score = 0.99
            dsp_reason = "Аномалия ВЧ (Сглаживание)"
        elif flatness < 0.20:
            final_spoof_score = 0.99
            dsp_reason = "Аномалия ВЧ (Артефакты)"

        # 5. Пограничная зона спектра
        else:
            final_spoof_score = w2v2_score
            dsp_reason = "Базовый ML-анализ"

        return self._build_result(
            path,
            segments,
            final_spoof_score,
            dsp_reason,
            spoof_threshold,
            t0,
            model_score=w2v2_score,
            dsp=dsp,
        )

    def _build_result(
        self,
        path: str,
        segments: list,
        spoof_score: float,
        reason: str,
        threshold: float,
        t0: float,
        model_score: float = 0.0,
        dsp: dict[str, float] | None = None,
    ) -> dict:
        d = dsp or {}
        return {
            "file": path,
            "duration": sum(e - s for s, e in segments) if segments else 0.0,
            "num_segments": len(segments),
            "spoof_score": spoof_score,
            "verdict": "spoof" if spoof_score > threshold else "bonafide",
            "dsp_reason": reason,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "model_score": model_score,
            "flatness": d.get("flatness", 0.0),
            "rolloff_hz": d.get("rolloff", 0.0),
            "spectrum_variance": d.get("spec_var", 0.0),
            "iqr_flatness": d.get("iqr_flat", 0.0),
            "formant_flatness": d.get("formant_flat", 0.0),
            "upper_band_ratio": d.get("upper_band_ratio", 0.0),
            "cpp": d.get("cpp", 0.0),
        }