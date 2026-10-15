"""Diagnostics and an explicitly simulated early-reflection listening variant.

These measurements are not SNR or a naturalness score. In particular, high
frequency energy also includes real fricatives and aspiration.
"""
from __future__ import annotations

import numpy as np


REFLECTIONS = ((.011, .26), (.017, .20), (.023, .16), (.031, .13),
               (.043, .10), (.059, .09), (.073, .06))


def early_reflections(audio: np.ndarray, sr: int, wet: float = .08) -> np.ndarray:
    """Add weak delayed copies, preserving the complete tail; no added noise.

    wet is the sum of reflected amplitudes relative to direct sound. This is
    a small FIR room approximation, not a measured room impulse response.
    """
    if sr <= 0 or not np.isfinite(wet) or not 0 <= wet <= 1:
        raise ValueError("sr должен быть положительным, wet — от 0 до 1")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("ожидался непустой конечный моно-сигнал")
    if wet == 0:
        return audio.copy()
    taps = [(max(1, round(delay * sr)), gain) for delay, gain in REFLECTIONS]
    result = np.zeros(len(audio) + taps[-1][0], dtype=np.float32)
    result[:len(audio)] = audio
    for delay, gain in taps:
        result[delay:delay + len(audio)] += wet * gain * audio
    # Convex sum: keep headroom without a compressor or a nonlinear limiter.
    return result / (1 + wet)


def diagnostics(audio: np.ndarray, sr: int) -> dict:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not len(audio) or sr <= 0 or not np.isfinite(audio).all():
        raise ValueError("ожидался непустой конечный моно-сигнал и sr > 0")
    n = max(1, round(.02 * sr))
    usable = audio[:len(audio) // n * n]
    frames = usable.reshape(-1, n) if len(usable) else audio[None, :]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    # Energy-based activity proxy, NOT VAD; explicitly labelled in the report.
    active = rms > max(10 ** (-45 / 20), float(np.max(rms)) * .05)
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), axis=1)) ** 2
    freqs = np.fft.rfftfreq(frames.shape[1], 1 / sr)
    band = (freqs >= 6000) & (freqs < min(8000, sr / 2))
    selected = spectrum[active]
    ratio = (float(selected[:, band].sum() / max(selected.sum(), 1e-30))
             if len(selected) and band.any() else None)
    def db(value):
        return round(float(20 * np.log10(max(float(value), 1e-12))), 3)
    return {
        "sr": sr, "duration_sec": round(len(audio) / sr, 6),
        "peak": float(np.max(np.abs(audio))),
        "overload_fraction": float(np.mean(np.abs(audio) > 1)),
        "rail_fraction": float(np.mean(np.abs(audio) >= .999)),
        "rms_dbfs": db(np.sqrt(np.mean(audio ** 2))),
        "rms_20ms_p10_dbfs": db(np.quantile(rms, .1)),
        "rms_20ms_p90_dbfs": db(np.quantile(rms, .9)),
        "energy_active_fraction": float(np.mean(active)),
        "active_hf_6_8khz_energy_ratio": ratio,
        "tail_20ms_rms_dbfs": db(np.sqrt(np.mean(audio[-n:] ** 2))),
    }
