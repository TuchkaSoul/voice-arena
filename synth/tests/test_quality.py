import numpy as np
import pytest

from synth.quality import REFLECTIONS, diagnostics, early_reflections


def test_reflection_impulse_preserves_direct_sound_and_full_tail():
    audio = np.array([1.0], dtype=np.float32)
    result = early_reflections(audio, 24000, wet=.08)
    assert len(result) == 1 + round(REFLECTIONS[-1][0] * 24000)
    assert result[0] == pytest.approx(1 / 1.08)
    for delay, gain in REFLECTIONS:
        assert result[round(delay * 24000)] == pytest.approx(.08 * gain / 1.08)
    assert np.sum(result) == pytest.approx(1)


def test_reflections_add_no_noise_and_keep_headroom():
    assert not np.any(early_reflections(np.zeros(1000, np.float32), 24000))
    audio = np.ones(24000, dtype=np.float32)
    assert np.max(np.abs(early_reflections(audio, 24000))) <= 1.000001
    assert np.array_equal(early_reflections(audio, 24000, wet=0), audio)


@pytest.mark.parametrize('wet', [-.1, 1.1, float('nan')])
def test_reflections_reject_invalid_weights(wet):
    with pytest.raises(ValueError):
        early_reflections(np.ones(10), 24000, wet=wet)


def test_diagnostics_do_not_call_silence_a_noise_estimate():
    result = diagnostics(np.zeros(24000), 24000)
    assert result['active_hf_6_8khz_energy_ratio'] is None
    assert result['energy_active_fraction'] == 0
    assert all(np.isfinite(v) for v in result.values() if v is not None)


def test_diagnostics_distinguish_high_frequency_energy():
    t = np.arange(24000) / 24000
    low = diagnostics(.2 * np.sin(2 * np.pi * 1000 * t), 24000)
    high = diagnostics(.2 * np.sin(2 * np.pi * 7000 * t), 24000)
    assert low['active_hf_6_8khz_energy_ratio'] < .001
    assert high['active_hf_6_8khz_energy_ratio'] > .99
