import sys
from types import SimpleNamespace

import numpy as np
import pytest

from synth.engines.f5 import F5Engine


def parameters(tmp_path):
    files = {
        "checkpoint": "weights.safetensors", "vocab": "vocab.txt",
        "speaker_wav": "speaker.wav", "ref_text_file": "speaker.txt",
    }
    params = {key: str(tmp_path / name) for key, name in files.items()}
    for name in files.values():
        (tmp_path / name).touch()
    (tmp_path / "vocab.txt").write_text("\n".join(" Привет.молк+о"), encoding="utf-8")
    (tmp_path / "speaker.txt").write_text("Привет.", encoding="utf-8")
    (tmp_path / "config.yaml").touch()
    (tmp_path / "pytorch_model.bin").touch()
    return {**params, "vocoder_path": str(tmp_path), "device": "cpu"}


def test_explicit_russian_weights_transcript_and_stress(tmp_path, monkeypatch):
    calls = []

    class FakeTTS:
        def __init__(self, **kwargs):
            calls.append(("load", kwargs))

        def infer(self, **kwargs):
            calls.append(("infer", kwargs))
            return np.zeros(240, dtype=np.float64), 24000, None

    monkeypatch.setitem(sys.modules, "f5_tts", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "f5_tts.api", SimpleNamespace(F5TTS=FakeTTS))
    engine = F5Engine(parameters(tmp_path))
    result = engine.say("молок+о")
    engine.say("Привет.")
    assert [call[0] for call in calls] == ["load", "infer", "infer"]
    assert calls[0][1]["vocab_file"] == str(tmp_path / "vocab.txt")
    assert calls[0][1]["ckpt_file"] == str(tmp_path / "weights.safetensors")
    assert calls[1][1]["ref_text"] == "Привет."
    assert calls[1][1]["gen_text"] == "молок+о"
    assert result.sr == 24000 and result.audio.dtype == np.float32


def test_missing_weights_fail_before_import(tmp_path):
    params = parameters(tmp_path)
    (tmp_path / "weights.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        F5Engine(params).load()


def test_empty_reference_does_not_trigger_asr(tmp_path):
    params = parameters(tmp_path)
    (tmp_path / "speaker.txt").write_text("  ", encoding="utf-8")
    with pytest.raises(ValueError, match="пуста"):
        F5Engine(params).load()


def test_unsupported_characters_are_not_silently_discarded():
    engine = F5Engine({})
    engine._loaded = True
    engine._vocab = set("текст")
    with pytest.raises(ValueError, match="словаре"):
        engine.say("текст🙂")


def test_overloaded_waveform_is_scaled_without_flattening_peaks():
    engine = F5Engine({"generate_params": {"seed": 1337, "cfg_strength": 1.5}})
    engine._loaded = True
    engine._vocab = set("текст")
    calls = []

    def infer(**kwargs):
        calls.append(kwargs)
        return np.array([0, .5, 1, 2, -2], dtype=np.float32), 24000, None

    engine._tts = SimpleNamespace(infer=infer)
    result = engine.say("текст")
    np.testing.assert_allclose(result.audio, [0, .25, .5, 1, -1])
    assert calls[0]['seed'] == 1337
    assert calls[0]['cfg_strength'] == 1.5
