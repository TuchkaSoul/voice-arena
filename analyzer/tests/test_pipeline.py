"""Smoke-тест: пайплайн должен отработать на реальном файле и вернуть валидный dict."""
import os
import pytest

from analyzer.pipeline import Analyzer

SAMPLE = "audio/sample.wav"

MODELS = dict(
    vad_path="models/silero_vad.onnx",
    aasist_path="models/aasist.onnx",
    ecapa_path="models/voxceleb_ECAPA512_LM.onnx",
)


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="нет audio/sample.wav")
def test_smoke():
    analyzer = Analyzer(**MODELS, device="cpu")
    result = analyzer.analyze(SAMPLE)

    assert 0.0 <= result["spoof_score"] <= 1.0
    assert result["verdict"] in ("bonafide", "spoof")
    assert result["num_segments"] >= 0
    assert result["duration"] > 0
    assert result["latency_ms"] > 0
    assert result["embedding"] is None or result["embedding"].ndim == 1