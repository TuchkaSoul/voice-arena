import hashlib
import json
from pathlib import Path

from provenance import collect, sha256_file


def test_hash_is_for_file_bytes_and_changes_after_edit(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"old recording")
    assert sha256_file(path) == hashlib.sha256(b"old recording").hexdigest()
    path.write_bytes(b"new recording")
    assert sha256_file(path) != hashlib.sha256(b"old recording").hexdigest()


def test_model_and_vad_fingerprints_can_be_compared_without_loading_models(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"test weights")
    (model_dir / "config.json").write_text(json.dumps({"id2label": {"0": "real", "1": "fake"}}))
    (model_dir / "unrelated.txt").write_text("not a model file")
    vad = tmp_path / "vad.onnx"
    vad.write_bytes(b"vad")
    ecapa = tmp_path / "ecapa.onnx"
    ecapa.write_bytes(b"ecapa")
    first = collect(model_dir, vad, ecapa)
    assert first["schema"] == "voice-arena/analyzer-provenance@1"
    assert {Path(row["path"]).name for row in first["model_files"]} == {"config.json", "model.safetensors"}
    assert first["vad"]["sha256"] == hashlib.sha256(b"vad").hexdigest()
    assert "torch" in first["versions"] and "transformers" in first["versions"]
    assert first["source_files"]
    weights.write_bytes(b"other weights")
    second = collect(model_dir, vad, ecapa)
    assert first["model_files"] != second["model_files"]
    assert first["source_files"] == second["source_files"]
