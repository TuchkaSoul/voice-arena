"""Версии и контрольные суммы для сравнения запусков, без загрузки нейросети."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256_file(path)}


def collect(model_path: Path, vad_path: Path, ecapa_path: Path) -> dict:
    source = Path(__file__).resolve().parent
    code = sorted((source / "analyzer").glob("*.py"))
    code += [source / name for name in ("main.py", "report.py", "provenance.py")]
    model_files = sorted(p for p in model_path.iterdir()
                         if p.is_file() and p.suffix in (".json", ".safetensors", ".bin", ".pt", ".pth"))
    versions = {}
    for name in ("torch", "torchaudio", "transformers", "safetensors", "numpy",
                 "onnxruntime", "onnxruntime-gpu", "librosa", "soundfile", "soxr"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema": "voice-arena/analyzer-provenance@1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "source_files": [fingerprint(p) for p in code],
        "model_files": [fingerprint(p) for p in model_files],
        "vad": fingerprint(vad_path),
        "ecapa": fingerprint(ecapa_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/wav2vec2-deepfake-voice-detector"))
    parser.add_argument("--vad-path", type=Path, default=Path("models/silero_vad.onnx"))
    parser.add_argument("--ecapa-path", type=Path, default=Path("models/voxceleb_ECAPA512.onnx"))
    parser.add_argument("--output", type=Path, default=Path("analyzer-provenance.json"))
    args = parser.parse_args()
    result = collect(args.model_path, args.vad_path, args.ecapa_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
