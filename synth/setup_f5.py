"""Download pinned F5 weights; optionally transcribe the team's reference once.

Downloads use curl with resume and verify Hugging Face's SHA-256 afterwards.
Run from synth/: python setup_f5.py [--transcribe-reference]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import get_hf_file_metadata, hf_hub_url

F5_REVISION = "ea166adeae4c80ec5ee423a671e2bdb83906cf84"
WHISPER_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"
VOCOS_REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(repo: str, revision: str, files: list[str], directory: Path) -> None:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl не найден в PATH")
    records = []
    for filename in files:
        url = hf_hub_url(repo, filename, revision=revision)
        metadata = get_hf_file_metadata(url)
        path = directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        # LFS etags are SHA-256; ordinary text files have Git blob SHA-1.
        expected = metadata.etag if len(metadata.etag or "") == 64 else None
        complete = path.is_file() and path.stat().st_size == metadata.size
        if complete and expected:
            complete = digest(path) == expected
        if not complete:
            print(f"Загрузка: {repo}/{filename}", flush=True)
            subprocess.run([
                curl, "-L", "--fail", "--retry", "5", "--retry-all-errors",
                "--connect-timeout", "20", "-C", "-", "-o", str(path), url,
            ], check=True)
        if path.stat().st_size != metadata.size:
            raise RuntimeError(f"Неполный файл: {path}")
        actual = digest(path)
        if expected and actual != expected:
            raise RuntimeError(f"SHA-256 не совпал: {path}; повторно скачай этот файл")
        records.append({"file": filename, "size": metadata.size, "sha256": actual})
    (directory / "source.json").write_text(json.dumps({
        "model_id": repo, "revision": revision, "files": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def transcribe_reference() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    directory = Path("models/whisper-base")
    download("openai/whisper-base", WHISPER_REVISION, [
        "config.json", "generation_config.json", "preprocessor_config.json",
        "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "added_tokens.json", "normalizer.json", "merges.txt", "vocab.json",
        "model.safetensors",
    ], directory)
    processor = WhisperProcessor.from_pretrained(directory, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(
        directory, local_files_only=True,
    ).to(device).eval()
    reference = Path("voices/speaker_ru_1.wav")
    audio, sr = sf.read(reference)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
    with torch.inference_mode():
        tokens = model.generate(
            inputs.input_features.to(device), attention_mask=inputs.attention_mask.to(device),
            language="ru", task="transcribe", num_beams=5,
        )
    text = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
    if not text:
        raise RuntimeError("Whisper не распознал образец")
    output = reference.with_suffix(".txt")
    if output.exists():
        raise FileExistsError(f"Расшифровка уже существует: {output}")
    output.write_text(text + "\n", encoding="utf-8")
    print(f"ASR-расшифровка (проверь на слух): {text}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcribe-reference", action="store_true")
    args = parser.parse_args()
    download("Misha24-10/F5-TTS_RUSSIAN", F5_REVISION, [
        "F5TTS_v1_Base/vocab.txt",
        "F5TTS_v1_Base/model_240000_inference.safetensors",
    ], Path("models/f5-russian"))
    download("charactr/vocos-mel-24khz", VOCOS_REVISION,
             ["config.yaml", "pytorch_model.bin"], Path("models/vocos-mel-24khz"))
    if args.transcribe_reference:
        transcribe_reference()


if __name__ == "__main__":
    main()
