"""Offline ASR and waveform checks on generated speech before playback padding."""
import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
import soundfile as sf

from synth.manifest import sha256_file, now_iso


def tokens(value):
    return re.findall(r"[а-яa-z0-9]+", value.lower().replace("ё", "е"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for key, value in {"USE_TF": "0", "USE_TORCH": "1", "HF_HUB_OFFLINE": "1"}.items():
        os.environ.setdefault(key, value)
    import torch
    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_dir = Path("models/whisper-base")
    processor = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    model_source = json.loads((model_dir/"source.json").read_text(encoding="utf-8"))
    model_weights_hash = sha256_file(model_dir/"model.safetensors")
    rows = []
    for source in sources:
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"source changed: {path}")
        audio, sr = sf.read(path, dtype="float32")
        resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        inputs = processor(resampled, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            output = model.generate(inputs.input_features.to(device),
                                    attention_mask=inputs.attention_mask.to(device),
                                    language="ru", task="transcribe", num_beams=5, do_sample=False)
        transcription = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
        expected = tokens(source["text"])[-2:]
        actual = tokens(transcription)
        found = any(actual[i:i+len(expected)] == expected for i in range(max(0, len(actual)-6), len(actual)))
        n = max(1, round(sr * .02))
        framed = audio[:len(audio)//n*n].reshape(-1, n)
        rms = np.sqrt(np.mean(framed.astype(np.float64)**2, axis=1))
        threshold = max(10**(-45/20), float(np.max(rms)) * .025)
        active = np.flatnonzero(rms > threshold)
        margin = len(audio)/sr - (int(active[-1])+1)*.02 if len(active) else len(audio)/sr
        tail = float(np.sqrt(np.mean(audio[-round(.1*sr):].astype(np.float64)**2)))
        row = dict(source_id=source["source_id"], sha256=source["sha256"], text=source["text"],
                   transcription=transcription, expected_ending=" ".join(expected),
                   ending_recognized=found, duration_sec=len(audio)/sr,
                   tail_100ms_rms_dbfs=20*float(np.log10(max(tail, 1e-12))),
                   energy_tail_margin_sec=margin,
                   caveat="ASR and energy checks support review; they do not replace listening.")
        rows.append(row)
        report = dict(created_at=now_iso(), sources_sha256=sha256_file(args.sources),
                      script_sha256=sha256_file(Path(__file__)),
                      model_source=model_source,
                      model_weights_sha256=model_weights_hash,
                      num_beams=5, rows=rows,
                      all_endings_recognized=all(r["ending_recognized"] for r in rows))
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f'{source["source_id"]}: ending={found}; tail={margin:.3f}s; {transcription}', flush=True)


if __name__ == "__main__":
    main()
