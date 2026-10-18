"""New phrases/seeds for channel validation, with unchanged F5 presets.

Run from synth/ in .venv-f5. No detector scores are read by this script.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil

import numpy as np
import soundfile as sf

from synth import manifest, presets, text

PHRASES = {
    "V0001": "Сегодня утром мы проверили запись ещё раз. Голос слышно отчётливо, а в конце остаётся короткая пауза.",
    "V0002": "Пожалуйста, передай мне синюю папку со стола. В ней лежат фотографии и список вопросов к следующей встрече.",
    "V0003": "На улице шумит дождь, поэтому окно лучше закрыть. Когда погода изменится, мы выйдем на короткую прогулку.",
    "V0004": "Сначала сохраним исходные файлы, затем сравним результаты. Если появится ошибка, повторим проверку с самого начала.",
}
SEEDS = (2207, 7811)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--reserve-sec", type=float, default=0,
                        help="extra actual generation time, before adding any playback silence")
    args = parser.parse_args()
    if not np.isfinite(args.reserve_sec) or args.reserve_sec < 0:
        parser.error("--reserve-sec must be finite and nonnegative")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / Path(__file__).name)
    for key, value in {"USE_TF": "0", "USE_TORCH": "1", "HF_HUB_OFFLINE": "1"}.items():
        os.environ.setdefault(key, value)
    import torch
    from f5_tts.api import F5TTS
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text, chunk_text

    selected = [presets.load(name) for name in ("f5_ru", "f5_matv")]
    params = selected[0].engine_params
    metadata = dict(created_at=manifest.now_iso(), status="planned", phrases=PHRASES,
                    seeds=args.seeds, reserve_sec=args.reserve_sec,
                    voices={p.name: p.engine_params for p in selected},
                    python=platform.python_version(), device="cuda" if torch.cuda.is_available() else "cpu",
                    versions={name: importlib.metadata.version(name) for name in
                              ("torch", "f5-tts", "vocos", "soundfile", "numpy")},
                    hashes={str(path): manifest.sha256_file(path) for path in
                            [Path(__file__), Path(params["checkpoint"]), Path(params["vocab"]),
                             Path(params["vocoder_path"]) / "pytorch_model.bin",
                             *[Path(p.source) for p in selected],
                             *[Path(p.engine_params["speaker_wav"]) for p in selected],
                             *[Path(p.engine_params["ref_text_file"]) for p in selected]]})

    def save_meta():
        (out / "generation.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    save_meta()
    model = F5TTS(model=params["model"], ckpt_file=params["checkpoint"],
                  vocab_file=params["vocab"], vocoder_local_path=params["vocoder_path"],
                  device=metadata["device"])
    metadata["status"] = "running"
    save_meta()
    sources = []
    vocab = set(Path(params["vocab"]).read_text(encoding="utf-8").splitlines())
    for preset in selected:
        ep = preset.engine_params
        reference_text = Path(ep["ref_text_file"]).read_text(encoding="utf-8-sig").strip()
        if args.reserve_sec:
            processed_ref, processed_text = preprocess_ref_audio_text(ep["speaker_wav"], reference_text)
            ref_info = sf.info(processed_ref)
            ref_frames = math.ceil(ref_info.frames * 24000 / ref_info.samplerate) // 256
        for seed in args.seeds:
            for utt_id, phrase in PHRASES.items():
                normalized = text.normalize(phrase, **preset.text_params)
                if set(normalized) - vocab:
                    raise ValueError("unknown characters")
                generate_params = dict(ep["generate_params"], seed=seed)
                if args.reserve_sec:
                    max_chars = int(len(processed_text.encode("utf-8")) /
                                    (ref_info.frames / ref_info.samplerate) *
                                    (22 - ref_info.frames / ref_info.samplerate))
                    if len(chunk_text(normalized, max_chars=max_chars)) != 1:
                        raise ValueError("duration reserve requires a single text chunk")
                    gen_frames = int(ref_frames / len(processed_text.encode("utf-8")) *
                                     len(normalized.encode("utf-8")) / generate_params.get("speed", 1))
                    generate_params["fix_duration"] = (ref_frames + gen_frames) * 256 / 24000 + args.reserve_sec
                audio, sr, _ = model.infer(ref_file=ep["speaker_wav"], ref_text=reference_text,
                                           gen_text=normalized, **generate_params)
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
                    raise ValueError("invalid waveform")
                peak = float(np.max(np.abs(audio)))
                if peak > 1:
                    audio /= peak
                source_id = f"{preset.name}_{seed}_{utt_id}"
                path = out / (source_id + ".wav")
                sf.write(path, audio, sr, subtype="FLOAT")
                sources.append(dict(source_id=source_id, voice=preset.name, utt_id=utt_id,
                                    text=phrase, text_normalized=normalized, label="spoof",
                                    source_run=out.name, path=path.as_posix(), sha256=manifest.sha256_file(path),
                                    seed=seed, generate_params=generate_params, peak_before_scale=peak,
                                    duration_sec=len(audio) / sr))
                (out / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"Generated {source_id}: {len(audio)/sr:.2f}s", flush=True)
    metadata["status"] = "completed"
    metadata["files"] = len(sources)
    save_meta()


if __name__ == "__main__":
    main()
