"""Controlled F5 listening experiment. Run from synth/ in .venv-f5.

No detector feedback is used to select a configuration. Raw floats, matched
loudness listening files, exact parameters and audio hashes are kept together.
"""
from __future__ import annotations

import argparse
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import shutil
import time

import numpy as np
import soundfile as sf

from synth import channel, manifest, presets, text
from synth.quality import REFLECTIONS, diagnostics, early_reflections


PHRASES = {
    "Q0001": "Подожди, я сейчас посмотрю. Да, всё верно: встречаемся завтра вечером. Ты ведь успеешь?",
    "Q0002": "Скажи-ка, дядя, ведь недаром. Стена разрушена Титаном. Ведь там была война.",
}
PROFILES = {
    "baseline": {"nfe_step": 32, "cfg_strength": 2.0},
    "steps48": {"nfe_step": 48, "cfg_strength": 2.0},
    "cfg1_5": {"nfe_step": 48, "cfg_strength": 1.5},
}
SHELF = "treble=g=-2.5:f=5000:t=q:w=0.707"
LOUDNESS = "loudnorm=I=-23:LRA=7:TP=-2"


def run_ffmpeg(source: Path, target: Path, filters: str, sr: int, subtype: str):
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [channel.ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(source), "-af", filters, "-ar", str(sr), "-ac", "1",
           "-c:a", subtype, str(target)]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return cmd


def write_page(out: Path, rows: list[dict]):
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
             '<title>F5: сравнение звучания</title><style>',
             'body{font:17px system-ui;max-width:1000px;margin:40px auto;padding:0 24px;'
             'background:#101722;color:#e5ecf4}h2{margin-top:38px}a{color:#8cd3ff}'
             'article{padding:16px;background:#1b2738;border-radius:12px;margin:12px 0}'
             'audio{width:100%;margin-top:8px}small{color:#acbdd2}p{line-height:1.6}</style>',
             '<h1>Сравнение звучания F5</h1>',
             '<p>Слушайте в одинаковых наушниках при одной громкости. '
             'Для каждой пары сначала сравните baseline и steps48, затем steps48 и cfg1_5. '
             'Обработки soft_hf и reflections сделаны из того же cfg1_5 без повторного синтеза.</p>',
             '<p>Оцените отдельно: шум на гласных, чёткость «с/ш/ф», интонацию, '
             'сходство тембра и окончания слов. Варианты пока экспериментальные. '
             'Ползунок громкости не показывает измеренную естественность.</p>']
    for voice in sorted({r['voice'] for r in rows}):
        for utt_id, phrase in PHRASES.items():
            parts.append(f'<h2>{html.escape(voice)} · {utt_id}</h2><p>{html.escape(phrase)}</p>')
            for r in rows:
                if r['voice'] != voice or r['utt_id'] != utt_id or r['channel'] != 'native_24k':
                    continue
                rel = Path(r['path']).resolve().relative_to(out.resolve()).as_posix()
                label = f"{r['profile']} · {r['effect']} · seed {r['seed']}"
                parts.append(f'<article><b>{html.escape(label)}</b><br>'
                             f'<small>24 кГц · {r["duration_sec"]:.2f} с · громкость −23 LUFS (цель)</small>'
                             f'<audio controls preload="none" src="{html.escape(rel, quote=True)}"></audio></article>')
    parts.append('<p>Сырые WAV, манифест и параметры находятся рядом. '
                 'Ранние отражения моделируются задержанными копиями, это не запись реальной комнаты. '
                 'Уменьшение ВЧ также затрагивает согласные и требует оценки на слух.</p></html>')
    (out / 'listen.html').write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--voices', nargs='+', default=['f5_ru', 'f5_matv'],
                        choices=['f5_ru', 'f5_matv'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1337])
    parser.add_argument('--reserve-sec', type=float, default=.65,
                        help='дополнительное время генерации для окончания фразы; 0 — штатная оценка')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.reserve_sec) or args.reserve_sec < 0:
        parser.error('--reserve-sec должен быть конечным и неотрицательным')
    # Fail before model loading or creating a partial run.
    channel.ffmpeg_path()
    selected = [presets.load(name) for name in args.voices]
    for p in selected:
        for key in ('checkpoint', 'vocab', 'speaker_wav', 'ref_text_file'):
            if not Path(p.engine_params[key]).is_file():
                raise FileNotFoundError(p.engine_params[key])
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.out / 'compare_f5.source.py')
    shutil.copyfile('synth/quality.py', args.out / 'quality.source.py')
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('USE_TORCH', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    import torch
    from f5_tts.api import F5TTS
    import f5_tts.api
    import f5_tts.infer.utils_infer
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text, chunk_text

    params = selected[0].engine_params
    # One model in memory; voice conditioning is supplied explicitly per call.
    model_keys = ('model', 'checkpoint', 'vocab', 'vocoder_path')
    for p in selected[1:]:
        if any(p.engine_params[k] != params[k] for k in model_keys):
            raise ValueError('этот эксперимент требует одинаковой модели для обоих голосов')
    experiment = {
        'created_at': manifest.now_iso(), 'status': 'running',
        'python': platform.python_version(), 'torch': torch.__version__,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'versions': {name: importlib.metadata.version(name)
                     for name in ('f5-tts', 'vocos', 'numpy', 'soundfile', 'torchaudio')},
        'phrases': PHRASES, 'profiles': PROFILES, 'seeds': args.seeds,
        'reserve_sec': args.reserve_sec,
        'soft_hf_filter': SHELF, 'reflection_wet': .08, 'reflection_taps': REFLECTIONS,
        'loudness_filter': LOUDNESS, 'voices': {},
        'source_hashes': {str(p): manifest.sha256_file(p) for p in
                          [Path(__file__), Path('synth/quality.py'),
                           Path(f5_tts.api.__file__),
                           Path(f5_tts.infer.utils_infer.__file__),
                           Path(params['checkpoint']), Path(params['vocab']),
                           Path(params['vocoder_path']) / 'config.yaml',
                           Path(params['vocoder_path']) / 'pytorch_model.bin']},
        'ffmpeg_version': subprocess.run([channel.ffmpeg_path(), '-version'],
                                         capture_output=True, text=True, check=True).stdout,
        'notes': 'Малый корпус и фиксированные seed: разведочная проверка, не доказательство улучшения. '
                 'Диагностика энергии не является SNR. Анализатор не выбирает настройки.',
    }
    metadata = args.out / 'experiment.json'
    def save_metadata():
        metadata.write_text(json.dumps(experiment, ensure_ascii=False, indent=2), encoding='utf-8')
    save_metadata()
    t0 = time.perf_counter()
    tts = F5TTS(model=params['model'], ckpt_file=params['checkpoint'],
                vocab_file=params['vocab'], vocoder_local_path=params['vocoder_path'],
                device='cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    manifest_rows = []
    with (args.out / 'comparison.jsonl').open('w', encoding='utf-8') as journal:
        for preset in selected:
            ep = preset.engine_params
            ref = Path(ep['speaker_wav'])
            ref_text = Path(ep['ref_text_file']).read_text(encoding='utf-8-sig').strip()
            vocab = set(Path(ep['vocab']).read_text(encoding='utf-8').splitlines())
            if not ref_text or set(ref_text) - vocab:
                raise ValueError('пустая расшифровка или неизвестные символы в образце')
            processed_ref, processed_text = preprocess_ref_audio_text(str(ref), ref_text)
            ref_info = sf.info(processed_ref)
            ref_frames = math.ceil(ref_info.frames * 24000 / ref_info.samplerate) // 256
            experiment['voices'][preset.name] = {
                'engine_params': ep, 'text_params': preset.text_params,
                'reference_sha256': manifest.sha256_file(ref),
                'reference_text_sha256': manifest.sha256_file(Path(ep['ref_text_file'])),
                'reference_diagnostics': diagnostics(sf.read(ref, dtype='float32')[0], sf.info(ref).samplerate),
            }
            save_metadata()
            for seed in args.seeds:
                for utt_id, phrase in PHRASES.items():
                    normalized = text.normalize(phrase, **preset.text_params)
                    if set(normalized) - vocab:
                        raise ValueError('неизвестные символы в тексте синтеза')
                    for profile, overrides in PROFILES.items():
                        gp = {**ep.get('generate_params', {}), **overrides, 'seed': seed}
                        gp.update(speed=1.0, remove_silence=False)
                        if args.reserve_sec:
                            # Match the installed F5 duration estimate, then add time.
                            # Our short probes must be single-chunk for this comparison.
                            max_chars = int(len(processed_text.encode('utf-8')) /
                                            (ref_info.frames / ref_info.samplerate) *
                                            (22 - ref_info.frames / ref_info.samplerate))
                            if len(chunk_text(normalized, max_chars=max_chars)) != 1:
                                raise ValueError('для проверки с запасом нужен односегментный текст')
                            gen_frames = int(ref_frames / len(processed_text.encode('utf-8')) *
                                             len(normalized.encode('utf-8')) / gp['speed'])
                            gp['fix_duration'] = (ref_frames + gen_frames) * 256 / 24000 + args.reserve_sec
                        folder = args.out / preset.name / f'seed{seed}' / utt_id / profile
                        folder.mkdir(parents=True)
                        print(f'{preset.name} {utt_id} {profile} seed={seed}', flush=True)
                        start = time.perf_counter()
                        wav, sr, _ = tts.infer(ref_file=str(ref), ref_text=ref_text,
                                             gen_text=normalized, **gp)
                        elapsed = time.perf_counter() - start
                        wav = np.asarray(wav, dtype=np.float32)
                        if wav.ndim != 1 or not len(wav) or not np.isfinite(wav).all():
                            raise ValueError('некорректный результат F5')
                        raw = folder / 'raw.wav'
                        sf.write(raw, wav, sr, subtype='FLOAT')
                        effects = {'dry': raw}
                        if profile == 'cfg1_5':
                            soft = folder / 'soft_hf.raw.wav'
                            run_ffmpeg(raw, soft, SHELF, sr, 'pcm_f32le')
                            room = folder / 'reflections.raw.wav'
                            sf.write(room, early_reflections(wav, sr), sr, subtype='FLOAT')
                            effects.update(soft_hf=soft, reflections=room)
                        for effect, source in effects.items():
                            data, source_sr = sf.read(source, dtype='float32')
                            source_metrics = diagnostics(data, source_sr)
                            native = folder / effect / 'native_24k' / f'{utt_id}.wav'
                            cmd = run_ffmpeg(source, native, LOUDNESS, 24000, 'pcm_s16le')
                            files = {'native_24k': native}
                            for name in ('clean', 'opus_24k'):
                                files[name] = channel.apply(source, folder / effect / name, name, utt_id)
                            for name, target in files.items():
                                final, final_sr = sf.read(target, dtype='float32')
                                row = {
                                    'voice': preset.name, 'utt_id': utt_id, 'profile': profile,
                                    'effect': effect, 'channel': name, 'seed': seed,
                                    'text': phrase, 'text_normalized': normalized,
                                    'generate_params': gp, 'inference_sec': elapsed,
                                    'path': target.as_posix(), 'sha256': manifest.sha256_file(target),
                                    'source_path': source.as_posix(),
                                    'source_sha256': manifest.sha256_file(source),
                                    'raw_path': raw.as_posix(), 'raw_sha256': manifest.sha256_file(raw),
                                    'raw_diagnostics': diagnostics(wav, sr),
                                    'source_diagnostics': source_metrics,
                                    'diagnostics': diagnostics(final, final_sr),
                                    'duration_sec': len(final) / final_sr,
                                    'native_render_command': cmd,
                                }
                                journal.write(json.dumps(row, ensure_ascii=False) + '\n')
                                journal.flush()
                                rows.append(row)
                                manifest_rows.append(manifest.ManifestRow(
                                    utt_id=utt_id, text=phrase, text_normalized=normalized,
                                    label='spoof', engine='f5',
                                    preset=f'{preset.name}__{profile}__{effect}__seed{seed}',
                                    channel=name, path=target.as_posix(), sr=final_sr, engine_sr=sr,
                                    duration_sec=round(len(final) / final_sr, 6), sha256=row['sha256'],
                                    run_id=args.out.name, seed=seed, created_at=manifest.now_iso(),
                                    corpus='phrases_quality_v1',
                                    engine_params={**ep, 'generate_params': gp,
                                                   'listening_effect': effect}))
                        manifest.write(args.out / 'manifest.jsonl', manifest_rows)
                        write_page(args.out, rows)
    experiment.update(status='complete', elapsed_sec=time.perf_counter() - t0,
                      files=len(rows), generations=len(selected) * len(args.seeds) * len(PHRASES) * len(PROFILES))
    save_metadata()
    print(f'Готово: {len(rows)} файлов. {args.out / "listen.html"}', flush=True)


if __name__ == '__main__':
    main()
