"""Render combined F5 effects from existing raw audio; no model inference."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
import soundfile as sf

from compare_f5 import SHELF, run_ffmpeg
from synth.channel import ffmpeg_path
from synth.manifest import now_iso, sha256_file
from synth.quality import REFLECTIONS, diagnostics, early_reflections


LABELS = {'dry': 'Исходный звук', 'soft_hf': 'Мягкие высокие частоты',
          'reflections': 'Ранние отражения', 'combined': 'Обе обработки вместе'}


def loudness(path: Path) -> dict:
    cmd = [ffmpeg_path(), '-hide_banner', '-nostats', '-i', str(path),
           '-af', 'loudnorm=I=-23:LRA=7:TP=-2:print_format=json', '-f', 'null', '-']
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Some ffmpeg versions append progress text after the measurement JSON.
    data, _ = json.JSONDecoder().raw_decode(result.stderr[result.stderr.rfind('{'):])
    return {'integrated_lufs': float(data['input_i']),
            'true_peak_dbfs': float(data['input_tp']), 'command': cmd}


def render_matched(source: Path, output: Path) -> dict:
    measured = loudness(source)
    if not np.isfinite(measured['integrated_lufs']):
        raise ValueError(f'невозможно выровнять громкость: {source}')
    gain_db = min(-23 - measured['integrated_lufs'], -2 - measured['true_peak_dbfs'])
    x, sr = sf.read(source, dtype='float32')
    y = x * 10 ** (gain_db / 20)
    if not np.isfinite(y).all() or np.max(np.abs(y)) >= 1:
        raise ValueError(f'перегрузка после усиления: {source}')
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, y, sr, subtype='PCM_16')
    after = loudness(output)
    if after['true_peak_dbfs'] > -1.9:
        raise ValueError(f'превышен предел пика: {output}')
    return {'input_loudness': measured, 'gain_db': gain_db,
            'output_loudness': after, 'diagnostics': diagnostics(y, sr)}


def write_page(out: Path, groups: list[dict]):
    parts = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>F5 — мягкие ВЧ и отражения вместе</title>',
             '<style>body{font:17px system-ui;background:#101722;color:#e5ecf4;'
             'max-width:1120px;margin:36px auto;padding:0 24px}p{line-height:1.6}'
             'a{color:#96d4ff}nav{display:flex;gap:18px;flex-wrap:wrap}h2{margin-top:44px}'
             '.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}'
             'article{background:#1b2738;padding:18px;border-radius:12px}'
             'article.combined{outline:1px solid #83bce8}small{color:#b1c1d6}'
             'audio{display:block;width:100%;margin-top:14px}'
             '@media(max-width:650px){.grid{grid-template-columns:1fr}}</style>',
             '<h1>Мягкие ВЧ + ранние отражения</h1>',
             '<p>В каждом блоке четыре версии одной и той же записи. Повторного '
             'синтеза нет. Громкость выровнена постоянным усилением; динамика фразы '
             'сохранена. Сначала сравните «Исходный звук» и «Обе обработки вместе», '
             'затем отдельные эффекты.</p>',
             '<nav><a href="#f5_ru">Основной голос</a><a href="#f5_matv">Голос matv</a></nav>',
             '<p><small>Слушайте при одной громкости в наушниках: шум на гласных, '
             'чёткость «с/ш/ф», окраска и ощущение пространства. Отражения не меняют '
             'интонацию исполнения. Мягкие ВЧ могут также приглушать согласные. '
             'Дефекты произношения исходной записи этой обработкой не исправляются.</small></p>']
    for voice, title in [('f5_ru', 'Основной голос'), ('f5_matv', 'Голос matv')]:
        parts.append(f'<h2 id="{voice}">{title}</h2>')
        for g in groups:
            if g['voice'] != voice:
                continue
            series = 'Штатная длительность — r10' if g['source_run'].startswith('r10') else 'С запасом на окончания — r11'
            parts.append(f'<h3>{series} · {html.escape(g["utt_id"])}</h3>'
                         f'<p>{html.escape(g["text"])}</p><div class="grid">')
            for key in ['dry', 'combined', 'soft_hf', 'reflections']:
                r = g['variants'][key]
                rel = Path(r['path']).resolve().relative_to(out.resolve()).as_posix()
                parts.append(f'<article class="{key}"><b>{LABELS[key]}</b><br>'
                             f'<small>{r["output_loudness"]["integrated_lufs"]:.1f} LUFS · '
                             f'{r["duration_sec"]:.2f} с · 24 кГц</small>'
                             f'<audio controls preload="metadata" aria-label="{title}, {g["utt_id"]}, {LABELS[key]}" '
                             f'src="{html.escape(rel, quote=True)}"></audio></article>')
            parts.append('</div>')
    parts.append('<p><small>Все варианты: CFG 1,5, 48 шагов, seed 1337. '
                 'В r11 распознаватель пропускал заключительный вопрос matv, '
                 'поэтому сохранена и серия r10. Проверяйте окончания на слух. '
                 'Измеренное ослабление высоких частот не доказывает исчезновение шума.</small></p></html>')
    (out / 'listen.html').write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, nargs='+', default=[
        Path('runs/r10-f5-quality-20261015'), Path('runs/r11-f5-quality-tail-20261015')])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    ffmpeg_path()
    selected = []
    for source_run in args.sources:
        for line in (source_run / 'comparison.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r['profile'] == 'cfg1_5' and r['effect'] == 'dry' and r['channel'] == 'native_24k':
                raw = Path(r['raw_path'])
                if sha256_file(raw) != r['raw_sha256']:
                    raise ValueError(f'изменился исходный WAV: {raw}')
                selected.append((source_run, r))
    if not selected:
        raise ValueError('нет исходных записей cfg1_5')
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.out / 'combine_f5.source.py')
    shutil.copyfile('synth/quality.py', args.out / 'quality.source.py')
    metadata = {'created_at': now_iso(), 'status': 'running', 'python': platform.python_version(),
                'new_inference': False, 'source_profile': 'cfg1_5', 'shelf_filter': SHELF,
                'reflection_wet': .08, 'reflection_taps': REFLECTIONS,
                'order': 'soft_hf -> reflections -> constant loudness gain',
                'target_lufs': -23, 'true_peak_ceiling_dbfs': -2,
                'source_hashes': {str(p): sha256_file(p) for p in
                                  [Path(__file__), Path('compare_f5.py'), Path('synth/quality.py')]},
                'input_indices': {str(p / 'comparison.jsonl'): sha256_file(p / 'comparison.jsonl')
                                  for p in args.sources}}
    def save_meta():
        (args.out / 'experiment.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    save_meta()
    groups = []
    for source_run, r in selected:
        print(source_run.name, r['voice'], r['utt_id'], flush=True)
        folder = args.out / r['voice'] / source_run.name / r['utt_id']
        folder.mkdir(parents=True)
        raw = folder / 'dry.float.wav'
        shutil.copyfile(r['raw_path'], raw)
        soft = folder / 'soft_hf.float.wav'
        shelf_command = run_ffmpeg(raw, soft, SHELF, 24000, 'pcm_f32le')
        x, sr = sf.read(raw, dtype='float32')
        xs, soft_sr = sf.read(soft, dtype='float32')
        assert sr == soft_sr == 24000 and len(x) == len(xs)
        sources = {'dry': raw, 'soft_hf': soft}
        for key, signal in [('reflections', x), ('combined', xs)]:
            target = folder / f'{key}.float.wav'
            sf.write(target, early_reflections(signal, sr), sr, subtype='FLOAT')
            sources[key] = target
        group = {key: r[key] for key in ['voice', 'utt_id', 'text', 'generate_params', 'seed']}
        group.update(source_run=source_run.name, original_path=r['raw_path'],
                     original_sha256=r['raw_sha256'], shelf_command=shelf_command, variants={})
        for key, source in sources.items():
            output = folder / f'{key}.wav'
            rendered = render_matched(source, output)
            info = sf.info(output)
            expected = len(x) + (round(.073 * sr) if key in ('reflections', 'combined') else 0)
            if info.frames != expected or info.channels != 1 or info.samplerate != 24000:
                raise ValueError(f'некорректная длина или формат: {output}')
            final, final_sr = sf.read(output, dtype='float32')
            if not np.isfinite(final).all() or np.max(np.abs(final)) >= 1:
                raise ValueError(f'некорректный результат: {output}')
            group['variants'][key] = {**rendered, 'path': output.as_posix(),
                                      'sha256': sha256_file(output), 'duration_sec': info.duration,
                                      'processed_source': source.as_posix(),
                                      'processed_source_sha256': sha256_file(source),
                                      'file_diagnostics': diagnostics(final, final_sr)}
        values = [v['output_loudness']['integrated_lufs'] for v in group['variants'].values()]
        if max(values) - min(values) > .15:
            raise ValueError(f'громкость вариантов не совпала: {values}')
        groups.append(group)
        (args.out / 'comparison.json').write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding='utf-8')
        write_page(args.out, groups)
    metadata.update(status='complete', source_recordings=len(groups), listening_files=len(groups)*4,
                    measured_lufs_min=min(v['output_loudness']['integrated_lufs'] for g in groups for v in g['variants'].values()),
                    measured_lufs_max=max(v['output_loudness']['integrated_lufs'] for g in groups for v in g['variants'].values()))
    save_meta()
    print(f'Готово: {len(groups)*4} записей. {args.out / "listen.html"}', flush=True)


if __name__ == '__main__':
    main()
