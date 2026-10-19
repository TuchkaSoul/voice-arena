"""Export the exact eight alternate-reference Opus inputs; never transform audio."""
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/r20-f5-reference-comparison-20261018-v2'
OUT = ROOT / 'runs/exports/voice-arena-winning-r20-20261019'
ARCHIVE = OUT.with_suffix('.zip')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(name):
    return json.loads((RUN / name).read_text(encoding='utf-8'))


def portable(value):
    if isinstance(value, dict):
        return {portable(k): portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str) and re.match(r'^[A-Za-z]:[/\\]', value):
        path = re.sub(r'[/\\]+', '/', value)
        assert '/voice-arena/' in path, path
        return path.split('/voice-arena/', 1)[1]
    return value


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


assert not OUT.exists() and not ARCHIVE.exists(), 'Export already exists; inspect before replacing.'
inputs = [json.loads(line) for line in (RUN / 'inputs.jsonl').read_text(encoding='utf-8').splitlines() if line]
selected = sorted((r for r in inputs if r['reference'] == 'alternate'), key=lambda r: (r['utt_id'], r['seed']))
assert len(selected) == 8
assert {(r['utt_id'], r['seed']) for r in selected} == {(f'V{i:04}', s) for i in range(1, 5) for s in (2207, 7811)}
results = {r['audio_sha256']: r for r in map(json.loads, (RUN / 'analysis.jsonl').read_text(encoding='utf-8').splitlines())}
endings = {r['source_id']: r for r in read('ending-checks.json')['rows']}
summary = read('summary.json')
quality = {r['source_id']: r for r in summary['per_file']}
generation = read('generation.json')
OUT.mkdir()
files, previous = [], []
for row in selected:
    source = Path(row['path'])
    data = source.read_bytes()
    digest = sha(data)
    assert digest == row['sha256'] == results[digest]['audio_sha256']
    assert data.startswith(b'OggS') and b'OpusHead' in data[:100]
    name = f"{row['utt_id']}_seed{row['seed']}.ogg"
    shutil.copyfile(source, OUT / name)
    ending = endings[row['source_id']]
    assert ending['sha256'] == row['raw_sha256']
    files.append({
        'file': name, 'sha256': digest, 'size_bytes': len(data),
        'source_id': row['source_id'], 'utt_id': row['utt_id'], 'seed': row['seed'],
        'ground_truth': 'spoof', 'text': row['text'],
        'duration_sec': row['duration_sec'], 'sample_rate_hz': row['sr'], 'channels': 1,
        'generate_params': row['generate_params'], 'raw_sha256_not_included': row['raw_sha256'],
        'target_output_frames': row['target_output_frames'], 'peak_before_scale': row['peak_before_scale'],
        'previous_quality_check_on_raw_wav': {
            **{k: v for k, v in ending.items() if k not in ('source_id', 'sha256', 'text')},
            'asr_wer': quality[row['source_id']]['asr_wer'],
            'scope': 'Raw WAV before channel processing; raw WAV is not included.',
        },
    })
    previous.append({'file': name, **{k: v for k, v in results[digest].items() if k != 'file'}})

write_json('manifest.json', {
    'schema': 'voice-arena/flat-audio-export@1',
    'exported_at': datetime.now(timezone.utc).isoformat(),
    'source_run': RUN.name, 'selection': 'reference=alternate; all four phrases and both seeds',
    'audio_transformed_for_export': False, 'all_ground_truth': 'spoof',
    'generation': portable({k: generation[k] for k in (
        'created_at', 'status', 'steps', 'cfg', 'extra_generation_sec', 'versions', 'hashes',
        'ffmpeg_version', 'vocos_center_padding_compensation_frames', 'device')}),
    'tts_model': {'model': 'F5TTS_v1_Base', 'model_id': 'Misha24-10/F5-TTS_RUSSIAN',
                  'revision': 'ea166adeae4c80ec5ee423a671e2bdb83906cf84'},
    'reference': {k: v for k, v in generation['references']['alternate'].items() if k not in ('path', 'text_path')},
    'reference_audio_included': False,
    'channel': {'name': 'wide', 'container': 'ogg', 'codec': 'opus', 'bitrate_bps': 24000,
                'application': 'voip', 'channels': 1, 'sample_rate_hz': 48000,
                'processing_before_encoding': 'loudnorm=I=-23:LRA=7:TP=-2,apad=pad_dur=0.45,aresample=48000:resampler=soxr:precision=28'},
    'ending_check_model': read('ending-checks.json')['model_source'],
    'ending_check_num_beams': read('ending-checks.json')['num_beams'],
    'files': files,
})
write_json('previous_analysis.json', {
    'note': 'Historical results, not a new evaluation. Provenance describes the complete 16-file comparison; this archive contains only eight alternate-reference files.',
    'provenance': portable(read('analysis.jsonl.metadata.json')),
    'groups_in_original_comparison': summary['groups'],
    'paired_scores_in_original_comparison': summary['pairs'],
    'original_reference_audio_included': False,
    'selected_results': previous,
})

(OUT / 'README_AGENT.md').write_text('''# Инструкция анализирующему агенту

Проанализируй восемь приложенных аудиозаписей и подготовь воспроизводимый
отчёт о результатах детектора и качестве речи. Все записи синтезированы:
истинная метка каждой — `spoof`. Вердикт `bonafide` означает ошибку
детектора на синтетической записи, а не подтверждение живого голоса.

## Что в архиве

Все файлы лежат в корне. `V0001_seed2207.ogg` и аналогичные имена — четыре
фразы × два seed (2207 и 7811), один основной голос, альтернативный
референс и прежний канал wide. Это точные входы успешной ветки
`r20-f5-reference-comparison-20261018-v2`, без изменений при упаковке.
Другой голос, обычный Opus, прежний референс и исходные WAV не включены.
`manifest.json` содержит тексты, параметры, длительности и хэши.
`previous_analysis.json` содержит исторические результаты и происхождение
анализатора. `SHA256SUMS.txt` покрывает все остальные файлы архива.
JSON и эта инструкция записаны в UTF-8.

## Порядок работы

1. Проверь SHA-256 всех файлов по SHA256SUMS.txt. Проверь наличие восьми
   уникальных Ogg, совпадение с manifest.json и доступность декодирования.
   При несовпадении сообщи о нём; не подменяй входы похожими записями.
2. Прогони все восемь оригинальных Ogg через доступный анализатор.
   Для сопоставления с прежним прогоном используй тот же код, веса,
   зависимости, режим вычислений и порог 0,5, если они доступны.
   Их идентификаторы и контрольные суммы есть в previous_analysis.json.
   Веса и код анализатора в этот архив не входят. Если их нет или они
   отличаются, явно обозначь новый анализатор и ограничение сравнения.
   Отсутствующие результаты не выдумывай.
3. Не перекодируй, не нормализуй, не обрезай паузы и не меняй громкость
   входов до анализа. Обычное декодирование и подготовку внутри самого
   анализатора сохрани штатными. Для прослушивания допустимо декодировать
   отдельную копию, сохранив исходный Ogg для анализа.
4. Сохрани по каждому файлу имя, SHA-256, model_score, итоговый spoof_score,
   verdict, длительность найденной речи, число сегментов, dsp_reason,
   спектральные признаки и ошибки, если эти поля доступны. Разделяй
   model_score нейросети и spoof_score всего каскада: они могут различаться.
   В прежнем коде verdict=spoof при spoof_score > 0,5; при равенстве —
   bonafide. Прежнее поле duration — длительность VAD-сегментов, а не
   полная длительность файла. Отсутствие речи или ошибка не считаются
   успешным прохождением детектора.
5. Отдельно проверь качество на слух и, при наличии, распознаванием:
   сопоставь речь с полным текстом в manifest.json, особенно первое слово
   и конец второго предложения. Отметь пропуски, повторы, щелчки,
   обрывы и неестественные паузы. Добавленные 0,45 секунды тишины сами
   по себе не доказывают, что последний звук произнесён полностью.
   Если прослушивание недоступно, честно укажи это.
6. Зафиксируй независимые результаты, затем сравни их с историческими.
   Не подстраивай порог и обработку под желаемые ответы. В отчёте дай
   таблицу восьми файлов, среднее и максимум model_score, запас до 0,5
   для худшего примера, число итоговых bonafide/spoof и отдельно ошибок.
   Для каждого расхождения укажи величину и возможную проверяемую причину.
   Приложи машиночитаемые результаты, параметры, версии, хэши весов/кода
   и точную команду повторного запуска использованного анализатора.

## Известный результат и ограничения

В историческом прогоне этой ветки: 8/8 bonafide, средний model_score
0,0660510, максимум 0,4232025 у V0003/seed2207. Минимальный запас до
порога 0,5 — 0,0767975; ориентир «максимум не выше 0,30» не достигнут.
На прежнем референсе в том же сравнении было 7/8 bonafide, среднее
0,3795565 и максимум 0,5039825. Его аудио в архиве нет, поэтому новое
парное сравнение по нему из этого архива выполнить нельзя.

Whisper Base распознал окончания всех восьми исходных WAV до обработки
каналом. Однако в V0001 и V0002 на обоих seed расшифровки пропускают
начальные «Сегодня» и «Пожалуйста»; это требует проверки на слух,
а не автоматического вывода об отсутствии слов в аудио. Точные прежние
расшифровки находятся в manifest.json. Качество на слух и сходство
тембра ранее не подтверждены. Живого референса в архиве нет.

Это небольшой набор знакомых фраз из настройки на одном детекторе,
а не независимый тест. Референс выбран до его проверки детектором.
Два seed на каждой фразе не дают восьми независимых речевых сценариев.
Все входы синтетические, живого контроля нет: не выводи из этих восьми
записей общую точность, EER, устойчивость на других голосах и текстах
или гарантию прохождения другого анализатора.
''', encoding='utf-8')

payload = sorted(OUT.iterdir())
(OUT / 'SHA256SUMS.txt').write_text(''.join(f'{sha(p.read_bytes())}  {p.name}\n' for p in payload), encoding='utf-8')
with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(OUT.iterdir()):
        archive.write(file, arcname=file.name)

with zipfile.ZipFile(ARCHIVE) as archive:
    assert archive.testzip() is None
    names = archive.namelist()
    assert len(names) == 12 and len(set(names)) == 12
    assert all('/' not in n and '\\' not in n for n in names)
    assert len([n for n in names if n.endswith('.ogg')]) == 8
    for file in OUT.iterdir():
        assert archive.read(file.name) == file.read_bytes()
    for item in files:
        assert sha(archive.read(item['file'])) == item['sha256']
    for line in archive.read('SHA256SUMS.txt').decode('utf-8').splitlines():
        digest, name = line.split('  ', 1)
        assert sha(archive.read(name)) == digest
    for name in ('manifest.json', 'previous_analysis.json'):
        decoded = archive.read(name).decode('utf-8')
        assert not re.search(r'[A-Za-z]:[/\\]', decoded), 'Absolute machine path leaked'
        json.loads(decoded)
    assert files[0]['text'].startswith('\u0421\u0435\u0433\u043e\u0434\u043d\u044f')

print(json.dumps({'archive': str(ARCHIVE), 'size_bytes': ARCHIVE.stat().st_size,
                  'sha256': sha(ARCHIVE.read_bytes()), 'audio_files': len(files),
                  'entries': names, 'crc_ok': True, 'hashes_ok': True}, ensure_ascii=True, indent=2))
