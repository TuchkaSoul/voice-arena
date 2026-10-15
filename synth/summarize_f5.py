"""Verify and summarize the controlled F5 experiment without selecting a winner."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from synth.manifest import sha256_file


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    out = args.run
    rows = read_rows(out / 'comparison.jsonl')
    experiment = json.loads((out / 'experiment.json').read_text(encoding='utf-8'))
    analysis = {str(Path(r['file']).resolve()): r for r in read_rows(out / 'analysis.jsonl')}
    groups = defaultdict(list)
    checked = set()
    for r in rows:
        for pathkey, hashkey in [('path', 'sha256'), ('source_path', 'source_sha256'), ('raw_path', 'raw_sha256')]:
            path = Path(r[pathkey])
            pair = (path.resolve(), r[hashkey])
            if pair not in checked:
                if sha256_file(path) != r[hashkey]:
                    raise ValueError(f'хэш не совпал: {path}')
                checked.add(pair)
        a = analysis[str(Path(r['path']).resolve())]
        if a.get('audio_sha256') != r['sha256']:
            raise ValueError(f'анализатор проверял другие байты: {r["path"]}')
        if 'model_score' not in a or 'error' in a:
            raise ValueError(f'нет полноценного результата анализатора: {r["path"]}')
        groups[r['voice'], r['profile'], r['effect'], r['channel']].append(a)
    summary = []
    report = [f'# Проверка звучания F5 — {out.name}', '',
              '## Постановка задачи', '',
              'Проверить шум во время речи, излишне ровную подачу и слабые отражения '
              'на двух собственных голосах команды. Сравнить настройки при одинаковых '
              'текстах, случайном зерне, скорости и времени генерации. Обученные веса '
              'модели и параметры анализатора не изменялись.', '',
              '## Выполненный прогон', '',
              f'Каталог: `{out.as_posix()}`. '
              f'Генераций: {experiment["generations"]}; выходов для проверки: {len(rows)}. '
              f'Все {len(rows)} файлов, источники обработок и сырые WAV проверены по SHA-256. '
              'Хэши также совпали с результатами анализатора.', '',
              f'Запас времени: {experiment.get("reserve_sec", 0)} с. '
              f'Seed: {experiment["seeds"]}; speed=1, sway=-1. '
              'baseline: 32 шага/CFG=2; steps48: 48/2; cfg1_5: 48/1.5. '
              'Для cfg1_5 отдельно получены soft_hf и reflections без повторной генерации.', '',
              'Soft_hf: полочный фильтр −2.5 дБ с частотой перехода 5 кГц. '
              'Reflections: задержки 11–73 мс, суммарная отражённая амплитуда 8% '
              'от прямой. Это модель ранних отражений, а не измеренная комната. '
              'Случайный шум не добавляется. Файлы для прослушивания имеют одинаковую '
              'целевую громкость −23 LUFS; фактическое достижение цели отдельно не измерялось.', '',
              '## Анализатор', '',
              'В таблице — число файлов, названных синтезом. Нейросеть: model_score > 0.5; '
              'итог: каскад модели и спектральных правил с порогом 0.5. '
              'Два текста и один seed не образуют независимую проверочную выборку; '
              'доля пропусков не переносится на другие тексты или модели.', '',
              '| Голос | Настройка | Обработка | Канал | Итог: синтез | Нейросеть: синтез |',
              '| --- | --- | --- | --- | ---: | ---: |']
    for key, values in sorted(groups.items()):
        s = dict(voice=key[0], profile=key[1], effect=key[2], channel=key[3], count=len(values),
                 spoof=sum(r['verdict'] == 'spoof' for r in values),
                 model_spoof=sum(r['model_score'] > .5 for r in values),
                 model_min=min(r['model_score'] for r in values),
                 model_max=max(r['model_score'] for r in values))
        summary.append(s)
        report.append(f'| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {s["spoof"]}/{len(values)} | {s["model_spoof"]}/{len(values)} |')
    report += ['', '## Диагностика исходных генераций', '',
               'RMS хвоста относится к последним 20 мс сырого выхода. '
               'Звучащий хвост — повод проверить окончание; это не доказательство обрыва. '
               'Энергия 6–8 кГц содержит также согласные и дыхание, поэтому не является '
               'измерением шума. Простое ослабление этой полосы не доказывает улучшения.', '',
               '| Голос | Текст | Настройка | Пик до обработки | RMS хвоста, dBFS |',
               '| --- | --- | --- | ---: | ---: |']
    for r in rows:
        if r['channel'] == 'native_24k' and r['effect'] == 'dry':
            d = r['raw_diagnostics']
            report.append(f'| {r["voice"]} | {r["utt_id"]} | {r["profile"]} | {d["peak"]:.4f} | {d["tail_20ms_rms_dbfs"]:.2f} |')
    asrpath = out / 'asr.jsonl'
    if asrpath.exists():
        asr = read_rows(asrpath)
        report += ['', '## Дополнительная проверка распознаванием', '',
                   'Whisper Base, локальные веса, greedy, num_beams=1. '
                   'Проверены сырые WAV r10 и r11. Это диагностика расхождений, '
                   'а не оценка произношения на слух.', '',
                   '| Прогон | Голос | Текст | Настройка | Распознанный текст |',
                   '| --- | --- | --- | --- | --- |']
        for r in asr:
            if sha256_file(Path(r['raw_path'])) != r['sha256']:
                raise ValueError('расхождение аудио ASR')
            report.append(f'| {r["run"]} | {r["voice"]} | {r["utt_id"]} | {r["profile"]} | {r["transcription"]} |')
    report += ['', '## Выводы и ограничения', '']
    findings = out / 'FINDINGS.md'
    if findings.exists():
        report += [findings.read_text(encoding='utf-8'), '']
    else:
        report += ['Автоматическая таблица не выбирает лучшую настройку. '
                   'Естественность и шум на слух необходимо оценить отдельно; '
                   'выводы по предыдущему запуску сюда не переносятся.', '']
    report += [
               '## Повторение и материалы', '',
               'Из каталога synth: ` .venv-f5/Scripts/python.exe compare_f5.py '
               '--voices f5_ru f5_matv --seeds 1337 --reserve-sec 0.65 --out runs/new-quality-check`.', '',
               'После запуска анализатора: `.venv-f5/Scripts/python.exe summarize_f5.py '
               'runs/new-quality-check`. Окружение, запуск анализатора, остановка и '
               'повтор с чистого окружения описаны в `synth/README.md`; подробная история — '
               'в `synth/log.md`.', '',
               'Сохранены `experiment.json`, `comparison.jsonl`, `manifest.jsonl`, '
               '`analysis.jsonl`, метаданные анализатора, `summary.json`, сырые WAV '
               'и HTML с плеерами. Настройки исходных пресетов сохранены.', '',
               'Источники: [F5 API](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/infer), '
               '[русская модель](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN). '
               'Фактические веса закреплены ревизией в experiment.json.']
    (out / 'summary.json').write_text(json.dumps(dict(audio_hashes_verified=True, files=len(rows), groups=summary),
                                                ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'Проверено {len(rows)} файлов; отчёт: {out / "REPORT.md"}')


if __name__ == '__main__':
    main()
