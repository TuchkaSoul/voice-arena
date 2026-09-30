"""Канал передачи: что происходит со звуком между синтезом и детектором.

Ступень вынесена отдельно от синтеза намеренно. Живой корпус обязан
пройти ровно этот же код, иначе детектор научится отличать не человека
от машины, а один способ записи от другого — и все метрики окажутся
бессмысленными. Поэтому apply() работает с файлом, а не с движком, и
зовётся из обеих веток: и из pipeline, и из команды `main.py channel`.

Все каналы приводят звук к 16 кГц моно — в этом формате работает
анализатор. Ресемплинг делается здесь, до кодека, а не после: иначе
артефакты получаются не те, что в реальном мессенджере.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

TARGET_SR = 16000


@dataclass(frozen=True)
class Channel:
    name: str
    ext: str
    args: tuple[str, ...]
    description: str


# Выравнивание громкости — общее для всех каналов и обеих сторон.
#
# Зачем: piper по умолчанию нормализует пик ровно в 1.0, XTTS — нет, а
# живая запись с микрофона тем более. Без выравнивания детектор выучил
# бы не артефакты синтеза, а уровень громкости: «пик равен единице —
# значит подделка». Это ровно та же ловушка, что водяной знак в
# chatterbox, только своими руками.
#
# Раз фильтр стоит в _COMMON, он применяется и к синтезу, и к живому
# корпусу — одним и тем же кодом. Отключать можно только одновременно
# для обеих сторон, иначе метрики станут фикцией.
_LOUDNORM = ("-af", "loudnorm=I=-23:LRA=7:TP=-2")

_COMMON = _LOUDNORM + ("-ar", str(TARGET_SR), "-ac", "1")

CHANNELS: dict[str, Channel] = {
    "clean": Channel(
        "clean", "wav",
        ("-c:a", "pcm_s16le") + _COMMON,
        "чистый wav 16 кГц — верхняя граница качества, детектору должно быть легко",
    ),
    "flac": Channel(
        "flac", "flac",
        # -sample_fmt s16 обязателен. Без него ffmpeg берёт разрядность с
        # выхода loudnorm и пишет 24 бита, тогда как clean — 16. Файлы
        # переставали быть побитово одинаковыми, различаясь шумом
        # квантования, и детектор мог бы выучить именно эту разницу
        # вместо артефактов синтеза.
        ("-c:a", "flac", "-sample_fmt", "s16") + _COMMON,
        "тот же звук, что clean, но другой контейнер — проверка загрузчика анализатора",
    ),
    "mp3_192": Channel(
        "mp3_192", "mp3",
        ("-c:a", "libmp3lame", "-b:a", "192k") + _COMMON,
        "mp3 192 кбит/с — на 16 кГц практически прозрачно, потерь почти нет",
    ),
    "mp3_128": Channel(
        "mp3_128", "mp3",
        ("-c:a", "libmp3lame", "-b:a", "128k") + _COMMON,
        "mp3 128 кбит/с — на слух почти прозрачно",
    ),
    "mp3_64": Channel(
        "mp3_64", "mp3",
        ("-c:a", "libmp3lame", "-b:a", "64k") + _COMMON,
        "mp3 64 кбит/с — высокие частоты уже срезаны",
    ),
    "opus_24k": Channel(
        "opus_24k", "ogg",
        ("-c:a", "libopus", "-b:a", "24k", "-application", "voip") + _COMMON,
        "opus 24 кбит/с в режиме voip — так сжимают голосовые в мессенджерах",
    ),
    "opus_12k": Channel(
        "opus_12k", "ogg",
        ("-c:a", "libopus", "-b:a", "12k", "-application", "voip") + _COMMON,
        "opus 12 кбит/с — заведомо тяжёлый случай, для нижней границы",
    ),
}

# Канал «колонка + микрофон» в ffmpeg не делается: это физическая запись
# в комнате. Появится в раунде 5 отдельной процедурой, а здесь оставлен
# явной заглушкой, чтобы имя канала было зарезервировано уже сейчас.
ROOM = "room"


class FfmpegMissing(RuntimeError):
    pass


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FfmpegMissing(
            "ffmpeg не найден в PATH. Windows: winget install Gyan.FFmpeg, "
            "затем перезапустить терминал."
        )
    return path


def get(name: str) -> Channel:
    if name == ROOM:
        raise NotImplementedError(
            "канал 'room' — это запись через колонку и микрофон, "
            "её нельзя получить из ffmpeg. Запланирован на раунд 5."
        )
    if name not in CHANNELS:
        raise KeyError(
            f"неизвестный канал {name!r}; доступны: {', '.join(sorted(CHANNELS))}"
        )
    return CHANNELS[name]


def apply(src: Path, dst_dir: Path, channel: str, stem: str) -> Path:
    """Прогоняет файл через канал. Возвращает путь к результату."""
    ch = get(channel)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{stem}.{ch.ext}"

    cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), *ch.args, str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg упал на {src.name} (канал {channel}):\n{result.stderr.strip()}"
        )
    return dst
