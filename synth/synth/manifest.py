"""Манифест прогона — контракт между стороной синтеза и стороной анализа.

Одна строка JSON на одну запись. Сторона анализа читает этот файл,
пишет рядом свой — с вердиктом и score, — и метрики считаются
соединением двух файлов по utt_id.

Пока формат один на всех, результаты разных раундов сравнимы. Как
только кто-то начнёт складывать результаты «по-своему», через месяц
будет несколько папок, которые нельзя сопоставить. Поэтому поле schema
стоит в каждой строке: если формат поменяется, это будет видно сразу,
а не через месяц.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# @2 добавила поле `corpus` — без него по манифесту нельзя сказать, из
# какого набора фраз запись, а наборов теперь два (короткий и длинный).
# Поле необязательное, читатели @1 его просто не заметят.
SCHEMA = "voice-arena/manifest@2"

LABEL_SPOOF = "spoof"        # синтез
LABEL_BONAFIDE = "bonafide"  # живой голос


@dataclass(frozen=True)
class ManifestRow:
    utt_id: str
    text: str                # что просили произнести
    text_normalized: str     # что движок реально получил на вход
    label: str
    engine: str
    preset: str
    channel: str
    path: str                # относительно корня synth/
    sr: int                  # частота итогового файла
    engine_sr: int           # что выдал движок до канала
    duration_sec: float
    sha256: str
    run_id: str
    seed: int
    created_at: str
    corpus: str = ""         # имя файла корпуса, из которого взята фраза
    engine_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, **asdict(self)}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(path: Path, rows: list[ManifestRow], append: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    return path


def read(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} — битая строка JSON: {e}") from e
    return rows
