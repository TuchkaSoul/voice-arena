"""Корпус фраз — общий для всех раундов вход.

Один текст на строку, строки с `#` и пустые игнорируются. utt_id
считается от порядкового номера значимой строки, поэтому фразы можно
только дописывать в конец: если вставить строку в середину, все
последующие идентификаторы сдвинутся, и прошлые раунды перестанут
сравниваться с новыми.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Phrase:
    utt_id: str
    text: str


def load(path: Path, prefix: str = "p") -> list[Phrase]:
    if not path.exists():
        raise FileNotFoundError(f"нет файла корпуса: {path}")

    phrases: list[Phrase] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        phrases.append(Phrase(utt_id=f"{prefix}{len(phrases) + 1:04d}", text=line))

    if not phrases:
        raise ValueError(f"{path}: ни одной фразы")
    return phrases
