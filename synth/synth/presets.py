"""Пресет — это файл, а не аргументы функции.

Так сделано ради правила из README: часть пресетов сторона атаки не
показывает детектору до прогона. Пока пресет лежит в presets/held_out/
(папка целиком в .gitignore), он физически недоступен второй команде.
Если бы настройки жили в коде или в аргументах CLI, соблюсти это
правило было бы нельзя — они утекли бы в историю команд и в ревью.

Второе следствие: имя пресета попадает в манифест, и по нему всегда
можно восстановить, чем именно был сделан тот или иной файл.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

OPEN_DIR = "presets/open"
HELD_OUT_DIR = "presets/held_out"
PRESET_DIRS = (OPEN_DIR, HELD_OUT_DIR)


@dataclass(frozen=True)
class Preset:
    name: str
    engine: str
    engine_params: dict = field(default_factory=dict)
    text_params: dict = field(default_factory=dict)
    notes: str = ""
    source: str = ""     # из какого файла загружен
    held_out: bool = False


def _from_file(path: Path) -> Preset:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "engine" not in data:
        raise ValueError(f"{path}: нет обязательного поля 'engine'")
    return Preset(
        name=data.get("name", path.stem),
        engine=data["engine"],
        engine_params=data.get("engine_params") or {},
        text_params=data.get("text") or {},
        notes=data.get("notes", ""),
        source=str(path),
        held_out=HELD_OUT_DIR in path.as_posix(),
    )


def load(name: str, root: Path = Path(".")) -> Preset:
    for directory in PRESET_DIRS:
        path = root / directory / f"{name}.yaml"
        if path.exists():
            return _from_file(path)
    known = ", ".join(p.name for p in discover(root)) or "ни одного"
    raise FileNotFoundError(f"нет пресета {name!r}; найдены: {known}")


def discover(root: Path = Path(".")) -> list[Preset]:
    presets: list[Preset] = []
    for directory in PRESET_DIRS:
        for path in sorted((root / directory).glob("*.yaml")):
            presets.append(_from_file(path))
    return presets
