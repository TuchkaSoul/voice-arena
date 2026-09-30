"""Сравнение манифестов с оценками анализатора.

Отдельно считает пропуски синтеза и ложные тревоги на живом голосе.
Записи без оценки не подменяются верным или неверным ответом.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synth import manifest


@dataclass
class Summary:
    channel: str
    label: str
    total: int = 0
    evaluated: int = 0
    mistakes: int = 0
    errors: int = 0

    @property
    def error_rate(self) -> float | None:
        return self.mistakes / self.evaluated if self.evaluated else None


def summarize(
    manifest_paths: list[Path], results_path: Path, root: Path = Path(".")
) -> list[Summary]:
    if not manifest_paths:
        raise ValueError("не переданы манифесты")

    results = {}
    for result in manifest.read(results_path):
        path = Path(result["file"]).resolve()
        if path in results:
            raise ValueError(f"повторная оценка файла {path}")
        results[path] = result

    groups: dict[tuple[str, str], Summary] = {}
    seen = set()
    for manifest_path in manifest_paths:
        for row in manifest.read(manifest_path):
            label = row["label"]
            if label not in (manifest.LABEL_SPOOF, manifest.LABEL_BONAFIDE):
                raise ValueError(f"неизвестная метка {label!r} в {manifest_path}")
            key = (row["channel"], label)
            if key not in groups:
                groups[key] = Summary(channel=key[0], label=key[1])
            group = groups[key]
            group.total += 1

            file_path = Path(row["path"])
            if not file_path.is_absolute():
                file_path = root / file_path
            file_path = file_path.resolve()
            if file_path in seen:
                raise ValueError(f"файл встречается в нескольких манифестах: {file_path}")
            seen.add(file_path)

            result = results.get(file_path)
            if result is None or "error" in result:
                group.errors += 1
                continue
            verdict = result.get("verdict")
            if verdict not in (manifest.LABEL_SPOOF, manifest.LABEL_BONAFIDE):
                group.errors += 1
                continue
            group.evaluated += 1
            group.mistakes += verdict != label

    return [groups[key] for key in sorted(groups)]
