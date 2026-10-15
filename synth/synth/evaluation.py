"""Сравнение манифестов с оценками анализатора.

Отдельно считает пропуски синтеза и ложные тревоги на живом голосе.
Записи без оценки не подменяются верным или неверным ответом.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
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
    run_id: str = ""
    preset: str = ""
    corpus: str = ""
    model_evaluated: int = 0
    model_spoof: int = 0
    verdict_changes: int = 0

    @property
    def error_rate(self) -> float | None:
        return self.mistakes / self.evaluated if self.evaluated else None


def summarize(
    manifest_paths: list[Path], results_path: Path, root: Path = Path("."),
    *, by_dataset: bool = False, model_threshold: float = 0.5,
) -> list[Summary]:
    if not manifest_paths:
        raise ValueError("не переданы манифесты")
    if not math.isfinite(model_threshold) or not 0 <= model_threshold <= 1:
        raise ValueError("порог нейросети должен быть от 0 до 1")

    results = {}
    for result in manifest.read(results_path):
        path = Path(result["file"]).resolve()
        if path in results:
            raise ValueError(f"повторная оценка файла {path}")
        results[path] = result

    groups: dict[tuple[str, ...], Summary] = {}
    seen = set()
    for manifest_path in manifest_paths:
        for row in manifest.read(manifest_path):
            label = row["label"]
            if label not in (manifest.LABEL_SPOOF, manifest.LABEL_BONAFIDE):
                raise ValueError(f"неизвестная метка {label!r} в {manifest_path}")
            dataset = (row.get("run_id", ""), row.get("preset", ""), row.get("corpus", ""))
            key = (*dataset, row["channel"], label) if by_dataset else (row["channel"], label)
            if key not in groups:
                groups[key] = Summary(channel=row["channel"], label=label)
                if by_dataset:
                    groups[key].run_id, groups[key].preset, groups[key].corpus = dataset
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
            if ("audio_sha256" in result and "sha256" in row
                    and result["audio_sha256"] != row["sha256"]):
                raise ValueError(f"оценка относится к другим байтам аудио: {file_path}")
            verdict = result.get("verdict")
            if verdict not in (manifest.LABEL_SPOOF, manifest.LABEL_BONAFIDE):
                group.errors += 1
                continue
            group.evaluated += 1
            group.mistakes += verdict != label
            model_score = result.get("model_score")
            if (type(model_score) in (int, float) and math.isfinite(model_score)
                    and 0 <= model_score <= 1):
                model_verdict = (manifest.LABEL_SPOOF if model_score > model_threshold
                                 else manifest.LABEL_BONAFIDE)
                group.model_evaluated += 1
                group.model_spoof += model_verdict == manifest.LABEL_SPOOF
                group.verdict_changes += model_verdict != verdict

    return [groups[key] for key in sorted(groups)]
