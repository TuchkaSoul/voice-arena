"""Сборка: текст → нормализация → движок → канал → файл + строка манифеста.

Пайплайн не знает, какой движок внутри, и не знает, каким кодеком
жмут. Он знает только порядок ступеней. Поэтому новый движок или новый
кодек не требуют правок в этом файле.
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import soundfile as sf

from synth import channel as channel_mod
from synth import manifest, text as text_mod
from synth.corpus import Phrase
from synth.engines import registry
from synth.presets import Preset


class SynthRun:
    """Один прогон: фиксированный пресет, один или несколько каналов, один run_id.

    Фраза синтезируется РОВНО ОДИН раз, а потом веером расходится по всем
    каналам. Это не оптимизация, а условие осмысленности эксперимента:
    и VITS, и XTTS берут шум случайно, поэтому повторный синтез той же
    фразы даёт другую запись. Если бы каждый канал синтезировал заново,
    `p0001` в `clean` и `p0001` в `opus_24k` были бы разными записями, и
    разница в метриках показывала бы не деградацию от кодека, а разброс
    самого синтеза. Между каналами должен меняться только канал.
    """

    def __init__(
        self,
        preset: Preset,
        channels: str | Iterable[str],
        out_dir: Path,
        run_id: str,
        seed: int = 1337,
        label: str = manifest.LABEL_SPOOF,
        corpus: str = "",
    ):
        self.preset = preset
        self.channels = [channels] if isinstance(channels, str) else list(channels)
        self.out_dir = Path(out_dir)
        self.run_id = run_id
        self.seed = seed
        self.label = label
        self.corpus = corpus

        if not self.channels:
            raise ValueError("не указан ни один канал")
        for name in self.channels:
            channel_mod.get(name)  # падаем сразу, а не после часа синтеза

    def audio_dir(self, channel: str) -> Path:
        # Корпус входит в путь: иначе два набора фраз в одном раунде
        # перезаписали бы файлы и манифесты друг друга.
        parts = [self.preset.name, self.corpus, channel]
        return self.out_dir.joinpath(self.run_id, *(p for p in parts if p))

    def manifest_path(self, channel: str) -> Path:
        stem = ".".join(p for p in (self.preset.name, self.corpus, channel) if p)
        return self.out_dir / self.run_id / f"{stem}.jsonl"

    def run(
        self,
        phrases: Iterable[Phrase],
        on_item: Callable[[Phrase, int, int], None] | None = None,
    ) -> dict[str, list[manifest.ManifestRow]]:
        phrases = list(phrases)

        # Сид фиксируем и записываем в манифест. Оговорка: полной
        # воспроизводимости он не гарантирует — VITS и XTTS берут шум
        # внутри графа, и ORT про наш random.seed ничего не знает.
        # Поэтому сид здесь — не обещание, а запись условий прогона.
        random.seed(self.seed)
        np.random.seed(self.seed)

        engine = registry.create(self.preset.engine, self.preset.engine_params)
        engine.ensure_loaded()

        rows: dict[str, list[manifest.ManifestRow]] = {ch: [] for ch in self.channels}
        with tempfile.TemporaryDirectory(prefix="voice-arena-") as tmp:
            for i, phrase in enumerate(phrases, 1):
                if on_item:
                    on_item(phrase, i, len(phrases))

                normalized = text_mod.normalize(phrase.text, **self.preset.text_params)
                utterance = engine.say(normalized)

                raw = Path(tmp) / f"{phrase.utt_id}.wav"
                sf.write(raw, utterance.audio, utterance.sr, subtype="PCM_16")

                for name in self.channels:
                    final = channel_mod.apply(
                        raw, self.audio_dir(name), name, phrase.utt_id
                    )
                    rows[name].append(
                        manifest.ManifestRow(
                            utt_id=phrase.utt_id,
                            text=phrase.text,
                            text_normalized=normalized,
                            label=self.label,
                            engine=self.preset.engine,
                            preset=self.preset.name,
                            channel=name,
                            path=final.as_posix(),
                            sr=channel_mod.TARGET_SR,
                            engine_sr=utterance.sr,
                            duration_sec=round(utterance.duration, 3),
                            sha256=manifest.sha256_file(final),
                            run_id=self.run_id,
                            seed=self.seed,
                            created_at=manifest.now_iso(),
                            corpus=self.corpus,
                            engine_params=dict(self.preset.engine_params),
                        )
                    )

        for name in self.channels:
            manifest.write(self.manifest_path(name), rows[name])
        return rows


def channel_only(
    src_dir: Path,
    out_dir: Path,
    channel: str,
    run_id: str,
    label: str = manifest.LABEL_BONAFIDE,
    preset_name: str = "live",
    patterns: tuple[str, ...] = ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg"),
) -> list[manifest.ManifestRow]:
    """Прогоняет готовые записи через канал, без синтеза.

    Нужно для живого корпуса: свои записи обязаны пройти ровно тот же
    кодек, что и синтез. Если этого не сделать, детектор научится
    отличать не человека от машины, а wav от opus.
    """
    src_dir = Path(src_dir)
    files = sorted(p for pattern in patterns for p in src_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"в {src_dir} нет аудиофайлов")

    audio_dir = Path(out_dir) / run_id / preset_name / channel
    rows: list[manifest.ManifestRow] = []

    for i, src in enumerate(files, 1):
        utt_id = f"live{i:04d}"
        final = channel_mod.apply(src, audio_dir, channel, utt_id)
        info = sf.info(str(src))
        rows.append(
            manifest.ManifestRow(
                utt_id=utt_id,
                text="",               # расшифровку кладёт сторона анализа
                text_normalized="",
                label=label,
                engine="none",
                preset=preset_name,
                channel=channel,
                path=final.as_posix(),
                sr=channel_mod.TARGET_SR,
                engine_sr=int(info.samplerate),
                duration_sec=round(info.frames / info.samplerate, 3),
                sha256=manifest.sha256_file(final),
                run_id=run_id,
                seed=0,
                created_at=manifest.now_iso(),
                engine_params={"source": src.as_posix()},
            )
        )

    path = Path(out_dir) / run_id / f"{preset_name}.{channel}.jsonl"
    manifest.write(path, rows)
    return rows
