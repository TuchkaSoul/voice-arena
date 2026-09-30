"""CLI стороны синтеза.

    python main.py presets
    python main.py channels
    python main.py normalize "Скидка 25% до 15 марта"
    python main.py synth   --preset piper_irina --channel opus_24k --run r0
    python main.py channel --src live_raw --channel opus_24k --run r0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from synth import channel as channel_mod
from synth import corpus, evaluation, manifest, presets, text
from synth.pipeline import SynthRun, channel_only

DEFAULT_CORPUS = Path("corpus/phrases_v1.txt")
DEFAULT_OUT = Path("runs")


def cmd_presets(args: argparse.Namespace) -> int:
    found = presets.discover()
    if not found:
        print("пресетов нет — положи yaml в presets/open/")
        return 1
    for preset in found:
        mark = " [закрытый]" if preset.held_out else ""
        print(f"{preset.name:<20} {preset.engine:<12}{mark}")
        if preset.notes:
            print(f"{'':<20} {preset.notes}")
    return 0


def cmd_channels(args: argparse.Namespace) -> int:
    for ch in channel_mod.CHANNELS.values():
        print(f"{ch.name:<10} .{ch.ext:<5} {ch.description}")
    print(f"{channel_mod.ROOM:<10} {'':<6} колонка + микрофон, раунд 5, пока не реализован")
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    """Быстрая проверка нормализации без единой модели."""
    params = presets.load(args.preset).text_params if args.preset else {}
    print(text.normalize(args.text, **params))
    return 0


def _resolve_channels(names: list[str]) -> list[str]:
    """`all` разворачивается в список всех каналов, кроме нереализованных."""
    if "all" in names:
        return list(channel_mod.CHANNELS)
    return names


def cmd_synth(args: argparse.Namespace) -> int:
    preset = presets.load(args.preset)
    corpus_path = Path(args.corpus)
    phrases = corpus.load(corpus_path, prefix=args.prefix)
    if args.limit:
        phrases = phrases[: args.limit]

    channels = _resolve_channels(args.channel)
    run = SynthRun(
        preset=preset,
        channels=channels,
        out_dir=Path(args.out),
        run_id=args.run,
        seed=args.seed,
        corpus=corpus_path.stem,
    )

    print(
        f"пресет {preset.name} ({preset.engine}), корпус {corpus_path.stem}, "
        f"{len(phrases)} фраз, каналы: {', '.join(channels)}"
    )

    def progress(phrase, i, total):
        print(f"  [{i}/{total}] {phrase.utt_id}  {phrase.text[:56]}", flush=True)

    rows = run.run(phrases, on_item=progress)

    seconds = sum(r.duration_sec for r in rows[channels[0]])
    files = sum(len(v) for v in rows.values())
    print(f"\nготово: {files} файлов, {seconds:.1f} с звука на канал")
    for name in channels:
        print(f"  {name:<10} {run.manifest_path(name)}")
    return 0


def cmd_channel(args: argparse.Namespace) -> int:
    for name in _resolve_channels(args.channel):
        rows = channel_only(
            src_dir=Path(args.src),
            out_dir=Path(args.out),
            channel=name,
            run_id=args.run,
            label=args.label,
            preset_name=args.name,
        )
        print(f"обработано {len(rows)} файлов, канал {name}")
        print(f"манифест: {Path(args.out) / args.run / f'{args.name}.{name}.jsonl'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    paths = [Path(name) for name in args.manifest]
    for row in evaluation.summarize(paths, Path(args.results), Path(args.root)):
        if row.error_rate is None:
            rate = "нет оценок"
        else:
            rate = f"{row.mistakes}/{row.evaluated} ({row.error_rate:.1%})"
        kind = "пропуски синтеза" if row.label == manifest.LABEL_SPOOF else "ложные тревоги"
        print(f"{row.channel:<10} {kind:<19} {rate}; без оценки: {row.errors}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voice-synth", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="синтезировать корпус выбранным пресетом")
    s.add_argument("--preset", required=True, help="имя пресета из presets/")
    s.add_argument(
        "--channel", nargs="+", default=["clean"],
        help="канал передачи; можно несколько через пробел, `all` — все сразу. "
             "Фраза синтезируется один раз и расходится по всем каналам",
    )
    s.add_argument("--run", required=True, help="идентификатор раунда, например r0")
    s.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    s.add_argument(
        "--prefix", default="p",
        help="приставка к utt_id. У каждого корпуса своя, иначе "
             "идентификаторы разных наборов столкнутся",
    )
    s.add_argument("--out", default=str(DEFAULT_OUT))
    s.add_argument("--seed", type=int, default=1337)
    s.add_argument("--limit", type=int, help="взять только первые N фраз (для отладки)")
    s.set_defaults(func=cmd_synth)

    c = sub.add_parser("channel", help="прогнать готовые записи через канал (живой корпус)")
    c.add_argument("--src", required=True, help="исходный аудиофайл или папка")
    c.add_argument("--channel", nargs="+", required=True,
                   help="один или несколько каналов, либо all")
    c.add_argument("--run", required=True)
    c.add_argument("--out", default=str(DEFAULT_OUT))
    c.add_argument("--label", default=manifest.LABEL_BONAFIDE,
                   choices=[manifest.LABEL_BONAFIDE, manifest.LABEL_SPOOF])
    c.add_argument("--name", default="live", help="имя «пресета» для этой пачки")
    c.set_defaults(func=cmd_channel)

    e = sub.add_parser("evaluate", help="сравнить манифесты с оценками анализатора")
    e.add_argument("--manifest", nargs="+", required=True,
                   help="файлы манифестов синтеза и живого голоса")
    e.add_argument("--results", required=True, help="JSONL от analyzer/main.py --jsonl")
    e.add_argument("--root", default=".", help="корень путей в манифестах (synth/)")
    e.set_defaults(func=cmd_evaluate)

    n = sub.add_parser("normalize", help="показать, что движок получит на вход")
    n.add_argument("text")
    n.add_argument("--preset", help="взять настройки нормализации из пресета")
    n.set_defaults(func=cmd_normalize)

    sub.add_parser("presets", help="список пресетов").set_defaults(func=cmd_presets)
    sub.add_parser("channels", help="список каналов").set_defaults(func=cmd_channels)

    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError, NotImplementedError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 2
    except channel_mod.FfmpegMissing as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nпрервано", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
