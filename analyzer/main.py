"""CLI: python main.py <audio_file> [--json]"""
import sys
import argparse

from analyzer.pipeline import Analyzer
from analyzer.report import format_report, format_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="voice-analyzer",
        description="Анализ аудиозаписи: детекция синтеза голоса.",
    )
    p.add_argument("audio", help="путь к аудиофайлу")
    p.add_argument(
        "--json",
        action="store_true",
        help="вывести результат в JSON (для интеграции)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        analyzer = Analyzer(
            vad_path="models/silero_vad.onnx",
            aasist_path="models/spectra-aasist3.onnx",
            ecapa_path="models/voxceleb_ECAPA512_LM.onnx",
        )
        result = analyzer.analyze(args.audio)
    except FileNotFoundError as e:
        print(f"Ошибка: файл не найден — {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Ошибка анализа: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(format_json(result))
    else:
        print(format_report(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())