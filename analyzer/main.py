import sys
import os
import glob
import argparse
import json
from pathlib import Path

from analyzer.pipeline import Analyzer
from report import print_header, format_row, print_footer
from provenance import collect as collect_provenance, sha256_file

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac", ".wma"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deepfake-detector",
        description="Analyze audio files for deepfake/spoofing artifacts.",
    )
    parser.add_argument(
        "audio_dir",
        nargs="?",
        default="audio",
        help="Directory with audio files (default: audio)",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Scan subdirectories recursively",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Spoof threshold for verdict (default: 0.5)",
    )
    parser.add_argument(
        "--jsonl",
        help="сохранить оценки и SHA-256 файлов в JSONL; рядом — *.metadata.json с версиями и весами",
    )
    parser.add_argument(
        "--manifest", nargs="+",
        help="взять только файлы из указанных манифестов",
    )
    parser.add_argument(
        "--manifest-root", default=".",
        help="корень относительных путей в манифестах (по умолчанию текущая директория)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Force inference device (default: auto)",
    )
    parser.add_argument(
        "--vad-path",
        default="models/silero_vad.onnx",
        help="Path to Silero VAD ONNX model",
    )
    parser.add_argument(
        "--model-path",
        default="models/wav2vec2-deepfake-voice-detector",
        help="Local directory for Wav2Vec2 deepfake detector",
    )
    parser.add_argument(
        "--model-id",
        default="garystafford/wav2vec2-deepfake-voice-detector",
        help="HuggingFace model id (used if local dir is missing)",
    )
    return parser.parse_args()


def collect_files(audio_dir: str, recursive: bool) -> list[str]:
    if not os.path.isdir(audio_dir):
        return []

    if recursive:
        found = []
        for root, _, files in os.walk(audio_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                    found.append(os.path.join(root, f))
        return found

    return [
        p for p in glob.glob(os.path.join(audio_dir, "*"))
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in AUDIO_EXTS
    ]


def collect_manifest_files(paths: list[str], root: str) -> list[str]:
    files = []
    seen = set()
    for name in paths:
        with open(name, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                path = Path(row["path"])
                if not path.is_absolute():
                    path = Path(root) / path
                path = path.resolve()
                if path in seen:
                    raise ValueError(f"файл повторяется в манифестах: {path}")
                if not path.is_file():
                    raise FileNotFoundError(path)
                seen.add(path)
                files.append(str(path))
    return files


def main() -> int:
    args = parse_args()

    try:
        files = (
            collect_manifest_files(args.manifest, args.manifest_root)
            if args.manifest
            else sorted(collect_files(args.audio_dir, args.recursive))
        )
    except (OSError, ValueError, KeyError) as e:
        print(f"Invalid manifest: {e}", file=sys.stderr)
        return 2

    if not files:
        print(f"No audio files found in '{args.audio_dir}'.", file=sys.stderr)
        return 0

    try:
        analyzer = Analyzer(
            vad_path=args.vad_path,
            model_path=args.model_path,
            model_id=args.model_id,
            device=args.device,
        )

        run_metadata = None
        if args.jsonl:
            run_metadata = collect_provenance(Path(args.model_path), Path(args.vad_path))
            model = analyzer.ast_w2v2.model
            run_metadata.update({
                "threshold": args.threshold,
                "device": str(analyzer.device),
                "model_dtype": str(model.dtype),
                "spoof_idx": analyzer.ast_w2v2.spoof_idx,
                "attention_implementation": getattr(model.config, "_attn_implementation", None),
                "id2label": model.config.id2label,
            })

        print(f"[*] Inference device: {str(analyzer.device).upper()}", file=sys.stderr)
        if args.manifest:
            print(f"[*] Reading {len(args.manifest)} manifest(s)", file=sys.stderr)
        else:
            print(f"[*] Scanning directory: {os.path.abspath(args.audio_dir)}"
                  f"{' (recursive)' if args.recursive else ''}", file=sys.stderr)
    except Exception as e:
        print(f"Initialization error: {e}", file=sys.stderr)
        return 2

    print(f"[*] Found {len(files)} file(s).", file=sys.stderr)
    print_header()

    records = []
    for path in files:
        try:
            audio_hash = sha256_file(Path(path)) if args.jsonl else None
            res = analyzer.analyze(path, spoof_threshold=args.threshold)
            print(format_row(res))
            records.append({
                **res,
                "file": os.path.abspath(path),
                **({"audio_sha256": audio_hash} if audio_hash is not None else {}),
            })
        except Exception as e:
            print(f"│ Error analyzing {path}: {e}")
            records.append({"file": os.path.abspath(path), "error": str(e)})

    print_footer(records)

    if args.jsonl:
        output = Path(args.jsonl)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        run_metadata.update({
            "results_sha256": sha256_file(output),
            "files": len(records),
            "errors": sum("error" in record for record in records),
        })
        metadata_path = Path(str(output) + ".metadata.json")
        metadata_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[*] Run metadata: {metadata_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())