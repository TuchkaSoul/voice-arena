"""Controlled channel experiment; run from synth/, without changing production channels.

python probe_resampling.py prepare --out runs/r13-resampling-20261018
Then run the unmodified analyzer on OUT/inputs.jsonl, and use summarize.
An optional --sources JSON list fixes a separate validation set before preparation.
"""
import argparse
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
import shutil
import subprocess

import soundfile as sf

from synth.manifest import sha256_file, now_iso

ROOT = Path(__file__).resolve().parent


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def readlines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def default_sources():
    index = ROOT / "runs/r12-f5-combined-20261015-v2/comparison.json"
    sources = []
    for i, row in enumerate(json.loads(index.read_text(encoding="utf-8"))):
        sources.append(dict(source_id=f"s{i:02}", voice=row["voice"],
                            utt_id=row["utt_id"], text=row["text"], label="spoof",
                            source_run=row["source_run"],
                            path=row["variants"]["dry"]["path"],
                            sha256=row["variants"]["dry"]["sha256"]))
    for voice, filename in (("live_main", "speaker_ru_1.wav"), ("live_matv", "matv.wav")):
        path = ROOT / "voices" / filename
        sources.append(dict(source_id=voice, voice=voice, utt_id=voice, text="",
                            label="bonafide", source_run="reference_control",
                            path=str(path), sha256=sha256_file(path)))
    return sources


def prepare(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "probe_resampling.source.py")
    sources = (json.loads(args.sources.read_text(encoding="utf-8"))
               if args.sources else default_sources())
    if len({s["source_id"] for s in sources}) != len(sources):
        raise ValueError("duplicate source_id")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg missing")
    (out / "ffmpeg-version.txt").write_text(
        subprocess.check_output([ffmpeg, "-version"], text=True), encoding="utf-8")
    dump(out / "sources.json", sources)
    rows, commands = [], []
    for source in sources:
        path = Path(source["path"])
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"source hash mismatch: {path}")
        branches = [("native", path, None)]
        norms = ("direct", "loudnorm") if args.profile == "resampling" else ("loudnorm",)
        for norm in norms:
            for resampler in ("swr", "soxr"):
                for codec in ("wav", "opus"):
                    name = f"{norm}_{resampler}_{codec}"
                    dest = out / "audio" / source["source_id"] / (name + (".wav" if codec == "wav" else ".ogg"))
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    filters = [] if norm == "direct" else ["loudnorm=I=-23:LRA=7:TP=-2"]
                    resample = f"aresample=16000:resampler={resampler}"
                    if resampler == "soxr":
                        resample += ":precision=28"
                    filters.append(resample)
                    encoder = (["-c:a", "pcm_s16le"] if codec == "wav" else
                               ["-c:a", "libopus", "-b:a", "24k", "-application", "voip"])
                    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                           "-i", str(path), "-af", ",".join(filters), "-ar", "16000", "-ac", "1",
                           *encoder, str(dest)]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    commands.append(dict(command=cmd, returncode=result.returncode, stderr=result.stderr))
                    dump(out / "commands.json", commands)
                    if result.returncode:
                        raise RuntimeError(result.stderr)
                    branches.append((name, dest, cmd))
        if args.profile in ("opus-bandwidth", "validation"):
            rates = (24000, 48000) if args.profile == "opus-bandwidth" else (48000,)
            applications = ("voip", "audio") if args.profile == "opus-bandwidth" else ("voip",)
            for rate in rates:
                for application in applications:
                    name = f"opus_{rate}_{application}"
                    dest = out / "audio" / source["source_id"] / (name + ".ogg")
                    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                           "-i", str(path), "-af",
                           f"loudnorm=I=-23:LRA=7:TP=-2,aresample={rate}:resampler=soxr:precision=28",
                           "-ar", str(rate), "-ac", "1", "-c:a", "libopus", "-b:a", "24k",
                           "-application", application, str(dest)]
                    completed = subprocess.run(cmd, capture_output=True, text=True)
                    commands.append(dict(command=cmd, returncode=completed.returncode, stderr=completed.stderr))
                    dump(out / "commands.json", commands)
                    completed.check_returncode()
                    branches.append((name, dest, cmd))
        for name, dest, cmd in branches:
            info = sf.info(dest)
            rows.append(dict(source, path=dest.as_posix(), sha256=sha256_file(dest),
                             source_path=path.as_posix(), source_sha256=source["sha256"],
                             channel=name, sr=info.samplerate, duration_sec=info.duration,
                             command=cmd, created_at=now_iso()))
    (out / "inputs.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    dump(out / "experiment.json", dict(created_at=now_iso(), files=len(rows),
         sources=len(sources), script_sha256=sha256_file(Path(__file__)), threshold=0.5,
         profile=args.profile,
         purpose="SWR/SoX and Opus input bandwidth; no added noise",
         independent_live_controls=False))
    print(f"Prepared {len(rows)} inputs from {len(sources)} sources: {out}", flush=True)


def summarize(args):
    out = args.out.resolve()
    inputs = readlines(out / "inputs.jsonl")
    scores = readlines(out / "analysis.jsonl")
    by_path = {str(Path(s["file"]).resolve()): s for s in scores}
    if len(inputs) != len(scores) or len(by_path) != len(scores):
        raise ValueError("missing or duplicate scores")
    groups = defaultdict(lambda: dict(n=0, bonafide=0, spoof=0, no_speech=0))
    full = []
    for row in inputs:
        score = by_path[str(Path(row["path"]).resolve())]
        if (score.get("error") or "model_score" not in score or
                score.get("audio_sha256") != row["sha256"] or
                sha256_file(Path(row["path"])) != row["sha256"] or
                sha256_file(Path(row["source_path"])) != row["source_sha256"]):
            raise ValueError(f"incomplete score or hash mismatch: {row['path']}")
        full.append(dict(row, **{k: score[k] for k in
                    ("model_score", "spoof_score", "verdict", "flatness", "rolloff_hz",
                     "spectrum_variance", "dsp_reason", "duration", "num_segments")}))
        group = groups[row["voice"], row["label"], row["channel"]]
        group["n"] += 1
        group[score["verdict"]] += 1
        group["no_speech"] += score["duration"] <= 0
    result = dict(files=len(full), audio_hashes_verified=True,
                  groups=[dict(voice=k[0], label=k[1], channel=k[2], **v)
                          for k, v in sorted(groups.items())], per_file=full)
    dump(out / "summary.json", result)
    columns = ["source_id", "voice", "source_run", "utt_id", "label", "channel", "verdict",
               "model_score", "spoof_score", "flatness", "rolloff_hz", "spectrum_variance",
               "duration", "dsp_reason", "path", "sha256", "source_sha256"]
    with (out / "scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(full)
    report = ["# Проверка ресемплинга — 18 октября 2026", "",
              "Проверен неизменённый текущий анализатор, порог 0.5. "
              "В таблице — число записей, принятых за живую речь. "
              "Для синтеза это пропуски детектора, для живого контроля — верные решения.", "",
              "| Голос | Канал | Приняты за живые |", "| --- | --- | ---: |"]
    for g in result["groups"]:
        report.append(f"| {g['voice']} | {g['channel']} | {g['bonafide']}/{g['n']} |")
    old_sources = any(r["source_run"].startswith(("r10-", "r11-")) for r in inputs)
    limits = ("Повторное использование прежних записей не образует независимый тест. "
              if old_sources else
              "Проверены новые тексты и seed, но использованы те же голоса, модель и детектор. ")
    limits += ("Живые контроли использованы как референсы синтеза; "
               if any(r["label"] == "bonafide" for r in inputs) else
               "Новых живых контролей в этой серии нет; ")
    report += ["", "## Границы вывода", "", limits +
               "EER не вычисляется. "
               "Все контрольные суммы сверены. Полные оценки — scores.csv и summary.json; "
               "код, веса и окружение анализатора — analysis.jsonl.metadata.json. "
               "Качество на слух, соответствие текста и перенос на другие корпуса нужно проверить отдельно."]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    page = ['<!doctype html><meta charset="utf-8"><title>Проверка ресемплинга</title>',
            '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ccc}audio{width:260px}</style>',
            '<h1>Проверка ресемплинга</h1><p>18 октября 2026. Все варианты сохранены; оценки детектора не измеряют естественность.</p>',
            '<table><tr><th>Источник</th><th>Канал</th><th>Решение</th><th>Wav2Vec2 / Flatness</th><th>Послушать</th></tr>']
    for r in full:
        # Absolute file URI permits opening the page directly from disk.
        page.append(f'<tr><td>{html.escape(r["voice"] + " / " + r["utt_id"] + " / " + r["source_run"])}</td>'
                    f'<td>{r["channel"]}</td><td>{r["verdict"]}</td>'
                    f'<td>{r["model_score"]:.4f} / {r["flatness"]:.4f}</td>'
                    f'<td><audio controls preload="none" src="{html.escape(Path(r["path"]).as_uri())}"></audio></td></tr>')
    page.append('</table>')
    (out / "listen.html").write_text("\n".join(page), encoding="utf-8")
    pairs = defaultdict(dict)
    for row in full:
        if row["channel"] in ("loudnorm_swr_opus", "opus_48000_voip"):
            pairs[row["source_id"]][row["channel"]] = row
    comparison = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
                  '<title>Сравнение двух каналов Opus</title>',
                  '<style>body{font:17px system-ui;max-width:980px;margin:36px auto;padding:0 20px}article{border:1px solid #bbb;padding:18px;margin:18px 0;border-radius:10px}audio{width:100%}p{line-height:1.5}small{color:#555}</style>',
                  '<h1>Сравнение двух каналов Opus</h1>',
                  '<p>Один исходник, 24 кбит/с и voip в обеих ветках. '
                  'Оцените чёткость слов, окончания, шум и естественность. '
                  'Вердикт детектора не заменяет прослушивание.</p>']
    for source_id, pair in sorted(pairs.items()):
        if len(pair) != 2:
            continue
        reference = pair["opus_48000_voip"]
        comparison.append(f'<article><h2>{html.escape(reference["voice"] + " / " + reference["utt_id"])}</h2>'
                          f'<small>{html.escape(source_id)}</small><p>{html.escape(reference["text"])}</p>')
        for name, label in (("loudnorm_swr_opus", "Штатный канал: SWR, вход 16 кГц"),
                            ("opus_48000_voip", "Проверяемый канал: SoX, вход 48 кГц")):
            row = pair[name]
            relative = Path(row["path"]).relative_to(out).as_posix()
            comparison.append(f'<p>{label}; решение: {row["verdict"]}; '
                              f'Wav2Vec2 {row["model_score"]:.4f}; Flatness {row["flatness"]:.4f}</p>'
                              f'<audio controls preload="none" src="{html.escape(relative)}"></audio>')
        comparison.append('</article>')
    comparison.append('</html>')
    (out / "comparison.html").write_text("\n".join(comparison), encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "summarize"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--profile", choices=("resampling", "opus-bandwidth", "validation"), default="resampling")
    arguments = parser.parse_args()
    (prepare if arguments.stage == "prepare" else summarize)(arguments)
