"""Build a local listening page after checking unpadded speech endings."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf

from synth.manifest import sha256_file, now_iso


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    checks = json.loads(args.checks.read_text(encoding="utf-8"))
    if checks["sources_sha256"] != sha256_file(args.sources):
        raise ValueError("QA refers to another source manifest")
    verified = {r["source_id"]: r for r in checks["rows"]}
    if len(verified) != len(sources) or not checks["all_endings_recognized"]:
        raise ValueError("not every source passed ending recognition; review or regenerate")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    template = Path(__file__).with_name("listening_template.html")
    shutil.copyfile(template, out/"listening_template.source.html")
    shutil.copyfile(args.checks, out/"ending-checks.json")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg missing")
    (out/"ffmpeg-version.txt").write_text(subprocess.check_output([ffmpeg,"-version"],text=True),encoding="utf-8")
    records, manifest, commands = [], [], []
    for source in sources:
        path = Path(source["path"]).resolve()
        if sha256_file(path) != source["sha256"] or verified[source["source_id"]]["sha256"] != source["sha256"]:
            raise ValueError("source changed after ending checks")
        raw_duration = sf.info(path).duration
        record = dict(source, ending_check=verified[source["source_id"]], variants=[])
        for key, rate, resampler, codec, label in (
                ("wav",24000,"soxr","wav","Исходный WAV"),
                ("standard",16000,"swr","opus","Обычный Opus · вход 16 кГц"),
                ("wide",48000,"soxr","opus","Новый Opus · вход 48 кГц")):
            relative = Path("audio")/source["source_id"]/(key+(".wav" if codec=="wav" else ".ogg"))
            dest = out/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            filters = f"loudnorm=I=-23:LRA=7:TP=-2,apad=pad_dur=0.45,aresample={rate}:resampler={resampler}"
            if resampler=="soxr":
                filters += ":precision=28"
            encoder = (["-c:a","pcm_s16le"] if codec=="wav" else
                       ["-c:a","libopus","-b:a","24k","-application","voip"])
            cmd = [ffmpeg,"-nostdin","-hide_banner","-loglevel","error","-n","-i",str(path),
                   "-af",filters,"-ar",str(rate),"-ac","1",*encoder,str(dest)]
            result = subprocess.run(cmd,capture_output=True,text=True)
            commands.append(dict(command=cmd,returncode=result.returncode,stderr=result.stderr))
            (out/"commands.json").write_text(json.dumps(commands,ensure_ascii=False,indent=2),encoding="utf-8")
            result.check_returncode()
            info = sf.info(dest)
            data, sr = sf.read(dest,dtype="float32")
            if not len(data) or not np.isfinite(data).all() or info.duration < raw_duration+.35:
                raise ValueError(f"incomplete playback file: {dest}")
            variant = dict(key=key,label=label,url=relative.as_posix(),path=dest.as_posix(),
                           sha256=sha256_file(dest),sr=sr,duration_sec=info.duration,
                           peak=float(np.max(np.abs(data))),source_sha256=source["sha256"])
            if codec == "opus":
                preview = dest.with_suffix(".preview.wav")
                decode_cmd = [ffmpeg,"-nostdin","-hide_banner","-loglevel","error","-n",
                              "-i",str(dest),"-ar","48000","-ac","1","-c:a","pcm_f32le",str(preview)]
                decoded = subprocess.run(decode_cmd,capture_output=True,text=True)
                commands.append(dict(command=decode_cmd,returncode=decoded.returncode,stderr=decoded.stderr))
                (out/"commands.json").write_text(json.dumps(commands,ensure_ascii=False,indent=2),encoding="utf-8")
                decoded.check_returncode()
                variant.update(playback_url=preview.relative_to(out).as_posix(),
                               playback_sha256=sha256_file(preview),
                               playback_duration_sec=sf.info(preview).duration)
            record["variants"].append(variant)
            manifest.append(dict(source,**{k:v for k,v in variant.items() if k not in ("key","label")},
                                 channel=key,created_at=now_iso()))
        records.append(record)
    (out/"inputs.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in manifest),encoding="utf-8")
    payload = dict(run_id=out.name,created_at=now_iso(),records=records,pause_sec=.45,
                   note="Новая серия для прослушивания; прежние оценки детектора к этим файлам не относятся.")
    (out/"listening.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    page = template.read_text(encoding="utf-8").replace("__DATA__",json.dumps(payload,ensure_ascii=False).replace("</","<\\/"))
    (out/"listen.html").write_text(page,encoding="utf-8")
    (out/"experiment.json").write_text(json.dumps(dict(created_at=now_iso(),sources_sha256=sha256_file(args.sources),
         source_manifest=str(args.sources.resolve()),checks_sha256=sha256_file(args.checks),
         script_sha256=sha256_file(Path(__file__)),template_sha256=sha256_file(template),
         generations=len(sources),files=len(manifest),playback_padding_sec=.45,
         normalization="loudnorm=I=-23:LRA=7:TP=-2",endings_recognized_before_padding=True),
         ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Prepared {len(manifest)} listening files: {out/'listen.html'}",flush=True)


if __name__=="__main__":
    main()
