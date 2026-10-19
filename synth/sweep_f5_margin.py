"""Compare F5 steps/CFG on one fixed channel and duration; run from synth/."""
import argparse
from collections import defaultdict
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf

from generate_channel_validation import PHRASES
from check_listening_endings import tokens
from synth import manifest, presets


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def read(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def asr_wer(expected, actual):
    a,b=tokens(expected),tokens(actual)
    previous=list(range(len(b)+1))
    for i,left in enumerate(a,1):
        current=[i]
        for j,right in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(left!=right)))
        previous=current
    return previous[-1]/max(1,len(a))


def generate(args):
    out = args.out.resolve()
    profiles = []
    for value in args.profiles:
        steps, cfg = value.split(":")
        steps, cfg = int(steps), float(cfg)
        if steps <= 0 or not np.isfinite(cfg) or cfg <= 0:
            raise ValueError("invalid steps/CFG")
        profiles.append((f'n{steps}_c{cfg:g}'.replace('.', 'p'), steps, cfg))
    if len(set(profiles)) != len(profiles):
        raise ValueError("duplicate profile")
    baseline_sources = Path("runs/r16-listening-tail-20261018/sources.json")
    original = {r["utt_id"]: r for r in json.loads(baseline_sources.read_text(encoding="utf-8"))
                if r["voice"] == "f5_ru"}
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg missing")
    preset = presets.load("f5_ru")
    ep = preset.engine_params
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    for key, value in {"USE_TF":"0", "USE_TORCH":"1", "HF_HUB_OFFLINE":"1"}.items():
        os.environ.setdefault(key, value)
    import torch
    import f5_tts.api
    import f5_tts.infer.utils_infer
    from f5_tts.api import F5TTS

    meta = dict(created_at=manifest.now_iso(), status="planned", profiles=profiles,
                utterances=args.utterances, seeds=args.seeds, preset=ep,
                fixed_duration_source=baseline_sources.as_posix(),
                fixed_duration_source_sha256=manifest.sha256_file(baseline_sources),
                target_max_model_score=.30, playback_padding_sec=.45,
                versions={n:importlib.metadata.version(n) for n in ("torch","f5-tts","vocos","numpy","soundfile")},
                hashes={str(p):manifest.sha256_file(p) for p in
                        [Path(__file__), Path(preset.source), Path(ep["checkpoint"]), Path(ep["vocab"]),
                         Path(ep["vocoder_path"])/"pytorch_model.bin", Path(ep["speaker_wav"]),
                         Path(ep["ref_text_file"]), Path(f5_tts.api.__file__), Path(f5_tts.infer.utils_infer.__file__)]},
                ffmpeg_version=subprocess.check_output([ffmpeg,"-version"],text=True))
    dump(out/"generation.json", meta)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = F5TTS(model=ep["model"], ckpt_file=ep["checkpoint"], vocab_file=ep["vocab"],
                vocoder_local_path=ep["vocoder_path"], device=device)
    reference_text = Path(ep["ref_text_file"]).read_text(encoding="utf-8-sig").strip()
    meta.update(status="running", device=device)
    dump(out/"generation.json", meta)
    sources, inputs, commands = [], [], []
    for profile, steps, cfg in profiles:
        for seed in args.seeds:
            for utt_id in args.utterances:
                base = original[utt_id]
                params = dict(base["generate_params"], nfe_step=steps, cfg_strength=cfg, seed=seed)
                source_id = f"f5_ru_{profile}_{seed}_{utt_id}"
                folder = out/source_id
                folder.mkdir()
                wav, sr, _ = tts.infer(ref_file=ep["speaker_wav"], ref_text=reference_text,
                                       gen_text=base["text_normalized"], **params)
                audio = np.asarray(wav, dtype=np.float32)
                if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
                    raise ValueError("invalid generated audio")
                peak = float(np.max(np.abs(audio)))
                if peak > 1:
                    audio /= peak
                raw = folder/"raw.wav"
                sf.write(raw, audio, sr, subtype="FLOAT")
                source = dict(source_id=source_id, voice="f5_ru", profile=profile,
                              nfe_step=steps, cfg_strength=cfg, utt_id=utt_id, seed=seed,
                              text=base["text"], text_normalized=base["text_normalized"], label="spoof",
                              source_run=out.name, generate_params=params, path=raw.as_posix(),
                              sha256=manifest.sha256_file(raw), duration_sec=len(audio)/sr, peak_before_scale=peak)
                sources.append(source)
                dump(out/"sources.json", sources)
                target = folder/"wide.ogg"
                cmd = [ffmpeg,"-nostdin","-hide_banner","-loglevel","error","-n","-i",str(raw),
                       "-af","loudnorm=I=-23:LRA=7:TP=-2,apad=pad_dur=0.45,aresample=48000:resampler=soxr:precision=28",
                       "-ar","48000","-ac","1","-c:a","libopus","-b:a","24k","-application","voip",str(target)]
                result = subprocess.run(cmd,capture_output=True,text=True)
                commands.append(dict(command=cmd,returncode=result.returncode,stderr=result.stderr))
                dump(out/"commands.json",commands)
                result.check_returncode()
                info = sf.info(target)
                if info.duration < source["duration_sec"]+.35:
                    raise ValueError("encoded file lost its tail")
                inputs.append(dict(source,path=target.as_posix(),sha256=manifest.sha256_file(target),
                                   raw_path=raw.as_posix(),raw_sha256=source["sha256"],
                                   duration_sec=info.duration,sr=info.samplerate,channel="wide"))
                (out/"inputs.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in inputs),encoding="utf-8")
                print(f"Generated {source_id}: {len(audio)/sr:.3f}s",flush=True)
    meta.update(status="completed",files=len(sources))
    dump(out/"generation.json",meta)


def summarize(args):
    out=args.out.resolve()
    rows=read(out/"inputs.jsonl")
    results=read(out/"analysis.jsonl")
    checks=json.loads((out/"ending-checks.json").read_text(encoding="utf-8"))
    if checks["sources_sha256"] != manifest.sha256_file(out/"sources.json"):
        raise ValueError("wrong ending checks")
    endings={r["source_id"]:r for r in checks["rows"]}
    scored={str(Path(r["file"]).resolve()):r for r in results}
    meta=json.loads((out/"analysis.jsonl.metadata.json").read_text(encoding="utf-8"))
    if len(rows)!=len(results) or len(scored)!=len(results) or meta["errors"] or meta["results_sha256"]!=manifest.sha256_file(out/"analysis.jsonl"):
        raise ValueError("missing or changed results")
    groups=defaultdict(list)
    full=[]
    for row in rows:
        r=scored[str(Path(row["path"]).resolve())]
        qa=endings[row["source_id"]]
        if (r.get("audio_sha256")!=row["sha256"] or manifest.sha256_file(Path(row["path"]))!=row["sha256"] or
                manifest.sha256_file(Path(row["raw_path"]))!=row["raw_sha256"] or qa["sha256"]!=row["raw_sha256"] or
                r.get("duration",0)<=0 or "model_score" not in r):
            raise ValueError(f"invalid score or source: {row['source_id']}")
        item=dict(row,**r,ending_recognized=qa["ending_recognized"],transcription=qa["transcription"],
                  asr_wer=asr_wer(row["text"],qa["transcription"]))
        groups[row["profile"]].append(item)
        full.append(item)
    stats=[]
    for profile, values in groups.items():
        stats.append(dict(profile=profile,nfe_step=values[0]["nfe_step"],cfg_strength=values[0]["cfg_strength"],
                          n=len(values),max_model_score=max(r["model_score"] for r in values),
                          mean_model_score=float(np.mean([r["model_score"] for r in values])),
                          max_asr_wer=max(r["asr_wer"] for r in values),
                          all_endings_recognized=all(r["ending_recognized"] for r in values),
                          all_bonafide=all(r["verdict"]=="bonafide" for r in values)))
    eligible=[s for s in stats if s["all_endings_recognized"] and s["all_bonafide"]]
    selected=min(eligible,key=lambda s:(s["max_model_score"],s["mean_model_score"])) if eligible else None
    validation=None
    if args.validation_of:
        frozen=json.loads((args.validation_of/"summary.json").read_text(encoding="utf-8"))["selected"]
        if frozen is None:
            raise ValueError("no frozen candidate")
        selected=next(s for s in stats if s["profile"]==frozen["profile"])
        new_conditions=[r for r in full if r["profile"]==frozen["profile"] and
                        (r["seed"]!=2207 or r["utt_id"] not in ("V0001","V0003"))]
        validation=dict(frozen_selection_from=args.validation_of.as_posix(),new_conditions=len(new_conditions),
                        max_model_score_new=max(r["model_score"] for r in new_conditions),
                        all_endings_new=all(r["ending_recognized"] for r in new_conditions),
                        all_bonafide_new=all(r["verdict"]=="bonafide" for r in new_conditions))
        paired=defaultdict(dict)
        for r in full:
            paired[r["utt_id"],r["seed"]][r["profile"]]=r
        comparisons=[]
        for (utt_id,seed), pair in sorted(paired.items()):
            a,b=pair["n32_c2"],pair[frozen["profile"]]
            comparisons.append(dict(utt_id=utt_id,seed=seed,baseline_ml=a["model_score"],
                                    candidate_ml=b["model_score"],delta=b["model_score"]-a["model_score"],
                                    baseline_verdict=a["verdict"],candidate_verdict=b["verdict"],
                                    baseline_ending=a["ending_recognized"],candidate_ending=b["ending_recognized"],
                                    outside_selection=seed!=2207 or utt_id not in ("V0001","V0003")))
        validation.update(pairs=comparisons,
                          improved=sum(r["delta"]<0 for r in comparisons),
                          worsened=sum(r["delta"]>0 for r in comparisons),
                          min_candidate_margin=meta["threshold"]-selected["max_model_score"])
    previous=json.loads(Path("runs/r16-listening-page-20261018-v2/detector-comparison/analysis.jsonl.metadata.json").read_text(encoding="utf-8"))
    keys=("source_files","model_files","vad","ecapa","versions","device","model_dtype","attention_implementation","threshold","spoof_idx","id2label")
    provenance={k:meta.get(k)==previous.get(k) for k in keys}
    summary=dict(files=len(full),groups=stats,selected=selected,validation=validation,provenance_comparison=provenance,
                 audio_hashes_verified=True,per_file=full)
    dump(out/"summary.json",summary)
    lines=["# Шаги и CFG: оценка нейросети", "", "18 октября 2026.", "",
           "Новый канал фиксирован: SoX 48 кГц → Opus 24 кбит/с voip. Длительности и запас генерации — как в r16.","",
           "| Шаги | CFG | Файлов | Максимум ML | Среднее ML | Макс. WER ASR | Окончания распознаны | Все bonafide |",
           "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for s in sorted(stats,key=lambda s:(s["nfe_step"],s["cfg_strength"])):
        lines.append(f'| {s["nfe_step"]} | {s["cfg_strength"]} | {s["n"]} | {s["max_model_score"]:.4f} | {s["mean_model_score"]:.4f} | {s["max_asr_wer"]:.3f} | {s["all_endings_recognized"]} | {s["all_bonafide"]} |')
    label="Ранее зафиксированный кандидат" if validation else "Выбор по максимуму, затем среднему ML"
    lines += ["",f"{label}: {selected['profile'] if selected else 'нет допустимого кандидата'}.",
              "",f"Код, веса и окружение анализатора совпали с r16: {all(provenance.values())}.","",
              "ASR проверяет последние слова сырых файлов, до паузы. WER — расхождение расшифровки с текстом, не подтверждённые на слух ошибки. Это не оценка естественности. "
              "Подбор по знакомому детектору требует отдельной проверки на других фразах и seed. "
              "Полные оценки и расшифровки — summary.json; источники и параметры — sources.json и generation.json."]
    if validation:
        lines += ["",f'Новых условий вне подбора: {validation["new_conditions"]}; '
                  f'максимум ML {validation["max_model_score_new"]:.4f}; '
                  f'окончания распознаны: {validation["all_endings_new"]}; '
                  f'все bonafide: {validation["all_bonafide_new"]}.']
        lines += ["", "| Фраза | Seed | База ML | Кандидат ML | Изменение | Кандидат: вердикт | Окончание |",
                  "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
        for r in validation["pairs"]:
            lines.append(f'| {r["utt_id"]} | {r["seed"]} | {r["baseline_ml"]:.4f} | {r["candidate_ml"]:.4f} | '
                         f'{r["delta"]:+.4f} | {r["candidate_verdict"]} | {r["candidate_ending"]} |')
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=("generate","summarize"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--profiles",nargs="+",default=[f"{n}:{c}" for n in (32,48,64) for c in (1.5,2.0,2.5)])
    parser.add_argument("--utterances",nargs="+",choices=list(PHRASES),default=["V0001","V0003"])
    parser.add_argument("--seeds",nargs="+",type=int,default=[2207])
    parser.add_argument("--validation-of",type=Path)
    args=parser.parse_args()
    (generate if args.stage=="generate" else summarize)(args)
