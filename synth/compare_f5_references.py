"""Matched comparison of two local conditioning references on the fixed wide Opus channel."""
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

from sweep_f5_margin import dump, read, asr_wer
from synth import manifest, presets


def generate(args):
    out=args.out.resolve()
    selected=presets.load("f5_ru")
    ep=selected.engine_params
    references=[("original",Path(ep["speaker_wav"]),Path(ep["ref_text_file"])),
                ("alternate",args.alternate_audio.resolve(),args.alternate_text.resolve())]
    audit=json.loads(args.reference_audit.read_text(encoding="utf-8"))
    alt=next(r for r in audit["references"] if Path(r["path"]).resolve()==references[1][1])
    if alt["sha256"]!=manifest.sha256_file(references[1][1]) or alt["source_correlation"]<.99:
        raise ValueError("alternate reference origin was not verified")
    if references[1][2].read_text(encoding="utf-8-sig").strip()!=alt["transcript"]:
        raise ValueError("alternate transcript differs from the recorded audit")
    baseline_path=Path("runs/r16-listening-tail-20261018/sources.json")
    baseline={r["utt_id"]:r for r in json.loads(baseline_path.read_text(encoding="utf-8")) if r["voice"]=="f5_ru"}
    ffmpeg=shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg missing")
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/Path(__file__).name)
    for key,value in {"USE_TF":"0","USE_TORCH":"1","HF_HUB_OFFLINE":"1"}.items():
        os.environ.setdefault(key,value)
    import torch
    import torchaudio
    import f5_tts.api
    import f5_tts.infer.utils_infer
    from f5_tts.api import F5TTS
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text, chunk_text
    meta=dict(created_at=manifest.now_iso(),status="planned",seeds=[2207,7811],steps=32,cfg=2,
              extra_generation_sec=args.extra_gen_sec,reference_audit=str(args.reference_audit.resolve()),
              reference_audit_sha256=manifest.sha256_file(args.reference_audit),references={},
              versions={n:importlib.metadata.version(n) for n in ("torch","torchaudio","f5-tts","vocos","numpy","soundfile")},
              hashes={str(p):manifest.sha256_file(p) for p in [Path(__file__),Path("sweep_f5_margin.py"),
                      Path(selected.source),Path(ep["checkpoint"]),Path(ep["vocab"]),
                      Path(ep["vocoder_path"])/"pytorch_model.bin",Path(f5_tts.api.__file__),
                      Path(f5_tts.infer.utils_infer.__file__),baseline_path]},
              ffmpeg_version=subprocess.check_output([ffmpeg,"-version"],text=True))
    dump(out/"generation.json",meta)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=F5TTS(model=ep["model"],ckpt_file=ep["checkpoint"],vocab_file=ep["vocab"],
                vocoder_local_path=ep["vocoder_path"],device=device)
    if model.vocoder.head.istft.padding!="center":
        raise ValueError("this duration calculation requires the pinned center-padding Vocos")
    meta["vocos_center_padding_compensation_frames"]=1
    meta.update(status="running",device=device)
    sources,inputs,commands=[],[],[]
    vocab=set(Path(ep["vocab"]).read_text(encoding="utf-8").splitlines())
    for key,reference,transcript in references:
        ref_text=transcript.read_text(encoding="utf-8-sig").strip()
        if not ref_text or set(ref_text)-vocab:
            raise ValueError("empty transcript or characters absent from F5 vocabulary")
        processed,processed_text=preprocess_ref_audio_text(str(reference),ref_text)
        ref_audio,ref_sr=torchaudio.load(processed)
        if ref_sr!=24000:
            ref_audio=torchaudio.transforms.Resample(ref_sr,24000)(ref_audio)
        ref_frames=ref_audio.shape[-1]//256
        meta["references"][key]=dict(path=str(reference),sha256=manifest.sha256_file(reference),
              text_path=str(transcript),text_sha256=manifest.sha256_file(transcript),text=ref_text,
              processed_reference_frames=ref_frames,processed_reference_sr=24000,
              processed_reference_duration_sec=ref_audio.shape[-1]/24000)
        dump(out/"generation.json",meta)
        for seed in (2207,7811):
            for utt_id,base in baseline.items():
                info=sf.info(base["path"])
                if info.samplerate!=24000 or info.frames%256:
                    raise ValueError("baseline raw duration is not a whole F5 frame")
                target_frames=info.frames//256+round(args.extra_gen_sec*24000/256)
                # F5's fix_duration includes the conditioning prefix. Preserve output frames.
                # Center-padded ISTFT outputs (mel_frames - 1) * hop_length samples.
                duration=float(np.nextafter((ref_frames+target_frames+1)*256/24000,float("inf")))
                params=dict(base["generate_params"],nfe_step=32,cfg_strength=2.,seed=seed,fix_duration=duration)
                max_chars=int(len(processed_text.encode("utf-8"))/(ref_audio.shape[-1]/24000)*(22-ref_audio.shape[-1]/24000))
                if len(chunk_text(base["text_normalized"],max_chars=max_chars))!=1:
                    raise ValueError("matched durations require a single text chunk")
                source_id=f"f5_ru_ref_{key}_{seed}_{utt_id}"
                folder=out/source_id
                folder.mkdir()
                wave,sr,_=model.infer(ref_file=str(reference),ref_text=ref_text,
                                      gen_text=base["text_normalized"],**params)
                wave=np.asarray(wave,dtype=np.float32)
                if wave.ndim!=1 or not len(wave) or not np.isfinite(wave).all() or sr!=24000:
                    raise ValueError("invalid generated waveform")
                if len(wave)!=target_frames*256:
                    sf.write(folder/"duration-mismatch.wav",wave,sr,subtype="FLOAT")
                    raise ValueError(f"generated duration drift: {len(wave)} != {target_frames*256}")
                peak=float(np.max(np.abs(wave)))
                if peak>1:
                    wave/=peak
                raw=folder/"raw.wav"
                sf.write(raw,wave,sr,subtype="FLOAT")
                row=dict(source_id=source_id,voice="f5_ru",reference=key,utt_id=utt_id,seed=seed,
                         text=base["text"],text_normalized=base["text_normalized"],label="spoof",source_run=out.name,
                         generate_params=params,path=raw.as_posix(),sha256=manifest.sha256_file(raw),
                         duration_sec=len(wave)/sr,peak_before_scale=peak,
                         reference_sha256=manifest.sha256_file(reference),target_output_frames=target_frames)
                sources.append(row)
                dump(out/"sources.json",sources)
                target=folder/"wide.ogg"
                cmd=[ffmpeg,"-nostdin","-hide_banner","-loglevel","error","-n","-i",str(raw),
                     "-af","loudnorm=I=-23:LRA=7:TP=-2,apad=pad_dur=0.45,aresample=48000:resampler=soxr:precision=28",
                     "-ar","48000","-ac","1","-c:a","libopus","-b:a","24k","-application","voip",str(target)]
                result=subprocess.run(cmd,capture_output=True,text=True)
                commands.append(dict(command=cmd,returncode=result.returncode,stderr=result.stderr))
                dump(out/"commands.json",commands)
                result.check_returncode()
                info=sf.info(target)
                if info.duration<len(wave)/sr+.35:
                    raise ValueError("encoded file lost its ending")
                inputs.append(dict(row,path=target.as_posix(),sha256=manifest.sha256_file(target),sr=info.samplerate,
                                   duration_sec=info.duration,channel="wide",raw_path=raw.as_posix(),raw_sha256=row["sha256"]))
                (out/"inputs.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in inputs),encoding="utf-8")
                print(f"Generated {source_id}: {len(wave)/sr:.3f}s",flush=True)
    meta.update(status="completed",files=len(sources))
    dump(out/"generation.json",meta)


def summarize(args):
    out=args.out.resolve()
    rows=read(out/"inputs.jsonl")
    results=read(out/"analysis.jsonl")
    qa=json.loads((out/"ending-checks.json").read_text(encoding="utf-8"))
    if qa["sources_sha256"]!=manifest.sha256_file(out/"sources.json"):
        raise ValueError("wrong ending checks")
    endings={r["source_id"]:r for r in qa["rows"]}
    scores={str(Path(r["file"]).resolve()):r for r in results}
    meta=json.loads((out/"analysis.jsonl.metadata.json").read_text(encoding="utf-8"))
    if len(rows)!=16 or len(scores)!=16 or len(results)!=16 or meta["errors"] or meta["results_sha256"]!=manifest.sha256_file(out/"analysis.jsonl"):
        raise ValueError("incomplete or changed results")
    pairs=defaultdict(dict)
    groups=defaultdict(list)
    full=[]
    for row in rows:
        r=scores[str(Path(row["path"]).resolve())]
        ending=endings[row["source_id"]]
        if (manifest.sha256_file(Path(row["path"]))!=row["sha256"] or r.get("audio_sha256")!=row["sha256"] or
                manifest.sha256_file(Path(row["raw_path"]))!=row["raw_sha256"] or ending["sha256"]!=row["raw_sha256"] or
                r.get("duration",0)<=0 or "model_score" not in r):
            raise ValueError("invalid input, ending check or score")
        raw,sr=sf.read(row["raw_path"],dtype="float32")
        tail=raw[-round(.02*sr):]
        item=dict(row,**r,ending_recognized=ending["ending_recognized"],transcription=ending["transcription"],
                  asr_wer=asr_wer(row["text"],ending["transcription"]),
                  last20ms_rms_dbfs=float(20*np.log10(max(float(np.sqrt(np.mean(tail*tail))),1e-12))))
        groups[row["reference"]].append(item)
        pairs[row["utt_id"],row["seed"]][row["reference"]]=item
        full.append(item)
    stats=[]
    comparisons=[]
    for reference,values in sorted(groups.items()):
        stats.append(dict(reference=reference,n=len(values),mean_ml=float(np.mean([r["model_score"] for r in values])),
                          max_ml=max(r["model_score"] for r in values),bonafide=sum(r["verdict"]=="bonafide" for r in values),
                          endings=sum(r["ending_recognized"] for r in values),max_asr_wer=max(r["asr_wer"] for r in values)))
    for (utt_id,seed),pair in sorted(pairs.items()):
        a,b=pair["original"],pair["alternate"]
        if a["target_output_frames"]!=b["target_output_frames"]:
            raise ValueError("unequal synthesis durations")
        comparisons.append(dict(utt_id=utt_id,seed=seed,original_ml=a["model_score"],alternate_ml=b["model_score"],
                                delta=b["model_score"]-a["model_score"],original_verdict=a["verdict"],
                                alternate_verdict=b["verdict"],alternate_ending=b["ending_recognized"]))
    previous=json.loads(Path("runs/r18-f5-margin-validation-20261018/analysis.jsonl.metadata.json").read_text(encoding="utf-8"))
    keys=("source_files","model_files","vad","ecapa","versions","device","model_dtype","attention_implementation","threshold","spoof_idx","id2label")
    provenance={k:meta.get(k)==previous.get(k) for k in keys}
    summary=dict(files=16,audio_hashes_verified=True,provenance_comparison=provenance,groups=stats,pairs=comparisons,per_file=full)
    dump(out/"summary.json",summary)
    lines=["# Сравнение референсов основного голоса", "", "18 октября 2026.", "",
           "32 шага / CFG=2; четыре фразы и два seed. Длительность генерируемой части совпадает в каждой паре. "
           "Канал прежний: SoX 48 кГц → Opus 24 кбит/с voip, loudnorm и пауза 0.45 с.","",
           "| Референс | Файлов | Среднее ML | Максимум ML | Bonafide | Окончания | Макс. WER ASR |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in stats:
        lines.append(f'| {s["reference"]} | {s["n"]} | {s["mean_ml"]:.4f} | {s["max_ml"]:.4f} | {s["bonafide"]}/{s["n"]} | {s["endings"]}/{s["n"]} | {s["max_asr_wer"]:.3f} |')
    lines += ["", "| Фраза | Seed | Прежний ML | Другой ML | Изменение | Другой: вердикт | Окончание |",
              "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for r in comparisons:
        lines.append(f'| {r["utt_id"]} | {r["seed"]} | {r["original_ml"]:.4f} | {r["alternate_ml"]:.4f} | {r["delta"]:+.4f} | {r["alternate_verdict"]} | {r["alternate_ending"]} |')
    lines += ["",f"Код, веса и окружение анализатора совпали с r18: {all(provenance.values())}.","",
              "Последние слова проверены ASR до добавления паузы. Расшифровка нового референса автоматическая, "
              "не подтверждена человеком. WER — расхождение ASR с текстом, не оценка естественности. "
              "Сходство тембра и качество на слух не измерены. Все примеры синтетические, живого контроля нет. "
              "Проверен один альтернативный фрагмент, выбранный до оценки детектором."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=("generate","summarize"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--alternate-audio",type=Path)
    parser.add_argument("--alternate-text",type=Path)
    parser.add_argument("--reference-audit",type=Path)
    parser.add_argument("--extra-gen-sec",type=float,default=0)
    args=parser.parse_args()
    if not np.isfinite(args.extra_gen_sec) or args.extra_gen_sec<0:
        parser.error("invalid extra duration")
    if args.stage=="generate" and not all((args.alternate_audio,args.alternate_text,args.reference_audit)):
        parser.error("generate requires alternate audio, text and reference audit")
    (generate if args.stage=="generate" else summarize)(args)
