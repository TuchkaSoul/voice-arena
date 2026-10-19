"""Locate local voice fragments in their source recording and transcribe them offline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import correlate

from synth.manifest import sha256_file, now_iso


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--segments",nargs="+",help="optional start_seconds:duration_seconds crops")
    args=parser.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/Path(__file__).name)
    recording=Path("voices/Запись.m4a")
    ffmpeg=shutil.which("ffmpeg")
    cmd=[ffmpeg,"-v","error","-i",str(recording),"-ar","16000","-ac","1","-f","f32le","pipe:1"]
    decoded=subprocess.run(cmd,capture_output=True,check=True)
    full=np.frombuffer(decoded.stdout,dtype="<f4")
    report=dict(created_at=now_iso(),original=str(recording.resolve()),original_sha256=sha256_file(recording),
                decode_command=cmd,original_duration_sec=len(full)/16000,script_sha256=sha256_file(Path(__file__)),
                references=[])
    references=[Path(f"voices/speaker_ru_{n}.wav") for n in (1,2,3)]
    if args.segments:
        references=[]
        report["crop_commands"]=[]
        for i, value in enumerate(args.segments,1):
            start,duration=map(float,value.split(":"))
            if start<0 or not 0<duration<=12:
                raise ValueError("expected nonnegative start and duration <=12s")
            path=out/f"candidate_{i}.wav"
            command=[ffmpeg,"-nostdin","-n","-v","error","-ss",str(start),"-i",str(recording),
                     "-t",str(duration),"-ar","16000","-ac","1","-c:a","pcm_s16le",str(path)]
            subprocess.run(command,capture_output=True,check=True)
            references.append(path)
            report["crop_commands"].append(command)
    for key,value in {"USE_TF":"0","USE_TORCH":"1","HF_HUB_OFFLINE":"1"}.items():
        os.environ.setdefault(key,value)
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    model_dir=Path("models/whisper-base")
    processor=WhisperProcessor.from_pretrained(model_dir,local_files_only=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=WhisperForConditionalGeneration.from_pretrained(model_dir,local_files_only=True).to(device).eval()
    report.update(asr_source=json.loads((model_dir/"source.json").read_text(encoding="utf-8")),
                  asr_weights_sha256=sha256_file(model_dir/"model.safetensors"),num_beams=5)
    for path in references:
        ref,sr=sf.read(path,dtype="float32")
        if sr!=16000 or ref.ndim!=1:
            raise ValueError("expected mono 16 kHz reference")
        # Match at 2 kHz first; no detector scores participate in this audit.
        a,b=full[::8].astype(np.float64),ref[::8].astype(np.float64)
        x=correlate(a,b,mode="valid",method="fft")
        energy=np.concatenate(([0.],np.cumsum(a*a)))
        window_energy=energy[len(b):]-energy[:-len(b)]
        normalized=x/np.sqrt(np.maximum(window_energy*np.sum(b*b),1e-30))
        coarse=int(np.argmax(normalized))*8
        fine=[]
        for start in range(max(0,coarse-8),min(len(full)-len(ref),coarse+8)+1):
            segment=full[start:start+len(ref)]
            corr=float(np.dot(segment.astype(np.float64),ref.astype(np.float64))/
                       max(np.linalg.norm(segment.astype(np.float64))*np.linalg.norm(ref.astype(np.float64)),1e-30))
            fine.append((corr,start))
        corr,start=max(fine)
        inputs=processor(ref,sampling_rate=sr,return_tensors="pt",return_attention_mask=True)
        with torch.inference_mode():
            output=model.generate(inputs.input_features.to(device),attention_mask=inputs.attention_mask.to(device),
                                  language="ru",task="transcribe",num_beams=5,do_sample=False)
        transcript=processor.batch_decode(output,skip_special_tokens=True)[0].strip()
        (out/f"{path.stem}.asr.txt").write_text(transcript+"\n",encoding="utf-8")
        tail=ref[-int(.02*sr):]
        row=dict(path=str(path.resolve()),sha256=sha256_file(path),sr=sr,duration_sec=len(ref)/sr,
                 source_offset_sec=start/sr,source_correlation=corr,transcript=transcript,
                 last20ms_rms_dbfs=float(20*np.log10(max(float(np.sqrt(np.mean(tail*tail))),1e-12))))
        report["references"].append(row)
        (out/"audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f'{path.name}: offset={start/sr:.3f}s; correlation={corr:.6f}; {transcript}',flush=True)


if __name__=="__main__":
    main()
