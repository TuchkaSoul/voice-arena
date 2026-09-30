import json

from synth.evaluation import summarize


def test_synthesis_misses_and_live_false_alarms_are_separate(tmp_path):
    synth = tmp_path / "synthetic.wav"
    live = tmp_path / "live.wav"
    synthetic_manifest = tmp_path / "synth.jsonl"
    live_manifest = tmp_path / "live.jsonl"
    synthetic_manifest.write_text(json.dumps({
        "path": str(synth), "channel": "clean", "label": "spoof"
    }) + "\n", encoding="utf-8")
    live_manifest.write_text(json.dumps({
        "path": str(live), "channel": "clean", "label": "bonafide"
    }) + "\n", encoding="utf-8")
    results = tmp_path / "scores.jsonl"
    results.write_text("\n".join(json.dumps(row) for row in [
        {"file": str(synth), "verdict": "bonafide"},
        {"file": str(live), "verdict": "spoof"},
    ]) + "\n", encoding="utf-8")

    summary = summarize([synthetic_manifest, live_manifest], results)
    by_label = {row.label: row for row in summary}
    assert by_label["spoof"].mistakes == 1
    assert by_label["bonafide"].mistakes == 1
    assert all(row.error_rate == 1.0 for row in summary)
