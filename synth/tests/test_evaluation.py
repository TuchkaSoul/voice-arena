import json
import pytest

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


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_dataset_groups_do_not_mix_voices_runs_or_corpora(tmp_path):
    datasets = [("r2", "f5_ru", "long"), ("r4", "f5_ru", "long"),
                ("r4", "f5_matv", "long"), ("r4", "f5_matv", "numeric")]
    rows = [{"run_id": run, "preset": preset, "corpus": corpus,
             "path": f"{run}/{preset}/{corpus}/L0001.wav", "channel": "clean", "label": "spoof"}
            for run, preset, corpus in datasets]
    manifests = tmp_path / "input.jsonl"
    results = tmp_path / "results.jsonl"
    write_jsonl(manifests, rows)
    write_jsonl(results, [{"file": str(tmp_path / row["path"]), "verdict": verdict}
                          for row, verdict in zip(rows, ["bonafide", "spoof", "spoof", "bonafide"])])
    separate = summarize([manifests], results, tmp_path, by_dataset=True)
    assert {(r.run_id, r.preset, r.corpus): r.mistakes for r in separate} == {
        datasets[0]: 1, datasets[1]: 0, datasets[2]: 0, datasets[3]: 1}
    combined = summarize([manifests], results, tmp_path)
    assert len(combined) == 1
    assert combined[0].evaluated == 4
    assert combined[0].mistakes == 2


def test_model_and_final_verdicts_are_counted_independently(tmp_path):
    manifests = tmp_path / "input.jsonl"
    results = tmp_path / "results.jsonl"
    pairs = [("spoof", 0.03), ("bonafide", 0.96), ("spoof", 0.99),
             ("bonafide", 0.5), ("bonafide", None), ("spoof", float("nan")),
             ("spoof", 1.1), ("spoof", "0.9"), ("spoof", True)]
    write_jsonl(manifests, [{"path": str(tmp_path / f"{i}.wav"), "channel": "clean", "label": "spoof"}
                           for i in range(len(pairs) + 3)])
    write_jsonl(results, [{"file": str(tmp_path / f"{i}.wav"), "verdict": verdict, "model_score": score}
                         for i, (verdict, score) in enumerate(pairs)] + [
        {"file": str(tmp_path / "9.wav"), "error": "decode failed", "model_score": 0.99},
        {"file": str(tmp_path / "10.wav"), "verdict": "unknown", "model_score": 0.99},
    ])
    row, = summarize([manifests], results)
    assert row.evaluated == 9
    assert row.errors == 3  # ошибка декодирования, неизвестный класс, отсутствующая строка
    assert row.model_evaluated == 4
    assert row.model_spoof == 2
    assert row.verdict_changes == 2
    assert row.mistakes == 3


def test_custom_model_threshold_changes_diagnostics_only(tmp_path):
    manifests = tmp_path / "input.jsonl"
    results = tmp_path / "results.jsonl"
    audio = str(tmp_path / "sample.wav")
    write_jsonl(manifests, [{"path": audio, "channel": "clean", "label": "spoof"}])
    write_jsonl(results, [{"file": audio, "verdict": "spoof", "model_score": 0.6}])
    default, = summarize([manifests], results)
    custom, = summarize([manifests], results, model_threshold=0.7)
    assert default.model_spoof == 1 and default.verdict_changes == 0
    assert custom.model_spoof == 0 and custom.verdict_changes == 1
    assert default.mistakes == custom.mistakes == 0


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_model_threshold_is_rejected(tmp_path, threshold):
    with pytest.raises(ValueError, match="порог нейросети"):
        summarize([tmp_path / "input.jsonl"], tmp_path / "results.jsonl", model_threshold=threshold)


def test_same_path_with_replaced_audio_is_not_counted_as_matching_result(tmp_path):
    manifests = tmp_path / "input.jsonl"
    results = tmp_path / "results.jsonl"
    audio = str(tmp_path / "L0001.wav")
    write_jsonl(manifests, [{"path": audio, "channel": "clean", "label": "spoof", "sha256": "a" * 64}])
    write_jsonl(results, [{"file": audio, "verdict": "bonafide", "audio_sha256": "b" * 64}])
    with pytest.raises(ValueError, match="другим байтам аудио"):
        summarize([manifests], results)
    write_jsonl(results, [{"file": audio, "verdict": "bonafide", "audio_sha256": "a" * 64}])
    row, = summarize([manifests], results)
    assert row.evaluated == 1 and row.mistakes == 1
