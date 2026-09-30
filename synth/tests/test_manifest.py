"""Манифест — контракт между двумя командами, поэтому его формат
проверяется отдельно: если он поедет, поедут все метрики проекта."""
import json

from synth import manifest


def _row(**over):
    base = dict(
        utt_id="p0001", text="привет", text_normalized="привет",
        label=manifest.LABEL_SPOOF, engine="piper", preset="piper_irina",
        channel="clean", path="runs/r0/piper_irina/clean/p0001.wav",
        sr=16000, engine_sr=22050, duration_sec=1.25, sha256="deadbeef",
        run_id="r0", seed=1337, created_at=manifest.now_iso(),
    )
    base.update(over)
    return manifest.ManifestRow(**base)


def test_roundtrip(tmp_path):
    rows = [_row(), _row(utt_id="p0002", label=manifest.LABEL_BONAFIDE)]
    path = manifest.write(tmp_path / "r0.jsonl", rows)

    loaded = manifest.read(path)
    assert len(loaded) == 2
    assert loaded[0]["utt_id"] == "p0001"
    assert loaded[1]["label"] == manifest.LABEL_BONAFIDE


def test_schema_stamped_on_every_line(tmp_path):
    path = manifest.write(tmp_path / "r0.jsonl", [_row()])
    line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert line["schema"] == manifest.SCHEMA


def test_russian_text_not_escaped(tmp_path):
    path = manifest.write(tmp_path / "r0.jsonl", [_row()])
    assert "привет" in path.read_text(encoding="utf-8")


def test_broken_line_names_the_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"ok": 1}\nне json\n', encoding="utf-8")
    try:
        manifest.read(path)
    except ValueError as e:
        assert ":2" in str(e)
    else:
        raise AssertionError("битая строка должна ломать чтение")


def test_sha256_of_file(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"")
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert manifest.sha256_file(f) == empty
