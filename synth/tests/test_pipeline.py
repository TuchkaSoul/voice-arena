"""Главный инвариант прогона: фраза синтезируется один раз и расходится
по всем каналам.

Ради этого теста движок подменяется заглушкой со счётчиком вызовов.
Настоящие модели здесь не нужны — проверяется порядок ступеней, а не
качество звука.
"""
import numpy as np
import pytest
import soundfile as sf
import subprocess

from synth import channel as channel_mod
from synth import manifest
from synth.corpus import Phrase
from synth.engines import registry
from synth.engines.base import SynthEngine, Utterance
from synth.pipeline import SynthRun, channel_only
from synth.presets import Preset

try:
    channel_mod.ffmpeg_path()
    HAVE_FFMPEG = True
except channel_mod.FfmpegMissing:
    HAVE_FFMPEG = False

needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg не найден в PATH")

CHANNELS = ["clean", "mp3_64", "opus_24k"]
PHRASES = [Phrase("p0001", "Скидка 25% на всё."), Phrase("p0002", "Нет.")]


class CountingEngine(SynthEngine):
    """Считает вызовы и каждый раз отдаёт другой сигнал.

    Разный сигнал нужен специально: если пайплайн синтезирует заново на
    каждый канал, записи одного utt_id разъедутся, и тест это увидит.
    """

    name = "counting"
    calls = 0

    def load(self) -> None:
        pass

    def say(self, text: str) -> Utterance:
        CountingEngine.calls += 1
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        freq = 200 * CountingEngine.calls
        return Utterance(audio=(0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr=sr)


@pytest.fixture
def run(tmp_path, monkeypatch):
    CountingEngine.calls = 0
    monkeypatch.setattr(registry, "create", lambda name, params: CountingEngine(params))
    preset = Preset(name="fake", engine="counting")
    return SynthRun(preset=preset, channels=CHANNELS, out_dir=tmp_path, run_id="rt")


@needs_ffmpeg
def test_synthesis_happens_once_per_phrase(run):
    run.run(PHRASES)
    assert CountingEngine.calls == len(PHRASES), (
        "движок вызван больше раза на фразу — значит каналы пересинтезируют, "
        "и записи в разных каналах разъедутся"
    )


@needs_ffmpeg
def test_every_channel_gets_every_phrase(run):
    rows = run.run(PHRASES)
    assert set(rows) == set(CHANNELS)
    for name in CHANNELS:
        assert [r.utt_id for r in rows[name]] == [p.utt_id for p in PHRASES]
        assert all(r.channel == name for r in rows[name])
        assert run.manifest_path(name).exists()


@needs_ffmpeg
def test_same_utterance_behind_every_channel(run):
    """Один utt_id во всех каналах — это один и тот же звук, только сжатый."""
    rows = run.run(PHRASES)

    for i in range(len(PHRASES)):
        tones = set()
        for name in CHANNELS:
            audio, sr = sf.read(rows[name][i].path)
            spectrum = np.abs(np.fft.rfft(audio[: 2 ** 14]))
            tones.add(round(np.argmax(spectrum) * sr / 2 ** 14 / 10))
        assert len(tones) == 1, f"{PHRASES[i].utt_id}: каналы несут разный звук — {tones}"


@needs_ffmpeg
def test_durations_match_across_channels(run):
    rows = run.run(PHRASES)
    per_utt = zip(*(rows[name] for name in CHANNELS))
    for group in per_utt:
        assert len({r.duration_sec for r in group}) == 1


@needs_ffmpeg
def test_rows_carry_label_and_run_id(run):
    rows = run.run(PHRASES)
    for name in CHANNELS:
        for row in rows[name]:
            assert row.label == manifest.LABEL_SPOOF
            assert row.run_id == "rt"
            assert row.sr == channel_mod.TARGET_SR


@needs_ffmpeg
def test_two_corpora_do_not_overwrite_each_other(tmp_path, monkeypatch):
    """Два набора фраз в одном раунде должны лежать врозь."""
    CountingEngine.calls = 0
    monkeypatch.setattr(registry, "create", lambda name, params: CountingEngine(params))
    preset = Preset(name="fake", engine="counting")

    paths = set()
    for corpus_name in ("short", "long"):
        run = SynthRun(
            preset=preset, channels=["clean"], out_dir=tmp_path,
            run_id="rt", corpus=corpus_name,
        )
        rows = run.run(PHRASES)
        paths.add(run.manifest_path("clean"))
        assert all(r.corpus == corpus_name for r in rows["clean"])

    assert len(paths) == 2, "манифесты двух корпусов совпали — один затрёт другой"
    assert all(p.exists() for p in paths)


def test_unknown_channel_rejected_before_synthesis(tmp_path):
    with pytest.raises(KeyError):
        SynthRun(
            preset=Preset(name="fake", engine="counting"),
            channels=["clean", "mp3_320"],
            out_dir=tmp_path,
            run_id="rt",
        )


def test_empty_channel_list_rejected(tmp_path):
    with pytest.raises(ValueError):
        SynthRun(
            preset=Preset(name="fake", engine="counting"),
            channels=[],
            out_dir=tmp_path,
            run_id="rt",
        )


@needs_ffmpeg
def test_live_m4a_file_gets_bonafide_manifest(tmp_path):
    raw = tmp_path / "source.wav"
    sf.write(raw, np.zeros(22050, dtype=np.float32), 22050)
    m4a = tmp_path / "live.m4a"
    subprocess.run(
        [channel_mod.ffmpeg_path(), "-y", "-loglevel", "error", "-i", str(raw), str(m4a)],
        check=True,
    )
    rows = channel_only(m4a, tmp_path / "runs", "clean", "r1", preset_name="live_control")
    assert len(rows) == 1
    assert rows[0].label == manifest.LABEL_BONAFIDE
    assert rows[0].engine_sr == 22050
    assert rows[0].channel == "clean"
    assert rows[0].duration_sec == pytest.approx(1.0, abs=0.05)
