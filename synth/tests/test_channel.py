"""Канал — единственное место, где мы зависим от внешнего ffmpeg.
Тесты пропускаются, если его нет, но сам факт пропуска виден в выводе."""
import numpy as np
import pytest
import soundfile as sf

from synth import channel

try:
    channel.ffmpeg_path()
    HAVE_FFMPEG = True
except channel.FfmpegMissing:
    HAVE_FFMPEG = False

needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg не найден в PATH")


@pytest.fixture
def tone(tmp_path):
    """Секунда синуса 440 Гц на 24 кГц — как отдаёт XTTS."""
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr, subtype="PCM_16")
    return path


def test_room_channel_is_not_silently_fake():
    with pytest.raises(NotImplementedError, match="раунд 5"):
        channel.get(channel.ROOM)


def test_unknown_channel_lists_known_ones():
    with pytest.raises(KeyError, match="opus_24k"):
        channel.get("mp3_320")


@needs_ffmpeg
@pytest.mark.parametrize("name", ["clean", "flac", "mp3_192", "mp3_64", "opus_24k"])
def test_every_channel_lands_at_16k_mono(tone, tmp_path, name):
    out = channel.apply(tone, tmp_path / "out", name, "p0001")
    assert out.exists() and out.stat().st_size > 0

    info = sf.info(str(out))
    assert info.samplerate == channel.TARGET_SR
    assert info.channels == 1
    assert 0.9 < info.frames / info.samplerate < 1.1


@needs_ffmpeg
def test_compression_actually_compresses(tone, tmp_path):
    clean = channel.apply(tone, tmp_path / "c", "clean", "p0001")
    opus = channel.apply(tone, tmp_path / "o", "opus_24k", "p0001")
    assert opus.stat().st_size < clean.stat().st_size / 4


@needs_ffmpeg
def test_flac_is_bit_identical_to_clean(tone, tmp_path):
    """flac существует ради проверки загрузчика, а не ради иного звука.

    Если он разойдётся с clean хотя бы разрядностью, у двух «одинаковых»
    каналов будет разный шум квантования — и детектор сможет выучить
    именно его.
    """
    wav, _ = sf.read(str(channel.apply(tone, tmp_path / "c", "clean", "p0001")))
    flac, _ = sf.read(str(channel.apply(tone, tmp_path / "f", "flac", "p0001")))
    assert np.array_equal(wav, flac)


@needs_ffmpeg
def test_channels_ordered_by_quality(tone, tmp_path):
    """Размеры должны убывать по мере ухудшения канала — иначе кто-то
    перепутал битрейт в таблице каналов."""
    sizes = [
        channel.apply(tone, tmp_path / name, name, "p0001").stat().st_size
        for name in ("mp3_192", "mp3_128", "mp3_64", "opus_24k", "opus_12k")
    ]
    assert sizes == sorted(sizes, reverse=True), sizes
