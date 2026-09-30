"""Каркас должен собираться без единой скачанной модели: пресеты
читаются, движки находятся в реестре, корпус парсится."""
from pathlib import Path

import pytest

from synth import corpus, presets
from synth.engines import registry

ROOT = Path(__file__).resolve().parents[1]


def test_registry_lists_all_engines():
    assert registry.available() == ["chatterbox", "piper", "xtts"]


def test_unknown_engine_names_the_alternatives():
    with pytest.raises(KeyError) as e:
        registry.create("wav2wav", {})
    assert "piper" in str(e.value)


def test_chatterbox_is_an_honest_stub():
    engine = registry.create("chatterbox", {})
    with pytest.raises(NotImplementedError, match="Perth"):
        engine.load()


def test_open_presets_load_and_reference_known_engines():
    found = presets.discover(ROOT)
    assert found, "в presets/open/ нет ни одного пресета"
    for preset in found:
        assert preset.engine in registry.available(), preset.name


def test_corpus_ids_are_stable_and_comments_skipped():
    phrases = corpus.load(ROOT / "corpus" / "phrases_v1.txt")
    assert phrases[0].utt_id == "p0001"
    assert len(phrases) == len({p.utt_id for p in phrases})
    assert not any(p.text.startswith("#") for p in phrases)
