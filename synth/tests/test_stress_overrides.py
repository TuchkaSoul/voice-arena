from pathlib import Path

import pytest

from synth import corpus, presets
from synth.text import normalize


@pytest.mark.parametrize("source, expected", [
    ("к зиме", "к зим+е"),
    ("Зиме", "Зим+е"),
    ("ЗИМЕ", "ЗИМ+Е"),
    ("зиме, зиме.", "зим+е, зим+е."),
])
def test_manual_stress_preserves_case(source, expected):
    assert normalize(source, strip_stress=False, stress_overrides={"зиме": "зим+е"}) == expected


def test_dictionary_does_not_change_other_forms_or_existing_stress():
    source = "зима, зимой, зим+е, з+име, назиме"
    assert normalize(source, strip_stress=False, stress_overrides={"зиме": "зим+е"}) == source


def test_piper_and_xtts_do_not_receive_stress_hints():
    assert normalize("к зиме", stress_overrides={"зиме": "зим+е"}) == "к зиме"


@pytest.mark.parametrize("marked", ["зим+а", "+зиме", "зим++е"])
def test_stress_hint_cannot_change_the_word_or_mark_a_consonant(marked):
    with pytest.raises(ValueError, match="подсказка ударения"):
        normalize("к зиме", strip_stress=False, stress_overrides={"зиме": marked})


def test_f5_preset_marks_winter_without_modifying_the_original_corpus():
    root = Path(__file__).resolve().parents[1]
    phrase = corpus.load(root / "corpus/phrases_long_v1.txt", prefix="L")[0]
    params = presets.load("f5_ru", root).text_params
    assert "к зиме" in phrase.text
    assert "к зим+е" in normalize(phrase.text, **params)
