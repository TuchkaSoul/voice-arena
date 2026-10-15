from pathlib import Path

from synth import corpus, presets, text


def test_numeric_corpus_has_stable_ids_and_reaches_f5_as_words():
    root = Path(__file__).resolve().parents[1]
    phrases = corpus.load(root / "corpus/phrases_numeric_v1.txt", prefix="N")
    preset = presets.load("f5_ru", root)
    assert [p.utt_id for p in phrases] == [f"N{i:04d}" for i in range(1, 13)]
    normalized = [text.normalize(p.text, **preset.text_params) for p in phrases]
    assert all(not any(char.isdigit() or char == "°" for char in s) for s in normalized)
    assert "к пятнадцатому марта" in normalized[7].lower()
    assert "с девяти часов пяти минут" in normalized[4].lower()
    assert "минус ноль целых пять десятых градуса цельсия" in normalized[6].lower()
