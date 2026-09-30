"""Нормализация текста — единственный модуль, который мы пишем сами,
поэтому он же единственный, который можно проверить без моделей."""
import pytest

from synth.text import normalize, number_to_words, plural


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, "ноль"),
        (1, "один"),
        (11, "одиннадцать"),
        (21, "двадцать один"),
        (100, "сто"),
        (247, "двести сорок семь"),
        (1000, "тысяча"),            # не «одна тысяча»
        (2450, "две тысячи четыреста пятьдесят"),
        (5000, "пять тысяч"),
        (1_000_000, "один миллион"),
        (-5, "минус пять"),
    ],
)
def test_number_to_words(n, expected):
    assert number_to_words(n) == expected


def test_number_gender():
    assert number_to_words(2, "f") == "две"
    assert number_to_words(2, "m") == "два"


@pytest.mark.parametrize(
    "n, expected",
    [(1, "рубль"), (2, "рубля"), (5, "рублей"), (11, "рублей"), (21, "рубль"), (104, "рубля")],
)
def test_plural(n, expected):
    assert plural(n, ("рубль", "рубля", "рублей")) == expected


def test_units_agree_with_number():
    assert "двадцать пять процентов" in normalize("Скидка 25% на всё")
    assert "две тысячи четыреста пятьдесят рублей" in normalize("Цена 2450 руб.")
    assert "один процент" in normalize("Всего 1%")


def test_grouped_digits_read_as_one_number():
    # Без склейки «4 250 000» распалось бы на три числа и прочиталось
    # как «четыре двести пятьдесят ноль»
    out = normalize("Выручка составила 4 250 000 рублей.")
    assert "четыре миллиона двести пятьдесят тысяч" in out
    assert "ноль" not in out


def test_grouping_does_not_glue_unrelated_numbers():
    out = normalize("начало в 10 часов 30 минут")
    assert "десять часов" in out and "тридцать минут" in out


@pytest.mark.parametrize(
    "source, expected",
    [
        ("1 мая", "первого мая"),
        ("15 марта", "пятнадцатого марта"),
        ("21 сентября", "двадцать первого сентября"),
        ("31 декабря", "тридцать первого декабря"),
    ],
)
def test_dates_are_read_as_ordinals(source, expected):
    assert expected in normalize(source)


def test_abbreviations():
    assert "так далее" in normalize("Отчёты, справки и т. д.")
    assert "номер" in normalize("Заказ № 178")


def test_stress_marks_stripped():
    # Знак ударения piper и XTTS прочитают вслух как «плюс»
    assert normalize("з+амок") == "замок"
    assert normalize("з+амок", strip_stress=False) == "з+амок"


def test_whitespace_and_quotes():
    assert normalize('  «привет»\n\nмир  ') == '"привет" мир'


def test_yo_preserved_by_default():
    assert "ё" in normalize("ещё")
    assert "ё" not in normalize("ещё", keep_yo=False)
