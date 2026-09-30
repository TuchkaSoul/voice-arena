"""Нормализация русского текста перед синтезом.

Это единственный модуль, который мы пишем сами от начала до конца, и
он реально влияет на результат: «2450 рублей» почти любая модель
прочитает как набор цифр или молча пропустит, а анализатор потом
посчитает это ошибкой распознавания.

Важно, что нормализованный текст попадает в манифест отдельным полем.
Сторона анализа считает WER относительно него, а не относительно
исходной строки — иначе мы будем штрафовать ASR за то, что синтезатор
вообще не произносил.
"""
from __future__ import annotations

import re
import unicodedata

# --- числительные -----------------------------------------------------------

_ONES = {
    "m": ("", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"),
    "f": ("", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"),
}
_TEENS = (
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
)
_TENS = (
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят",
    "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
)
_HUNDREDS = (
    "", "сто", "двести", "триста", "четыреста",
    "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот",
)

# (1 штука, 2-4 штуки, 5-0 штук)
_GROUPS = (
    (1_000_000_000, "m", ("миллиард", "миллиарда", "миллиардов")),
    (1_000_000, "m", ("миллион", "миллиона", "миллионов")),
    (1_000, "f", ("тысяча", "тысячи", "тысяч")),
)


def plural(n: int, forms: tuple[str, str, str]) -> str:
    """Русское согласование: 1 рубль, 2 рубля, 5 рублей, 11 рублей."""
    tail = abs(n) % 100
    if 11 <= tail <= 19:
        return forms[2]
    tail %= 10
    if tail == 1:
        return forms[0]
    if 2 <= tail <= 4:
        return forms[1]
    return forms[2]


def _triple(n: int, gender: str) -> list[str]:
    """Число от 1 до 999 словами."""
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words.append(_HUNDREDS[hundreds])
    if 10 <= rest <= 19:
        words.append(_TEENS[rest - 10])
    else:
        tens, ones = divmod(rest, 10)
        if tens:
            words.append(_TENS[tens])
        if ones:
            words.append(_ONES[gender][ones])
    return words


def number_to_words(n: int, gender: str = "m") -> str:
    """Целое число словами. gender — род существительного после числа."""
    if n < 0:
        return "минус " + number_to_words(-n, gender)
    if n == 0:
        return "ноль"

    words: list[str] = []
    rest = n
    for base, group_gender, forms in _GROUPS:
        count, rest = divmod(rest, base)
        if not count:
            continue
        # «тысяча рублей», а не «одна тысяча рублей» — так говорят по-русски
        if not (count == 1 and base == 1_000):
            words += _triple(count, group_gender)
        words.append(plural(count, forms))
    if rest:
        words += _triple(rest, gender)

    return " ".join(w for w in words if w)


# --- единицы, которые требуют согласования ----------------------------------

_UNITS: tuple[tuple[str, tuple[str, str, str], str], ...] = (
    (r"%", ("процент", "процента", "процентов"), "m"),
    (r"руб\.?|₽", ("рубль", "рубля", "рублей"), "m"),
    (r"коп\.?", ("копейка", "копейки", "копеек"), "f"),
    (r"кг", ("килограмм", "килограмма", "килограммов"), "m"),
    (r"км", ("километр", "километра", "километров"), "m"),
    (r"мин\.?", ("минута", "минуты", "минут"), "f"),
    (r"сек\.?", ("секунда", "секунды", "секунд"), "f"),
    (r"час(?:ов|а)?", ("час", "часа", "часов"), "m"),
    (r"градус(?:ов|а)?", ("градус", "градуса", "градусов"), "m"),
)

# Сокращения без числа рядом. Список растёт из реальных провалов прогонов,
# а не из фантазии: услышали ошибку — добавили строчку.
_ABBREV: tuple[tuple[str, str], ...] = (
    (r"\bи\s+т\.\s*д\.", "и так далее"),
    (r"\bи\s+т\.\s*п\.", "и тому подобное"),
    (r"\bт\.\s*е\.", "то есть"),
    (r"\bи\s+др\.", "и другие"),
    (r"\bт\.\s*к\.", "так как"),
    (r"\bсм\.", "смотри"),
    (r"№\s*", "номер "),
    (r"\bг\.\s*(?=[А-ЯЁ])", "город "),
    (r"\+(?=\s*\d)", "плюс "),
    (r"(?<=\d)\s*°\s*C\b", " градусов цельсия"),
    (r"&", " и "),
)


def _join_digit_groups(text: str) -> str:
    """«4 250 000» → «4250000».

    По-русски большие числа принято разбивать пробелами по три цифры. Без
    склейки такое число распадается на три отдельных и читается как
    «четыре двести пятьдесят ноль». Делать это надо до разворачивания
    чисел словами.
    """
    return re.sub(
        r"(?<!\d)(\d{1,3})((?: \d{3})+)(?!\d)",
        lambda m: m.group(1) + m.group(2).replace(" ", ""),
        text,
    )


def _expand_units(text: str) -> str:
    for unit_re, forms, gender in _UNITS:
        pattern = re.compile(rf"(\d+)\s*(?:{unit_re})(?![\w])", re.IGNORECASE)

        def repl(m: re.Match) -> str:
            n = int(m.group(1))
            return f"{number_to_words(n, gender)} {plural(n, forms)}"

        text = pattern.sub(repl, text)
    return text


def _expand_bare_numbers(text: str) -> str:
    """Оставшиеся голые числа. Род угадать нельзя — берём мужской."""
    return re.sub(r"\d+", lambda m: number_to_words(int(m.group())), text)


def _expand_abbrev(text: str) -> str:
    for pattern, replacement in _ABBREV:
        text = re.sub(pattern, replacement, text)
    return text


def _clean_unicode(text: str, keep_yo: bool = True) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")            # неразрывный пробел
    text = re.sub(r"[«»“”„]", '"', text)
    text = re.sub(r"[–—‒]", "-", text)            # разные тире → дефис
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    if not keep_yo:
        text = text.replace("ё", "е").replace("Ё", "Е")
    return text


def normalize(
    text: str,
    expand_numbers: bool = True,
    expand_abbrev: bool = True,
    strip_stress: bool = True,
    keep_yo: bool = True,
) -> str:
    """Приводит текст к виду, который движок сможет произнести.

    strip_stress убирает знак ударения `+`: piper и XTTS его не понимают
    и прочитают вслух как «плюс». Для F5-TTS_RUSSIAN, наоборот, ударения
    нужны — там этот флаг выключается в пресете.
    """
    text = _clean_unicode(text, keep_yo=keep_yo)

    if strip_stress:
        text = re.sub(r"\+(?=[аеёиоуыэюяАЕЁИОУЫЭЮЯ])", "", text)
    if expand_abbrev:
        text = _expand_abbrev(text)
    if expand_numbers:
        text = _join_digit_groups(text)
        text = _expand_units(text)
        text = _expand_bare_numbers(text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


# TODO раунд 1: дробные числа («3,5 секунды»), порядковые («15 марта»,
# сейчас читается как «пятнадцать марта»), римские цифры, латиница
# внутри русского текста. Всё это всплывёт на живом корпусе — добавляем
# по мере того, как слышим ошибки, а не заранее.
