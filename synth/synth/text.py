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
    (r"&", " и "),
)

_MONTHS = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)
_DAY_ORDINALS = (
    "", "первого", "второго", "третьего", "четвёртого", "пятого",
    "шестого", "седьмого", "восьмого", "девятого", "десятого",
    "одиннадцатого", "двенадцатого", "тринадцатого", "четырнадцатого",
    "пятнадцатого", "шестнадцатого", "семнадцатого", "восемнадцатого",
    "девятнадцатого", "двадцатого", "двадцать первого", "двадцать второго",
    "двадцать третьего", "двадцать четвёртого", "двадцать пятого",
    "двадцать шестого", "двадцать седьмого", "двадцать восьмого",
    "двадцать девятого", "тридцатого", "тридцать первого",
)
_DATE_PREPOSITIONS = {
    "к": "dat", "ко": "dat",
    "на": "acc", "по": "acc", "в": "acc", "во": "acc",
    "о": "prep", "об": "prep", "обо": "prep", "при": "prep",
    "перед": "inst",
    "до": "gen", "с": "gen", "со": "gen", "от": "gen", "после": "gen",
}
_DATE_RE = re.compile(
    rf"\b(?:(?P<prep>{'|'.join(_DATE_PREPOSITIONS)})\s+)?"
    rf"(?P<day>\d{{1,2}})\s+(?P<month>{'|'.join(_MONTHS)})\b", re.IGNORECASE,
)
_DATE_RANGE_RE = re.compile(
    rf"\b(с|со)\s+(\d{{1,2}})\s+(по|до)\s+(\d{{1,2}})\s+({'|'.join(_MONTHS)})\b",
    re.IGNORECASE,
)


def _day_words(day: int, month: str, case: str = "gen") -> str:
    max_day = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[_MONTHS.index(month.lower())]
    if not 1 <= day <= max_day:
        raise ValueError(f"некорректная дата: {day} {month}")
    words = _DAY_ORDINALS[day].split()
    last = words[-1]
    if case != "gen":
        if last == "третьего":
            last = {"dat": "третьему", "acc": "третье", "prep": "третьем", "inst": "третьим"}[case]
        else:
            last = last[:-3] + {"dat": "ому", "acc": "ое", "prep": "ом", "inst": "ым"}[case]
    return " ".join(words[:-1] + [last])


def _expand_dates(text: str) -> str:
    """Число месяца; склоняем последний компонент по ближайшему предлогу."""
    def range_repl(match: re.Match) -> str:
        start_prep, start, end_prep, end, month = match.groups()
        start_words = _day_words(int(start), month)
        end_words = _day_words(int(end), month, _DATE_PREPOSITIONS[end_prep.lower()])
        return f"{start_prep} {start_words} {end_prep} {end_words} {month}"

    text = _DATE_RANGE_RE.sub(range_repl, text)

    def repl(match: re.Match) -> str:
        prep = match.group("prep") or ""
        case = _DATE_PREPOSITIONS.get(prep.lower(), "gen")
        words = _day_words(int(match.group("day")), match.group("month"), case)
        return f"{prep + ' ' if prep else ''}{words} {match.group('month')}"

    return _DATE_RE.sub(repl, text)


_TIME_RE = re.compile(r"(?<![\w:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\w:])")
_SMALL_GENITIVE = (
    "нуля", "одного", "двух", "трёх", "четырёх", "пяти", "шести", "семи", "восьми", "девяти",
    "десяти", "одиннадцати", "двенадцати", "тринадцати", "четырнадцати", "пятнадцати",
    "шестнадцати", "семнадцати", "восемнадцати", "девятнадцати",
)


def _clock_number(n: int, case: str, feminine: bool) -> str:
    if case == "gen":
        if n < 20:
            words = _SMALL_GENITIVE[n]
        else:
            tens, ones = divmod(n, 10)
            words = {2: "двадцати", 3: "тридцати", 4: "сорока", 5: "пятидесяти"}[tens]
            if ones:
                words += " " + _SMALL_GENITIVE[ones]
        if feminine:
            words = re.sub(r"\bодного\b", "одной", words)
        return words
    words = number_to_words(n, "f" if feminine else "m")
    if feminine and case == "acc":
        words = re.sub(r"\bодна\b", "одну", words)
    return words


def _expand_times(text: str) -> str:
    """24-часовое HH:MM[:SS]. Не принимаем 24:00 или неверные минуты."""
    def repl(match: re.Match) -> str:
        hour, minute, second = match.groups()
        values = [int(hour), int(minute)] + ([int(second)] if second is not None else [])
        if values[0] > 23 or any(n > 59 for n in values[1:]):
            raise ValueError(f"некорректное время: {match.group(0)}")
        prefix = re.search(r"\b(в|во|на|с|со|до|после|от)\s+$", text[:match.start()], re.IGNORECASE)
        case = "nom" if prefix is None else ("gen" if prefix.group(1).lower() in {"с", "со", "до", "после", "от"} else "acc")
        units = [("час", "часа", "часов"), ("минута", "минуты", "минут"), ("секунда", "секунды", "секунд")]
        parts = []
        for i, n in enumerate(values):
            word = _clock_number(n, case, feminine=i > 0)
            if case == "gen":
                noun = units[i][1] if n % 10 == 1 and n % 100 != 11 else units[i][2]
            else:
                noun = plural(n, units[i])
                if case == "acc" and i > 0 and noun == units[i][0]:
                    noun = noun[:-1] + "у"
            parts.append(f"{word} {noun}")
        return " ".join(parts)
    return _TIME_RE.sub(repl, text)


_DECIMAL_RE = re.compile(r"(?<![\w.,])(\d+)[,.](\d+)(?![\w]|[.,]\d)")
_DENOMINATORS = (
    ("десятая", "десятых"), ("сотая", "сотых"), ("тысячная", "тысячных"),
    ("десятитысячная", "десятитысячных"), ("стотысячная", "стотысячных"),
    ("миллионная", "миллионных"), ("десятимиллионная", "десятимиллионных"),
    ("стомиллионная", "стомиллионных"), ("миллиардная", "миллиардных"),
)
_DECIMAL_UNIT_NAMES = (
    (r"руб(?:лей|ля|ль)", "рубля"),
    (r"копей(?:ка|ки|ку|ке|кой)|копеек", "копейки"),
    (r"килограмм(?:ов|а)?", "килограмма"),
    (r"километр(?:ов|а)?", "километра"),
    (r"минут(?:а|ы|у|е|ой)?", "минуты"),
    (r"секунд(?:а|ы|у|е|ой)?", "секунды"),
)


def _expand_decimals(text: str) -> str:
    """Запятая/точка, 1–9 знаков; точность, включая нули, сохраняется."""
    def repl(match: re.Match) -> str:
        whole, fraction = match.groups()
        if len(fraction) > len(_DENOMINATORS):
            raise ValueError("дробь содержит больше девяти знаков после разделителя")
        n, tail = int(whole), int(fraction)
        singular, plural_form = _DENOMINATORS[len(fraction) - 1]
        words = (
            f"{number_to_words(n, 'f')} {plural(n, ('целая', 'целых', 'целых'))} "
            f"{number_to_words(tail, 'f')} {plural(tail, (singular, plural_form, plural_form))}"
        )
        return words

    # Падеж единицы у десятичной дроби — родительный единственного числа.
    units = [(unit_re, _fraction_unit(forms)) for unit_re, forms, _ in _UNITS] + list(_DECIMAL_UNIT_NAMES)
    for unit_re, noun in units:
        pattern = re.compile(rf"({_DECIMAL_RE.pattern})\s*(?:{unit_re})(?![\w])", re.IGNORECASE)
        text = pattern.sub(lambda m: repl(_DECIMAL_RE.fullmatch(m.group(1))) + " " + noun, text)
    return _DECIMAL_RE.sub(repl, text)


def _fraction_unit(forms: tuple[str, str, str]) -> str:
    return {"копейка": "копейки", "минута": "минуты", "секунда": "секунды"}.get(forms[0], forms[1])


def _expand_signs(text: str) -> str:
    # Дефис внутри слова/диапазона не является знаком отрицательного числа.
    def minus_repl(match: re.Match) -> str:
        before = text[:match.start()].rstrip()
        return match.group(0) if before and before[-1].isdigit() else "минус "
    text = re.sub(r"(?<!\w)-\s*(?=\d)", minus_repl, text)
    text = re.sub(r"−\s*(?=\d)", "минус ", text)
    return re.sub(r"\+(?=\s*\d)", "плюс ", text)


def _expand_temperature(text: str) -> str:
    # У дробей нужен «градуса», у целых — обычное согласование.
    pattern = re.compile(r"(?<![\w.,])(\d+(?:[,.]\d+)?)\s*°\s*([CС])\b", re.IGNORECASE)
    def repl(match: re.Match) -> str:
        number = match.group(1)
        noun = "градуса" if re.search(r"[,.]", number) else plural(int(number), ("градус", "градуса", "градусов"))
        return f"{number} {noun} цельсия"
    return pattern.sub(repl, text)


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
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    if not keep_yo:
        text = text.replace("ё", "е").replace("Ё", "Е")
    return text


def _apply_stress_overrides(text: str, overrides: dict[str, str]) -> str:
    """Явные ударения для известных слов; сохраняем буквы и их регистр."""
    checked = {}
    for word, marked in overrides.items():
        if (not re.fullmatch(r"[а-яё]+", word, re.IGNORECASE)
                or marked.count("+") != 1
                or not re.search(r"\+(?=[аеёиоуыэюя])", marked, re.IGNORECASE)
                or marked.replace("+", "").casefold() != word.casefold()):
            raise ValueError(f"некорректная подсказка ударения: {word!r} → {marked!r}")
        checked[word.casefold()] = marked
    if not checked:
        return text
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(word) for word in checked) + r")\b", re.IGNORECASE)

    def repl(match: re.Match) -> str:
        letters = iter(match.group(0))
        return "".join("+" if char == "+" else next(letters) for char in checked[match.group(0).casefold()])

    return pattern.sub(repl, text)


def normalize(
    text: str,
    expand_numbers: bool = True,
    expand_abbrev: bool = True,
    strip_stress: bool = True,
    keep_yo: bool = True,
    stress_overrides: dict[str, str] | None = None,
) -> str:
    """Приводит текст к виду, который движок сможет произнести.

    strip_stress убирает знак ударения `+`: piper и XTTS его не понимают
    и прочитают вслух как «плюс». Для F5-TTS_RUSSIAN, наоборот, ударения
    нужны — там этот флаг выключается в пресете.
    stress_overrides применяется только при strip_stress=False. Это
    проверенный словарь отдельных слов, не автоматическая расстановка.
    """
    text = _clean_unicode(text, keep_yo=keep_yo)

    if strip_stress:
        text = re.sub(r"\+(?=[аеёиоуыэюяАЕЁИОУЫЭЮЯ])", "", text)
    if expand_abbrev:
        text = _expand_abbrev(text)
    if expand_numbers:
        text = _join_digit_groups(text)
        text = _expand_signs(text)
        text = _expand_dates(text)
        text = _expand_times(text)
        text = _expand_temperature(text)
        text = _expand_decimals(text)
        text = _expand_units(text)
        text = _expand_bare_numbers(text)

    if not strip_stress and stress_overrides:
        text = _apply_stress_overrides(text, stress_overrides)

    # Только после чисел: сохраняем различие тире и унарного минуса.
    text = re.sub(r"[–—‒]", " - ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Пока без общего склонения числительных («с 1200 до 870»), дробей 1/2,
# числовых дат 15.03.2026, римских цифр и латиницы внутри русского текста.
