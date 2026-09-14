"""Форматирование результата анализа в человекочитаемый отчёт."""
from datetime import datetime


# ANSI-цвета
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def _bar(value: float, width: int = 30, filled: str = "█", empty: str = "░") -> str:
    """Рисует прогресс-бар для значения [0, 1]."""
    filled_n = int(round(value * width))
    return filled * filled_n + empty * (width - filled_n)


def _verdict_style(verdict: str) -> tuple[str, str, str]:
    """Возвращает (цвет, символ, человекочитаемый текст)."""
    if verdict == "bonafide":
        return C.GREEN, "✓", "ЖИВОЙ ГОЛОС"
    if verdict == "spoof":
        return C.RED, "✗", "СИНТЕЗИРОВАННЫЙ ГОЛОС"
    return C.YELLOW, "?", "НЕОПРЕДЕЛЕНО"


def _confidence_label(score: float) -> str:
    """Словесная оценка уверенности модели."""
    # для bonafide: чем ниже spoof_score, тем увереннее
    if score < 0.1:
        return "очень высокая"
    if score < 0.3:
        return "высокая"
    if score < 0.5:
        return "средняя"
    if score < 0.7:
        return "низкая"
    return "очень высокая (синтез)"


def format_report(result: dict) -> str:
    """Собирает текстовый отчёт из dict-результата Analyzer.analyze()."""
    color, symbol, label = _verdict_style(result["verdict"])

    spoof_score = result["spoof_score"]
    bonafide_score = 1.0 - spoof_score

    # для прогресс-бара показываем уверенность в итоговом вердикте
    confidence = bonafide_score if result["verdict"] == "bonafide" else spoof_score
    bar = _bar(confidence)
    conf_label = _confidence_label(spoof_score)

    duration = result["duration"]
    mins, secs = divmod(duration, 60)

    lines = [
        "",
        f"{C.BOLD}{C.CYAN}╭─ Voice Analyzer ──────────────────────────────────────────╮{C.RESET}",
        f"{C.CYAN}│{C.RESET}  {C.DIM}Файл:{C.RESET}     {result['file']}",
        f"{C.CYAN}│{C.RESET}  {C.DIM}Длительность:{C.RESET} {int(mins):02d}:{secs:05.2f}",
        f"{C.CYAN}│{C.RESET}  {C.DIM}Сегментов речи:{C.RESET} {result['num_segments']}",
        f"{C.CYAN}│{C.RESET}  {C.DIM}Обработано за:{C.RESET} {result['latency_ms']/1000:.2f} с",
        f"{C.CYAN}╰───────────────────────────────────────────────────────────╯{C.RESET}",
        "",
        f"  {C.BOLD}Вердикт:{C.RESET}  {color}{C.BOLD}{symbol}  {label}{C.RESET}",
        f"  {C.DIM}Уверенность: {conf_label}{C.RESET}",
        "",
        f"  {C.DIM}Живой голос   {C.RESET}{C.GREEN}{_bar(bonafide_score)}{C.RESET}  "
        f"{bonafide_score*100:5.1f}%",
        f"  {C.DIM}Синтез        {C.RESET}{C.RED}{_bar(spoof_score)}{C.RESET}  "
        f"{spoof_score*100:5.1f}%",
        "",
        f"  {C.BOLD}Пол:{C.RESET}      {C.DIM}не определён (модуль в разработке){C.RESET}",
        "",
    ]

    return "\n".join(lines)


def format_json(result: dict) -> str:
    """JSON без embedding — для интеграции с другими модулями."""
    import json
    import numpy as np

    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return super().default(o)

    clean = {k: v for k, v in result.items() if k != "embedding"}
    return json.dumps(clean, indent=2, ensure_ascii=False, cls=_Enc)