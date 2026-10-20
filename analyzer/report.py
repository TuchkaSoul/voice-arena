"""Console reporting formatter for Voice Liveness Analyzer."""
from pathlib import Path


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"


# Ширина колонок таблицы
W_FILE = 34
W_VERD = 9
W_CONF = 8
W_ML = 6
W_FLAT = 6
W_ROLL = 9
W_UBR = 6
W_TIME = 8
W_RULE = 26

TOTAL_INNER = (
    W_FILE + W_VERD + W_CONF + W_ML + W_FLAT + W_ROLL + W_UBR + W_TIME + W_RULE + 26
)


def _hline(left: str, mid: str, right: str) -> str:
    cols = [
        "─" * (W_FILE + 2),
        "─" * (W_VERD + 2),
        "─" * (W_CONF + 2),
        "─" * (W_ML + 2),
        "─" * (W_FLAT + 2),
        "─" * (W_ROLL + 2),
        "─" * (W_UBR + 2),
        "─" * (W_TIME + 2),
        "─" * (W_RULE + 2),
    ]
    return f"{C.CYAN}{left}{mid.join(cols)}{right}{C.RESET}"


def print_header() -> None:
    title = " Voice Liveness Analyzer "
    top_bar = "─" * (TOTAL_INNER - len(title) - 1)
    print(f"\n{C.BOLD}{C.CYAN}╭─{title}{top_bar}╮{C.RESET}")
    print(
        f"{C.CYAN}│{C.RESET} "
        f"{C.BOLD}{'ФАЙЛ':<{W_FILE}}{C.RESET} {C.CYAN}│{C.RESET} "
        f"{C.BOLD}{'ВЕРДИКТ':<{W_VERD}}{C.RESET} {C.CYAN}│{C.RESET} "
        f"{C.BOLD}{'УВЕРЕН.':>{W_CONF}}{C.RESET} {C.CYAN}│{C.RESET} "
        f"{C.BOLD}{'ML (W)':>{W_ML}}{C.RESET} {C.CYAN}│{C.RESET} "
        f"{C.BOLD}{'FLAT':>{W_FLAT}}{C.RESET} {C.CYAN}│{C.RESET} "
        f"{C.BOLD}{'ROLLOFF':>{W_ROLL}}{C.RESET} {C.CYAN}│{C.RESET} "
        f"{C.BOLD}{'UBR':>{W_UBR}}{C.RESET} {C.CYAN}│{C.RESET} "
        f"{C.BOLD}{'ВРЕМЯ':>{W_TIME}}{C.RESET} {C.CYAN}│{C.RESET} "
        f"{C.BOLD}{'ПРАВИЛО КАСКАДА':<{W_RULE}}{C.RESET} "
        f"{C.CYAN}│{C.RESET}"
    )
    print(_hline("├", "┼", "┤"))


def format_row(res: dict) -> str:
    file_name = Path(res["file"]).name
    if len(file_name) > W_FILE:
        file_name = file_name[: W_FILE - 3] + "..."

    is_spoof = res["verdict"] == "spoof"
    color = C.RED if is_spoof else C.GREEN
    verdict_str = "СИНТЕЗ" if is_spoof else "ЖИВОЙ"

    score = float(res.get("spoof_score", 0.0))
    conf = score if is_spoof else (1.0 - score)
    conf_str = f"{conf * 100:>6.1f}%"

    ml_str = f"{float(res.get('model_score', 0.0)):>6.2f}"
    flat_str = f"{float(res.get('flatness', 0.0)):>6.2f}"
    roll_str = f"{float(res.get('rolloff_hz', 0.0)):>6.0f} Hz"
    ubr_str = f"{float(res.get('upper_band_ratio', 0.0)):>6.2f}"
    time_str = f"{float(res.get('latency_ms', 0.0)):>5.0f} ms"

    rule = str(res.get("dsp_reason", ""))
    if len(rule) > W_RULE:
        rule = rule[: W_RULE - 3] + "..."

    return (
        f"{C.CYAN}│{C.RESET} "
        f"{file_name:<{W_FILE}} {C.CYAN}│{C.RESET} "
        f"{color}{C.BOLD}{verdict_str:<{W_VERD}}{C.RESET} {C.CYAN}│{C.RESET} "
        f"{conf_str:>{W_CONF}} {C.CYAN}│{C.RESET} "
        f"{ml_str:>{W_ML}} {C.CYAN}│{C.RESET} "
        f"{flat_str:>{W_FLAT}} {C.CYAN}│{C.RESET} "
        f"{roll_str:>{W_ROLL}} {C.CYAN}│{C.RESET} "
        f"{ubr_str:>{W_UBR}} {C.CYAN}│{C.RESET} "
        f"{C.DIM}{time_str:>{W_TIME}}{C.RESET} {C.CYAN}│{C.RESET} "
        f"{C.DIM}{rule:<{W_RULE}}{C.RESET} "
        f"{C.CYAN}│{C.RESET}"
    )


def print_footer(records: list[dict] | None = None) -> None:
    if not records:
        print(_hline("╰", "┴", "╯") + "\n")
        return

    valid = [r for r in records if "verdict" in r]
    if not valid:
        print(_hline("╰", "┴", "╯") + "\n")
        return

    print(_hline("├", "┴", "┤"))
    total = len(valid)
    live_cnt = sum(1 for r in valid if r["verdict"] == "bonafide")
    spoof_cnt = total - live_cnt
    avg_ms = sum(float(r.get("latency_ms", 0.0)) for r in valid) / total

    summary_plain = (
        f" ИТОГО: {total} файл(ов)  │  ЖИВОЙ: {live_cnt}  │  "
        f"СИНТЕЗ: {spoof_cnt}  │  Среднее время: {avg_ms:.0f} ms"
    )
    pad = max(0, TOTAL_INNER - len(summary_plain))
    summary_colored = (
        f" {C.BOLD}ИТОГО:{C.RESET} {total} файл(ов)  {C.CYAN}│{C.RESET}  "
        f"{C.GREEN}{C.BOLD}ЖИВОЙ:{C.RESET} {live_cnt}  {C.CYAN}│{C.RESET}  "
        f"{C.RED}{C.BOLD}СИНТЕЗ:{C.RESET} {spoof_cnt}  {C.CYAN}│{C.RESET}  "
        f"{C.DIM}Среднее время: {avg_ms:.0f} ms{C.RESET}" + " " * pad
    )
    print(f"{C.CYAN}│{C.RESET}{summary_colored}{C.CYAN}│{C.RESET}")
    print(f"{C.CYAN}╰{'─' * TOTAL_INNER}╯{C.RESET}\n")