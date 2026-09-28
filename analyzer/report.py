"""Console reporting formatter for Liveness Analyzer."""

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"

def print_header() -> None:
    print(f"\n{C.BOLD}{C.CYAN}╭─ Liveness Analyzer {'─'*72}╮{C.RESET}")
    print(f"{C.CYAN}│{C.RESET} {C.BOLD}{'ФАЙЛ':<25} │ {'ВЕРДИКТ':<9} │ {'УВЕРЕН.':<7} │ {'ДЕТАЛИ АНАЛИЗА':<43}{C.RESET} {C.CYAN}│{C.RESET}")
    print(f"{C.CYAN}├{'─'*27}┼{'─'*11}┼{'─'*9}┼{'─'*45}┤{C.RESET}")

def print_footer() -> None:
    print(f"{C.CYAN}╰{'─'*94}╯{C.RESET}\n")

def format_row(res: dict) -> str:
    file_name = res['file'].split('/')[-1]
    if len(file_name) > 25:
        file_name = file_name[:22] + "..."

    is_spoof = res['verdict'] == "spoof"
    color = C.RED if is_spoof else C.GREEN
    verdict_str = "СИНТЕЗ" if is_spoof else "ЖИВОЙ"
    
    score = res['spoof_score']
    conf = score if is_spoof else (1.0 - score)
    conf_pct = f"{conf * 100:>5.1f}%"
    
    reason = str(res.get('dsp_reason', ''))
    if len(reason) > 43:
        reason = reason[:40] + "..."

    return f"{C.CYAN}│{C.RESET} {file_name:<25} │ {color}{C.BOLD}{verdict_str:<9}{C.RESET} │ {conf_pct:<7} │ {C.DIM}{reason:<43}{C.RESET} {C.CYAN}│{C.RESET}"