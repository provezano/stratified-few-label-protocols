"""B7 (EXPLORATORY): random-effects heterogeneity (REML tau, I^2, prediction intervals) with seven FMs.

(i) DEV: e9_meta_analysis.py (Study 7 Part (a) draws, k = 20) re-run with the original code;
    the cross-FM variance components ("all") become a 20 x 7 design.
(ii) T1': analysis/t1p_heterogeneity.py (Study 12 draws, k = 52) re-run with the original code;
    the THet* macro block of make_numbers.py is copied verbatim (ranges over FM become ranges
    over the seven FMs). All seven FMs share the draws on each set.
Run "4" must reproduce out/e9_*.csv, analysis/out/t1p_meta.csv, the ExENine* lines of
exploratory_numbers.tex and the THet* lines of numbers_p1.tex.
Usage: nice -n 15 .venv/bin/python paper/p1/exploratory/sevenfm/b7_heterogeneity.py {4|7}
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

S, FMS, OUT = H.setup("b7")
sys.path.insert(0, str(H.EXPL.parent / "analysis"))

import common as cm  # noqa: E402

cm.FMS, cm.OUT = FMS, OUT
cm.FM_NAMES.update(H.NAMES_X)
import e9_meta_analysis as e9  # noqa: E402
import make_exploratory_numbers as men  # noqa: E402
import t1p_heterogeneity as th  # noqa: E402
from simaudit.settings import settings  # noqa: E402

men.OUT = OUT
if S == "7":
    men.FT.update(H.FMW_X)
th.SRC, th.OUT, th.FMS = settings.results_dir / "study12", OUT, list(FMS)
th.FM_NAMES.update(H.NAMES_X)
P1 = H.EXPL.parent


def thet_macros(path: Path) -> list[str]:
    """Verbatim copy of the THet block of paper/p1/make_numbers.py (FM word map extended)."""
    import make_numbers_shim as mn  # noqa: F401  (functions f, pct, macro_name, put collected below)
    return mn.lines(path)


def main() -> None:
    e9.main()
    th.main()
    men.MAC.clear(), men.SRC.clear()
    men.e9()
    e9_lines = [f"\\newcommand{{\\{k}}}{{{v}}} % {men.SRC[k]}" for k, v in sorted(men.MAC.items())]
    th_lines = thet_macros(OUT / "t1p_meta.csv")
    if S == "4":
        bad = []
        for f in ("e9_meta.csv", "e9_varcomp.csv"):
            bad += [f] if not H.verify_csv(OUT / f, H.EXPL / "out" / f) else []
        bad += [] if H.verify_csv(OUT / "t1p_meta.csv", P1 / "analysis" / "out" / "t1p_meta.csv") else ["t1p_meta"]
        bad += H.verify_lines(e9_lines, H.EXPL / "out" / "exploratory_numbers.tex", "E9")
        bad += H.verify_lines(th_lines, P1 / "numbers_p1.tex", "THet")
        print("VERIFY B7:", "PASS" if not bad else f"FAIL {bad}")
        return
    note = "B7, seven FMs; same draws (DEV Study 7 Part (a), T1' Study 12)"
    keep = [ln for ln in e9_lines + th_lines if H.keep7(H.macro_name(ln))]
    lines = []
    for ln in keep:
        ln = ln.replace("\\newcommand{\\THet", "\\newcommand{\\ExSevFmTHet") if "{\\THet" in ln else H.rename(ln, note)
        if "ExSevFmTHet" in ln:
            ln += f" [SEVEN-FM EXTENSION, exploratory: {note}]"
        lines.append(ln)
    H.write_tex(H.HERE / "out" / "sevenfm_b7.tex", lines, "B7 heterogeneity, seven FMs")
    print(len(lines), "macros")


if __name__ == "__main__":
    main()
