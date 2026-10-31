"""B8 (EXPLORATORY): E13 FM gap minus classical-learner gap (DEV) with seven FMs.

Original code: e13_fm_vs_classical.py and make_exploratory_numbers.e13, imported unchanged.
Inputs: the E1 classical fits (out/e1_runs.parquet, read-only copy), which are keyed by the
Study 7 Part (a) selections; the extras' fits use the same selections (rows and seeds checked
in harness/B-notes), so the stored E1 gaps apply without refits. FM gaps: e6_cells.csv of B1
(run "4": the existing out/e6_cells.csv; run "7": sevenfm/out/b1/seven/e6_cells.csv, seven FMs).
Holm over the FMs within classical x metric x m: over seven FMs in run "7" (exploratory).
Usage: nice -n 15 .venv/bin/python paper/p1/exploratory/sevenfm/b8_fm_vs_classical.py {4|7}
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

S, FMS, OUT = H.setup("b8")

import common as cm  # noqa: E402

cm.FMS, cm.OUT = FMS, OUT
cm.FM_NAMES.update(H.NAMES_X)
import e13_fm_vs_classical as e13  # noqa: E402
import make_exploratory_numbers as men  # noqa: E402

men.OUT = OUT
if S == "7":
    men.FT.update(H.FMW_X)


def main() -> None:
    shutil.copy2(H.EXPL / "out" / "e1_runs.parquet", OUT / "e1_runs.parquet")
    src = H.EXPL / "out" / "e6_cells.csv" if S == "4" else H.HERE / "out" / "b1" / "seven" / "e6_cells.csv"
    shutil.copy2(src, OUT / "e6_cells.csv")
    e13.main()
    men.MAC.clear(), men.SRC.clear()
    men.e13()
    lines = [f"\\newcommand{{\\{k}}}{{{v}}} % {men.SRC[k]}" for k, v in sorted(men.MAC.items())]
    if S == "4":
        bad = [] if H.verify_csv(OUT / "e13_diff.csv", H.EXPL / "out" / "e13_diff.csv") else ["e13_diff"]
        bad += H.verify_lines(lines, H.EXPL / "out" / "exploratory_numbers.tex", "E13")
        print("VERIFY B8:", "PASS" if not bad else f"FAIL {bad}")
        return
    note = "B8, seven FMs, DEV Study 7 Part (a) labelled sets; Holm over the seven FMs"
    keep = [ln for ln in lines if H.keep7(H.macro_name(ln))]
    H.write_tex(H.HERE / "out" / "sevenfm_b8.tex", [H.rename(ln, note) for ln in keep], "B8 E13, seven FMs")
    print(len(keep), "macros")


if __name__ == "__main__":
    main()
