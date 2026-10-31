"""B6 (EXPLORATORY): E7 common-row re-scoring (DEV, Study 7 Part (a)) with seven FMs.

Original code: e7_common_eval.py (run, summarise) and make_exploratory_numbers.e7, imported
unchanged; FM list re-bound, worker pool capped at 2. All seven FMs use the same Study 7 Part (a)
selections, so the common row sets are identical across FMs. Holm in E7 is over the FMs within
metric x m x variant: in run "7" it is over the seven FMs (exploratory, its own Holm; the four-FM
Holm values stay those of the existing macros).
Usage: nice -n 15 .venv/bin/python paper/p1/exploratory/sevenfm/b6_common_rows.py {4|7}
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

S, FMS, OUT = H.setup("b6")

import pandas as pd  # noqa: E402

import common as cm  # noqa: E402

cm.FMS, cm.OUT = FMS, OUT
cm.FM_NAMES.update(H.NAMES_X)
import e7_common_eval as e7  # noqa: E402
import make_exploratory_numbers as men  # noqa: E402

men.OUT = OUT
if S == "7":
    men.FT.update(H.FMW_X)
e7.ProcessPoolExecutor = lambda n: ProcessPoolExecutor(2)


def main() -> None:
    e7.main()
    men.MAC.clear(), men.SRC.clear()
    men.e7()
    lines = [f"\\newcommand{{\\{k}}}{{{v}}} % {men.SRC[k]}" for k, v in sorted(men.MAC.items())]
    if S == "4":
        bad = []
        for f in ("e7_cells.csv", "e7_gaps.csv"):
            bad += [] if H.verify_csv(OUT / f, H.EXPL / "out" / f) else [f]
        bad += H.verify_lines(lines, H.EXPL / "out" / "exploratory_numbers.tex", "E7")
        print("VERIFY B6:", "PASS" if not bad else f"FAIL {bad}")
        return
    note = "B6, seven FMs, DEV Study 7 Part (a) draws; Holm over the seven FMs within metric x m x variant"
    keep = [ln for ln in lines if H.keep7(H.macro_name(ln))]
    H.write_tex(H.HERE / "out" / "sevenfm_b6.tex", [H.rename(ln, note) for ln in keep], "B6 E7 common rows, seven FMs")
    print(len(keep), "macros")


if __name__ == "__main__":
    main()
