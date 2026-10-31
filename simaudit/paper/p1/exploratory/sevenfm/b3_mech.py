"""B3 (EXPLORATORY): coverage formula vs observed (Sec. 6.5 / Fig. 4(a)) with seven FMs.

Original code: review_reanalysis/mech_recompute.py, imported unchanged; its FM list is re-bound
and its worker pool is capped at 2 processes (the original uses 4; the result does not depend
on it). Draws: DEV Study 7 Part (a), T1' Study 12, the same for all seven FMs. The leave-one-
table-out balance fit (kappa) is per FM, so adding FMs does not change the four existing FMs.
Run "4" must reproduce out/mech_tables.csv, mech_summary.csv and the ExRvMech* lines.
Usage: nice -n 15 .venv/bin/python paper/p1/exploratory/sevenfm/b3_mech.py {4|7}
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

S, FMS, OUT = H.setup("b3")

import rv_common  # noqa: E402

rv_common.OUT, rv_common.FMS = OUT, FMS
rv_common.FMW.update(H.FMW_X)
import mech_recompute as mr  # noqa: E402

RV_OUT = H.EXPL / "review_reanalysis" / "out"


def main() -> None:
    mr.OUT, mr.FMS = OUT, FMS
    mr.Pool = lambda n: mp.get_context("spawn").Pool(2)
    mr.main()
    lines = (OUT / "rv_mech.tex").read_text().splitlines()
    if S == "4":
        bad = []
        for f in ("mech_tables.csv", "mech_summary.csv"):
            bad += [] if H.verify_csv(OUT / f, RV_OUT / f) else [f]
        bad += H.verify_lines(lines, RV_OUT / "rv_mech.tex", "rv_mech.tex")
        bad += H.verify_lines(lines, RV_OUT / "rv_numbers.tex", "rv_numbers.tex")
        print("VERIFY B3:", "PASS" if not bad else f"FAIL {bad}")
        return
    note = "B3, seven FMs; same draws (DEV Study 7 Part (a), T1' Study 12)"
    keep = [ln.replace("over 4 FMs", "over 7 FMs") for ln in lines if H.keep7(H.macro_name(ln))]
    H.write_tex(H.HERE / "out" / "sevenfm_b3.tex", [H.rename(ln, note) for ln in keep], "B3 coverage mechanism, seven FMs")
    print(len(keep), "macros")


if __name__ == "__main__":
    main()
