"""B5 (EXPLORATORY): E2 all-class AUC gap (DEV, Study 7 Part (a)) with seven FMs.

Original code: e2_auc.py (run, main) imported unchanged (FM list re-bound, pool capped at 2).
The macro block is make_exploratory_numbers.e2 copied verbatim with its FM tag map extended
(the original hard-codes the four tags). Holm over the FMs within metric x budget: over seven
FMs in run "7" (exploratory). Same draws for all seven FMs.
Usage: nice -n 15 .venv/bin/python paper/p1/exploratory/sevenfm/b5_auc.py {4|7}
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

S, FMS, OUT = H.setup("b5")

import pandas as pd  # noqa: E402

import common as cm  # noqa: E402

cm.FMS, cm.OUT = FMS, OUT
import e2_auc as e2  # noqa: E402
import make_exploratory_numbers as men  # noqa: E402

e2.ProcessPoolExecutor = lambda n: ProcessPoolExecutor(2)
TAGS = dict(fm1="FMone", fm2="FMtwo", fm3="FMthree", fm5="FMfive")
if S == "7":
    TAGS.update(H.FMW_X)


def e2_macros() -> list[str]:
    """make_exploratory_numbers.e2, verbatim except the tag map."""
    men.MAC.clear(), men.SRC.clear()
    put, f, gap_macros = men.put, men.f, men.gap_macros
    src = "e2_auc_gaps.csv"
    g = pd.read_csv(OUT / src)
    for fm, t in TAGS.items():
        gap_macros(g, f"ExETwo{t}AllMOne", fm, "auc_all", 1, src)
        gap_macros(g, f"ExETwo{t}All", fm, "auc_all", "all", src)
        gap_macros(g, f"ExETwo{t}Seen", fm, "auc_seen", "all", src)
    for m in (1, 3, 5, 10, "all"):
        mt = "All" if m == "all" else f"M{m}"
        for metric, tag in (("auc_all", "AUCall"), ("auc_seen", "AUCseen")):
            s = g[(g.metric == metric) & (g.mult.astype(str) == str(m))]
            put(f"ExETwo{tag}{mt}Min", f(s["mean"].min(), sign=True), src)
            put(f"ExETwo{tag}{mt}Max", f(s["mean"].max(), sign=True), src)
            put(f"ExETwo{tag}{mt}NSig", str(int((s.p_holm < 0.05).sum())), src)
    runs = pd.read_parquet(OUT / "e2_auc_runs.parquet")
    put("ExETwoSeenNaN", str(int(runs.auc_seen.isna().sum())), "e2_auc_runs.parquet")
    put("ExETwoDraws", str(len(runs)), "e2_auc_runs.parquet")
    return [f"\\newcommand{{\\{k}}}{{{v}}} % {men.SRC[k]}" for k, v in sorted(men.MAC.items())]


def main() -> None:
    e2.main()
    lines = e2_macros()
    if S == "4":
        bad = [] if H.verify_csv(OUT / "e2_auc_gaps.csv", H.EXPL / "out" / "e2_auc_gaps.csv") else ["e2_auc_gaps"]
        a = pd.read_parquet(OUT / "e2_auc_runs.parquet")
        b = pd.read_parquet(H.EXPL / "out" / "e2_auc_runs.parquet")
        ok = a.equals(b)
        print(f"[verify parquet] e2_auc_runs.parquet: {'IDENTICAL' if ok else 'DIFFERS'}")
        bad += [] if ok else ["e2_auc_runs"]
        bad += H.verify_lines(lines, H.EXPL / "out" / "exploratory_numbers.tex", "E2")
        print("VERIFY B5:", "PASS" if not bad else f"FAIL {bad}")
        return
    note = "B5, seven FMs, DEV Study 7 Part (a) draws; Holm over the seven FMs within metric x budget"
    keep = [ln for ln in lines if H.keep7(H.macro_name(ln))]
    H.write_tex(H.HERE / "out" / "sevenfm_b5.tex", [H.rename(ln, note) for ln in keep], "B5 E2 all-class AUC, seven FMs")
    print(len(keep), "macros")


if __name__ == "__main__":
    main()
