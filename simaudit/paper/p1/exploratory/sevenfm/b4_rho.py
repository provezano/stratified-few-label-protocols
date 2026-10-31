"""B4 (EXPLORATORY, PARTIAL): rho(1) with seven FMs on T1'.

Original code: claims_audit/rho_all.py = e11_study9_sensitivity.per_table / rho_summary over
the Study 9 nested learning curves (m in {1, 2, 3, 5, 7, 10, 15, 20, 30}). The extras have no
Study 9 fits; they have Study 12 fits at m in {1, 3, 5, 10} only (independent, not nested,
draws per m). So the full extension is impossible. What is done:
- run "4": rho_all.py re-run (Study 9, four FMs), must reproduce claims_audit/out/rho_all*.csv;
- run "7": the same per_table / rho_summary code on the Study 12 curves (m in {1, 3, 5, 10})
  for ALL seven FMs (like for like: the four pre-registered FMs are re-estimated on the same
  coarse Study 12 curve, so the extras are compared with them on equal terms, not with the
  Study 9 values). Limits: coarse 4-point grid (rho(1) intervals are wide and right-censored
  when the random curve never reaches the stratified value by m = 10), non-nested random
  draws, 20 random reps per m.
Usage: nice -n 15 .venv/bin/python paper/p1/exploratory/sevenfm/b4_rho.py {4|7}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

S, FMS, OUT = H.setup("b4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import e11_study9_sensitivity as e  # noqa: E402
from simaudit.settings import settings  # noqa: E402

FMW = dict(fm1="FMone", fm2="FMtwo", fm3="FMthree", fm5="FMfive", **H.FMW_X)


def f(x, nd=2):
    return f"{x:.{nd}f}" if np.isfinite(x) else "$\\infty$"


def pct(x):
    return f"{100 * x:.0f}\\%"


def main() -> None:
    if S == "4":
        raw = e.a9.load_results(e.cm.settings.results_dir / "study9" / "t1prime")
        curve = e.a9.curves(raw)
        t = pd.concat([e.per_table(curve, lr)[0] for lr in FMS])
        t.to_csv(OUT / "rho_all_tables.csv", index=False)
        r = pd.DataFrame(e.rho_summary(t))
        r.to_csv(OUT / "rho_all.csv", index=False)
        ca = H.EXPL / "claims_audit" / "out"
        bad = [x for x in ("rho_all.csv", "rho_all_tables.csv") if not H.verify_csv(OUT / x, ca / x)]
        print("VERIFY B4:", "PASS" if not bad else f"FAIL {bad}")
        return
    fr = []
    for fm in FMS:
        for p in sorted((settings.results_dir / "study12").glob(f"*_{fm}.parquet")):
            d = pd.read_parquet(p, columns=["dataset", "learner", "strategy", "mult", "rep", "macro_f1"])
            fr.append(d[d.strategy.isin(["stratified", "random"])])
    raw = pd.concat(fr, ignore_index=True).rename(columns={"mult": "m"})
    curve = e.a9.curves(raw)
    t = pd.concat([e.per_table(curve, lr)[0] for lr in FMS])
    t.to_csv(OUT / "rho_s12_tables.csv", index=False)
    r = pd.DataFrame(e.rho_summary(t))
    r.to_csv(OUT / "rho_s12.csv", index=False)
    print(r[["learner", "model", "median", "lo", "hi", "n", "n_inf", "share_gt1"]].to_string())
    L, src = [], "sevenfm/out/b4/seven/rho_s12.csv (T1' Study 12 curves, m in 1,3,5,10; PARTIAL: coarse grid"
    for lr in FMS:
        g = r[r.learner == lr]
        fz = g[g.model == "frozen"].iloc[0]
        pw = g[g.model == "power"].iloc[0]
        w = FMW[lr]
        L += [f"\\newcommand{{\\ExSevFmRhoSTwelve{w}}}{{{f(fz['median'])}}} % {src}, {lr}, frozen rule, median)",
              f"\\newcommand{{\\ExSevFmRhoSTwelve{w}Lo}}{{{f(fz.lo)}}} % {src}, {lr}, frozen, bootstrap 95%)",
              f"\\newcommand{{\\ExSevFmRhoSTwelve{w}Hi}}{{{f(fz.hi)}}} % {src}, {lr}, frozen, bootstrap 95%)",
              f"\\newcommand{{\\ExSevFmRhoSTwelve{w}Pow}}{{{f(pw['median'])}}} % {src}, {lr}, power-law fit, median)",
              f"\\newcommand{{\\ExSevFmRhoSTwelve{w}Share}}{{{pct(fz.share_gt1)}}} % {src}, {lr}, frozen, share of tables with rho>1)",
              f"\\newcommand{{\\ExSevFmRhoSTwelve{w}NInf}}{{{int(fz.n_inf)}}} % {src}, {lr}, right-censored tables)"]
    fz = r[r.model == "frozen"]
    L += [f"\\newcommand{{\\ExSevFmRhoSTwelveSevenMin}}{{{f(fz['median'].min())}}} % {src}, min over seven FMs)",
          f"\\newcommand{{\\ExSevFmRhoSTwelveSevenMax}}{{{f(fz['median'].max())}}} % {src}, max over seven FMs)",
          f"\\newcommand{{\\ExSevFmRhoSTwelveSevenNGtOne}}{{{int((fz.lo > 1).sum())}}} % {src}, FMs whose bootstrap lower bound > 1)"]
    H.write_tex(H.HERE / "out" / "sevenfm_b4.tex", [ln + " [SEVEN-FM EXTENSION, exploratory, PARTIAL: B4]" for ln in L],
                "B4 rho(1), seven FMs on Study 12 curves (partial)")
    print(len(L), "macros")


if __name__ == "__main__":
    main()
