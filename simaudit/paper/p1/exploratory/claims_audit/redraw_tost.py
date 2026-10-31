"""EXPLORATORY (claims audit, 2026-10-28): stratified minus redraw, m = 3-10, T1' (Study 12 stored fits).

Per-table mean over m in {3, 5, 10} of (mean stratified - mean redraw) macro-F1; Wilcoxon
(reproduces the Study 12 S-1 row) and TOST with the frozen margin 0.01 (simaudit tost_p).
The TOST is not part of the Study 12 Holm family. Writes out/redraw_tost.csv.
Run from repo root:  nice -n 15 .venv/bin/python -I paper/p1/exploratory/claims_audit/redraw_tost.py
"""
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon
from simaudit.analysis.study7 import tost_p

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
d = pd.concat([pd.read_parquet(f, columns=["dataset", "learner", "strategy", "mult", "rep", "macro_f1"])
               for f in sorted((ROOT / "results" / "study12").glob("*_fm*.parquet"))])
m = d.groupby(["learner", "dataset", "strategy", "mult"]).macro_f1.mean().unstack("strategy")
m["sr"] = m.stratified - m.redraw
rows = []
for lr in ["fm1", "fm2", "fm3", "fm5"]:
    x = m.loc[lr]
    x = x[x.index.get_level_values("mult") >= 3]
    v = x.sr.groupby("dataset").mean().dropna().to_numpy()
    rows.append(dict(learner=lr, n=len(v), mean=v.mean(), p_wilcoxon=wilcoxon(v).pvalue, p_tost=tost_p(v)))
r = pd.DataFrame(rows)
(HERE / "out").mkdir(exist_ok=True)
r.to_csv(HERE / "out" / "redraw_tost.csv", index=False)
print(r.to_string())
