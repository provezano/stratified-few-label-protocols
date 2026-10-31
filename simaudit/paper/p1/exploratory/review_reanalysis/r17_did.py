"""Round-17 check (EXPLORATORY; stored metrics only, no fits): two contrasts that the reviews asked for.

(1) Difference in differences for the ranking claim, per table: mean over m in {3, 5, 10} of
    [(TabICLv2 - TabPFN v3.5) under random] - [(TabICLv2 - TabPFN v3.5) under stratified],
    on DEV (Study 7, Part (a)) and T1' (Study 12). Not pre-registered.
(2) The primary contrast (stratified - random, macro-F1) without m = 1: mean over m in {3, 5, 10}, per FM.

Unit = table (mean over repetitions first). Two-sided Wilcoxon signed-rank; percentile bootstrap 95% CI over
tables (10,000 resamples, seed 20261030). Reads only the metric columns of results/study7/a and results/study12.
Sanity check: the mean per-table (ii-a) and (ii-c) differences must equal the reported pre-registered means.
Writes out/rv_r17.tex and out/r17_did.csv.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r17_did.py
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from rv_common import Macros, OUT, fmt, pfmt

REPO = OUT.parents[4]
SETS = {"Dev": "results/study7/a", "Tp": "results/study12"}
FMS = ("fm1", "fm2", "fm3", "fm5")
COLS = ["dataset", "learner", "strategy", "mult", "rep", "macro_f1"]
rng = np.random.default_rng(20261030)


def load(folder: str) -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(REPO / folder / "*.parquet"))):
        t = pq.read_table(f, columns=COLS).to_pandas()
        parts.append(t[t.strategy.isin(["stratified", "random"]) & t.learner.isin(FMS)])
    d = pd.concat(parts)
    d["dataset"] = d.dataset.astype(str)
    return d.groupby(["dataset", "learner", "strategy", "mult"]).macro_f1.mean().unstack("strategy")


def summary(x: np.ndarray, gen: np.random.Generator | None = None) -> tuple[float, float, float, float]:
    """Mean, percentile bootstrap 95% CI over tables (10,000 resamples) and two-sided Wilcoxon p."""
    boot = (rng if gen is None else gen).choice(x, size=(10_000, len(x)), replace=True).mean(1)
    return x.mean(), *np.percentile(boot, [2.5, 97.5]), stats.wilcoxon(x).pvalue


def gap_m3to10(g: pd.DataFrame, fm: str) -> np.ndarray:
    """Per-table S - R in macro-F1, mean over m in {3, 5, 10} (g as returned by load())."""
    mid = g[g.index.get_level_values("mult").isin([3, 5, 10])]
    t = mid.groupby(["dataset", "learner"]).mean().xs(fm, level="learner")
    return (t.stratified - t.random).dropna().to_numpy()


def main() -> None:
    M = Macros("r17")
    rows = []
    for tag, folder in SETS.items():
        g = load(folder)
        src = f"{folder} (metric columns), r17_did.py"
        mid = g[g.index.get_level_values("mult").isin([3, 5, 10])]
        t = mid.groupby(["dataset", "learner"]).mean()
        d2, d5 = t.xs("fm2", level="learner"), t.xs("fm5", level="learner")
        iia = (d2.random - d5.random).dropna()
        iic = (d2.stratified - d5.stratified).dropna()
        did = (iia - iic).dropna()
        for name, x in (("IIa", iia), ("IIc", iic), ("Did", did)):
            m, lo, hi, p = summary(x.to_numpy())
            rows.append(dict(set=tag, what=name, n=len(x), mean=m, lo=lo, hi=hi, p=p))
            if name == "Did":
                M.put(f"Did{tag}", fmt(m, sign=True), src + " (DiD, mean over tables)")
                M.put(f"Did{tag}Lo", fmt(lo, sign=True), src + " (DiD, bootstrap 2.5%)")
                M.put(f"Did{tag}Hi", fmt(hi, sign=True), src + " (DiD, bootstrap 97.5%)")
                M.put(f"Did{tag}P", pfmt(p), src + " (DiD, two-sided Wilcoxon, unadjusted)")
                M.put(f"Did{tag}N", str(len(x)), src + " (tables)")
        gaps = []
        for fm in FMS:
            x = gap_m3to10(g, fm)
            m, lo, hi, p = summary(x)
            rows.append(dict(set=tag, what=f"gap_m3to10_{fm}", n=len(x), mean=m, lo=lo, hi=hi, p=p))
            gaps.append((m, p))
        M.put(f"GapNoOne{tag}Min", fmt(min(a for a, _ in gaps), sign=True), src + " (S-R, mean over m=3,5,10, min over 4 FMs)")
        M.put(f"GapNoOne{tag}Max", fmt(max(a for a, _ in gaps), sign=True), src + " (S-R, mean over m=3,5,10, max over 4 FMs)")
        M.put(f"GapNoOne{tag}PMax", pfmt(max(p for _, p in gaps)), src + " (largest unadjusted Wilcoxon p over 4 FMs)")
        M.put(f"GapNoOne{tag}FMtwo", fmt(gaps[1][0], sign=True), src + " (TabICLv2)")
        M.put(f"GapNoOne{tag}FMtwoP", pfmt(gaps[1][1]), src + " (TabICLv2, unadjusted Wilcoxon p)")
        M.put(f"GapNoOne{tag}FMthree", fmt(gaps[2][0], sign=True), src + " (Kumo-Tabular)")
        M.put(f"GapNoOne{tag}FMone", fmt(gaps[0][0], sign=True), src + " (TabPFN v2)")
        M.put(f"GapNoOne{tag}FMfive", fmt(gaps[3][0], sign=True), src + " (TabPFN v3.5)")
    pd.DataFrame(rows).to_csv(OUT / "r17_did.csv", index=False)
    M.write()
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
