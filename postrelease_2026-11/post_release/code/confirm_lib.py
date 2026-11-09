"""Per-table quantities for the fresh-table confirmation of P1 (frozen plan PLAN.txt). Used unchanged for the pilot
(stored Study 15 newer-source fits, already seen) and for the fresh tables. Arm T only (complement scoring).

Per table and FM (means over reps within (strategy, m), then over m):
- gap_{M}: stratified - random, m in {1,3,5,10}, M in {mf (macro-F1), ba, acc}  (P1 Table main / Study 15 unit)
- bal_{M}: stratified - redraw (covered random), m in {3,5,10}  (balance part)
- cov_{M}: redraw - random, m in {1,3,5,10}  (coverage part)
- bal1_mf, cov1_mf: the same at m = 1 only
- flips: clear pairwise comparisons (both protocols separate the two FMs by > 0.01 macro-F1) whose winner changes,
  protocol = S (5 reps) vs R reps 0-4; noise = R reps 5-9 vs R reps 10-14 (matched 5-rep means), per (pair, m)
- q_stored: share of stored random draws missing a class (mean over m); q_exact: exact hypergeometric miss
  probability from the class counts of the pool the draws come from (all rows not in the labelled set's pool are
  unused: arm T draws from the whole table)
"""

from __future__ import annotations

import itertools
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

FMS = ("fm1", "fm2", "fm3", "fm5")
MULTS = (1, 3, 5, 10)
M3 = (3, 5, 10)
METRICS = {"mf": "macro_f1", "ba": "balanced_accuracy", "acc": "accuracy"}
CLEAR = 0.01
COLS = ["arm", "strategy", "mult", "rep", "coverage", "macro_f1", "balanced_accuracy", "accuracy", "seconds"]


def load_fits(folder: Path, key: str, fm: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    f = folder / f"{key}_{fm}.parquet"
    have = set(pq.read_schema(f).names)
    t = pd.read_parquet(f, columns=[c for c in COLS if c in have])
    return t[t.arm == "T"] if "arm" in t.columns else t


def cell_means(t: pd.DataFrame) -> pd.DataFrame:
    return t.groupby(["strategy", "mult"])[list(METRICS.values())].mean()


def exact_miss(counts: np.ndarray, B: int) -> float:
    """P(a uniform B-subset of the table misses at least one class), inclusion-exclusion."""
    n, C = int(counts.sum()), len(counts)
    if B > n:
        return 0.0
    tot = comb(n, B)
    p = 0.0
    for k in range(1, C):
        for S in itertools.combinations(range(C), k):
            rest = n - int(counts[list(S)].sum())
            p += (-1) ** (k + 1) * comb(rest, B) / tot if rest >= B else 0.0
    return float(p)


def table_quantities(folder: Path, key: str, counts: np.ndarray) -> dict:
    rec: dict = dict(dataset=key, n=int(counts.sum()), C=len(counts))
    fits = {fm: load_fits(folder, key, fm) for fm in FMS}
    cm = {fm: cell_means(fits[fm]) for fm in FMS}
    for fm in FMS:
        c = cm[fm]
        for short, col in METRICS.items():
            s = lambda strat, ms: np.array([c.loc[(strat, m), col] for m in ms])  # noqa: E731
            rec[f"gap_{short}_{fm}"] = float(np.mean(s("stratified", MULTS) - s("random", MULTS)))
            rec[f"bal_{short}_{fm}"] = float(np.mean(s("stratified", M3) - s("redraw", M3)))
            rec[f"cov_{short}_{fm}"] = float(np.mean(s("redraw", MULTS) - s("random", MULTS)))
        rec[f"bal1_mf_{fm}"] = float(c.loc[("stratified", 1), "macro_f1"] - c.loc[("redraw", 1), "macro_f1"])
        rec[f"cov1_mf_{fm}"] = float(c.loc[("redraw", 1), "macro_f1"] - c.loc[("random", 1), "macro_f1"])
        rec[f"seconds_{fm}"] = float(fits[fm].seconds.sum())
    # flips (protocol vs matched noise), per (pair, m)
    def rep_mean(fm, strat, reps, m):
        t = fits[fm]
        return float(t[(t.strategy == strat) & (t.mult == m) & t.rep.isin(reps)].macro_f1.mean())
    pf = pc = nf = nc = 0
    for a, b in itertools.combinations(FMS, 2):
        for m in MULTS:
            dS = rep_mean(a, "stratified", range(5), m) - rep_mean(b, "stratified", range(5), m)
            dR = rep_mean(a, "random", range(0, 5), m) - rep_mean(b, "random", range(0, 5), m)
            d1 = rep_mean(a, "random", range(5, 10), m) - rep_mean(b, "random", range(5, 10), m)
            d2 = rep_mean(a, "random", range(10, 15), m) - rep_mean(b, "random", range(10, 15), m)
            if abs(dS) > CLEAR and abs(dR) > CLEAR:
                pc += 1
                pf += int(np.sign(dS) != np.sign(dR))
            if abs(d1) > CLEAR and abs(d2) > CLEAR:
                nc += 1
                nf += int(np.sign(d1) != np.sign(d2))
    rec.update(flip_prot=pf, clear_prot=pc, flip_noise=nf, clear_noise=nc)
    r = fits["fm1"][fits["fm1"].strategy == "random"]
    qs, qe = [], []
    for m in MULTS:
        qs.append(float((r[r.mult == m].coverage < 1 - 1e-9).mean()))
        qe.append(exact_miss(counts, m * len(counts)))
    rec["q_stored"], rec["q_exact"] = float(np.mean(qs)), float(np.mean(qe))
    for short in METRICS:
        for part in ("gap", "bal", "cov"):
            rec[f"{part}_{short}_mean"] = float(np.mean([rec[f"{part}_{short}_{fm}"] for fm in FMS]))
    rec["cov1_mf_mean"] = float(np.mean([rec[f"cov1_mf_{fm}"] for fm in FMS]))
    return rec


# ----------------------------------------------------------------------------- tests used by the plan
def wilcoxon_p(x: np.ndarray) -> float:
    from scipy import stats

    x = np.asarray(x, float)
    x = x[np.abs(x) > 0]
    if len(x) < 2:
        return 1.0
    return float(stats.wilcoxon(x, alternative="two-sided").pvalue)


def tost_p(x: np.ndarray, margin: float) -> float:
    """Paired TOST (one-sample t on per-table values), max of the two one-sided p values."""
    from scipy import stats

    x = np.asarray(x, float)
    p1 = stats.ttest_1samp(x, -margin, alternative="greater").pvalue
    p2 = stats.ttest_1samp(x, margin, alternative="less").pvalue
    return float(max(p1, p2))


def spearman_one_sided(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy import stats

    r = stats.spearmanr(a, b)
    p = r.pvalue / 2 if r.statistic > 0 else 1 - r.pvalue / 2
    return float(r.statistic), float(p)


def holm(ps: dict[str, float]) -> dict[str, float]:
    keys = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    k = len(keys)
    for i, key in enumerate(keys):
        run = max(run, min(1.0, (k - i) * ps[key]))
        out[key] = run
    return out
