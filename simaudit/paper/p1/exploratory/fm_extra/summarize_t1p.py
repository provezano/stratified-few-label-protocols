"""EXPLORATORY summary of the T1' fm6/fm7 runs (run_t1p.py), arm T, macro-F1 on E \\ L.

Per (FM, m) and for the mean over m in {1, 3, 5, 10} (prereg Study 12 contrast (i)
shape): per-table mean over reps of stratified and of random, gap = stratified -
random, mean over tables, 10k bootstrap 95% CI over tables, two-sided Wilcoxon
signed-rank p (unadjusted). Also within binary / multiclass tables, and the DEV
Part (a) gap of the same FM for magnitude comparison (out/s7a_summary.csv for fm6,
out/s7a_v2_summary.csv for the seeded fm7).

Two table sets: ``all`` (52) and ``excl257`` (51; without pmlb_1000000257 = UCI 257,
whose labels "Very Low" and "very_low" are one class split in two by the loader).

fm1-fm5 (results/study12) are NOT read by default: Study 12's confirmatory report
does not exist yet, and this exploratory script must not unblind it. Pass
``--with-stored`` only after reports/STUDY12_REPORT.md is written.
There is no Study 12 classical arm, so the FM-gap-minus-classical comparison is
not computed (classical learners on T1' exist only from Study 13 S-1 onwards).

Usage: PYTHONPATH=src .venv_fm_extra/bin/python paper/p1/exploratory/fm_extra/summarize_t1p.py
Writes out/t1p_cells.csv, out/t1p_summary.csv, out/T1P_SUMMARY.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from simaudit.settings import settings

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RUNS = OUT / "t1p"
S12 = settings.results_dir / "study12"
UCI257 = "pmlb_1000000257"
NAMES = dict(fm1="TabPFN v2", fm2="TabICLv2", fm3="Kumo-Tabular", fm5="TabPFN v3.5",
             fm6="TabDPT v1.3", fm7="Mitra-v2 (seeded)")
COLS = ["dataset", "learner", "strategy", "mult", "rep", "n_classes", "macro_f1", "seconds"]
DEV = {"fm6": ("s7a_summary.csv", "fm6"), "fm7": ("s7a_v2_summary.csv", "fm7")}


def load(with_stored: bool) -> pd.DataFrame:
    files = [f for f in sorted(RUNS.glob("*_fm[67].parquet"))]
    if with_stored:
        if not (settings.results_dir.parent / "reports" / "STUDY12_REPORT.md").exists():
            raise SystemExit("refusing --with-stored: Study 12 report not written yet")
        files += [f for f in sorted(S12.glob("*_fm[1235].parquet"))]
    df = pd.concat([pd.read_parquet(f, columns=COLS) for f in files])
    return df[df.strategy.isin(["stratified", "random"])]


def boot(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def wilcoxon(x: np.ndarray) -> float:
    return float(stats.wilcoxon(x).pvalue) if len(x) > 1 and np.any(x != 0) else 1.0


def cells(df: pd.DataFrame) -> pd.DataFrame:
    c = df.groupby(["learner", "dataset", "mult", "strategy"]).macro_f1.mean().unstack("strategy")
    c["gap"] = c.stratified - c.random
    c = c.reset_index()
    C = df.groupby("dataset").n_classes.first()
    c["C"] = c.dataset.map(C)
    return c  # each FM over its own finished tables ("tables" column)


def dev_gap(fm: str, m) -> float:  # type: ignore[no-untyped-def]
    name, key = DEV[fm]
    path = OUT / name
    if not path.exists():
        return float("nan")
    s = pd.read_csv(path)
    row = s[(s.fm == key) & (s.mult.astype(str) == str(m))]
    return float(row.gap.iloc[0]) if len(row) else float("nan")


def summary(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sets = {"all": c, "excl257": c[c.dataset != UCI257]}
    for set_name, cs in sets.items():
        for subset, sel in (("all", cs), ("binary", cs[cs.C == 2]), ("multiclass", cs[cs.C > 2])):
            for fm, g in sel.groupby("learner"):
                per_m = [(m, gm) for m, gm in g.groupby("mult")]
                mean = g.groupby("dataset")[["stratified", "random", "gap"]].mean().reset_index()
                for m, gm in per_m + [("mean", mean)]:
                    x = gm.gap.to_numpy()
                    lo, hi = boot(x)
                    rows.append(dict(set=set_name, subset=subset, fm=fm, model=NAMES[fm], mult=m,
                                     tables=len(x), mf1_strat=gm.stratified.mean(),
                                     mf1_random=gm.random.mean(), gap=x.mean(), ci_lo=lo, ci_hi=hi,
                                     wilcoxon_p=wilcoxon(x), win=float(np.mean(x > 0)),
                                     dev_gap=dev_gap(fm, m) if fm in DEV and subset == "all" else np.nan))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-stored", action="store_true")
    a = ap.parse_args()
    df = load(a.with_stored)
    c = cells(df)
    s = summary(c)
    c.to_csv(OUT / "t1p_cells.csv", index=False)
    s.to_csv(OUT / "t1p_summary.csv", index=False)
    secs = df.groupby("learner").seconds.agg(["mean", "median", "sum"])
    L = ["# EXPLORATORY: fm6 TabDPT / fm7 Mitra-v2 on T1' (Study 12 design, arm T)", "",
         "Not pre-registered. Same Study 12 selections, seeds, permutation and panel; "
         "stratified and random only (no redraw arm). Macro-F1 on E \\ L; gap = stratified - random; "
         "mean over tables [10k bootstrap 95% CI]; Wilcoxon two-sided, unadjusted. "
         + ("fm1-fm5 = stored Study 12 outputs (results/study12, read-only; report frozen in "
            "reports/STUDY12_REPORT.md), arm T stratified/random rows only. " if a.with_stored else
            "fm1-fm5 not read (Study 12 report pending). ")
         + "No Study 12 classical arm exists, so no FM-gap-minus-classical comparison.", "",
         f"Tables per FM: {c.groupby('learner').dataset.nunique().to_dict()}", "",
         "Seconds per fit (mean / median / worker-hours): " + "; ".join(
             f"{k} {r['mean']:.1f} / {r['median']:.1f} / {r['sum'] / 3600:.1f} h" for k, r in secs.iterrows()), ""]
    for set_name in ("all", "excl257"):
        g = s[(s.set == set_name) & (s.subset == "all")]
        L += [f"## {set_name} ({'52' if set_name == 'all' else '51, without UCI 257'} tables)", "",
              g.drop(columns=["set", "subset"]).round(4).to_markdown(index=False), ""]
    g = s[(s.set == "all") & (s.subset != "all") & (s.mult == "mean")]
    L += ["## binary vs multiclass, mean over m (all tables)", "",
          "T1' composition: 35 binary / 17 multiclass tables, per Study 12 Erratum 1 "
          "(prereg/STUDY12_PROTOCOL_T1PRIME.md; n_classes stored per table).", "",
          g.drop(columns=["set", "dev_gap"]).round(4).to_markdown(index=False), ""]
    (OUT / "T1P_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
