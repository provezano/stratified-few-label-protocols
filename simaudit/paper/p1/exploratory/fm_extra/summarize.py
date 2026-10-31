"""Summarise fm_extra Part (a) runs next to the stored fm1-fm5 rows (read-only).

Per (FM, m): mean over tables of the per-table means (reps first) of macro-F1 under
stratified and random, the gap stratified - random, its bootstrap 95% CI over
tables, Wilcoxon p, and the mean seconds per fit. Only tables present for every
listed FM are used (complete case), so the stored FMs are compared on the same
tables and labelled sets.

Usage: PYTHONPATH=src .venv_fm_extra/bin/python paper/p1/exploratory/fm_extra/summarize.py \
           [--dir out/s7a] [--name s7a]
Writes out/{name}_cells.csv and out/{name}_summary.csv; prints a Markdown table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from simaudit.settings import settings

HERE = Path(__file__).resolve().parent
S7A = settings.results_dir / "study7" / "a"
NAMES = dict(fm1="TabPFN v2", fm2="TabICLv2", fm3="Kumo-Tabular", fm5="TabPFN v3.5",
             fm6="TabDPT v1.3", fm7="Mitra-v2", fm8="LimiX-16M")
COLS = ["dataset", "learner", "strategy", "mult", "rep", "macro_f1", "accuracy", "seconds"]


def load(run_dir: Path) -> pd.DataFrame:
    extra = [pd.read_parquet(f, columns=COLS) for f in sorted(run_dir.glob("*_fm*.parquet"))
             if not f.name.endswith(".partial.parquet")]
    new = pd.concat(extra)
    keys = new[["dataset", "strategy", "mult", "rep"]].drop_duplicates()
    old = [pd.read_parquet(S7A / f"{d}_{fm}.parquet", columns=COLS)
           for d in new.dataset.unique() for fm in ("fm1", "fm2", "fm3", "fm5")]
    old = pd.concat(old).merge(keys, on=["dataset", "strategy", "mult", "rep"])
    return pd.concat([old, new])


def boot(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df[df.strategy.isin(["stratified", "random"])]
    cells = df.groupby(["learner", "dataset", "mult", "strategy"]).macro_f1.mean().unstack("strategy")
    cells["gap"] = cells.stratified - cells.random
    cells = cells.reset_index()
    complete = cells.groupby("dataset").learner.nunique()
    keep = complete[complete == cells.learner.nunique()].index
    cells = cells[cells.dataset.isin(keep)]
    secs = df.groupby("learner").seconds.mean()
    rows = []
    for (fm, m), g in list(cells.groupby(["learner", "mult"])) + [
        ((fm, "mean"), g.groupby("dataset")[["stratified", "random", "gap"]].mean().reset_index())
        for fm, g in cells.groupby("learner")
    ]:
        x = g.gap.to_numpy()
        lo, hi = boot(x)
        p = float(stats.wilcoxon(x).pvalue) if np.any(x != 0) and len(x) > 1 else 1.0
        rows.append(dict(fm=fm, model=NAMES[fm], mult=m, tables=len(x), mf1_strat=g.stratified.mean(),
                         mf1_random=g.random.mean(), gap=x.mean(), ci_lo=lo, ci_hi=hi, wilcoxon_p=p,
                         sec_per_fit=secs[fm]))
    return cells, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "out" / "s7a"))
    ap.add_argument("--name", default="s7a")
    a = ap.parse_args()
    cells, s = summary(load(Path(a.dir)))
    cells.to_csv(HERE / "out" / f"{a.name}_cells.csv", index=False)
    s.to_csv(HERE / "out" / f"{a.name}_summary.csv", index=False)
    print(s.round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
