"""Between-table heterogeneity of the stratified-minus-random gap on T1' (Study 12).

Reads results/study12/*.parquet (read-only). For each FM, metric and budget (and the
mean over budgets) fits a random-effects model (REML; DerSimonian-Laird start) to the
per-table gaps and writes tau, I^2 and a 95% prediction interval for a new table
(t with k-2 df). Same estimator as exploratory/e9_meta_analysis.py, applied to T1'.
Output: analysis/out/t1p_meta.csv. The forest plot (DEV and T1', all FMs) is drawn by
figures/make_figures.py (fig_forest), which imports effects() and meta() from here.
Run from paper/p1:  ../../.venv/bin/python analysis/t1p_heterogeneity.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "results" / "study12"
OUT = HERE / "out"
FMS = ["fm1", "fm2", "fm3", "fm5"]
FM_NAMES = {"fm1": "TabPFN v2", "fm2": "TabICLv2", "fm3": "Kumo-Tabular", "fm5": "TabPFN v3.5"}
METRICS = ["macro_f1", "accuracy", "balanced_accuracy"]


def load() -> pd.DataFrame:
    cols = ["dataset", "learner", "strategy", "mult", "rep"] + METRICS
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(SRC.glob("*.parquet"))], ignore_index=True)
    return d[d.strategy.isin(["stratified", "random"])].rename(columns={"learner": "fm"})


def effects(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = d.groupby(["fm", "dataset", "mult", "strategy"])[metric].agg(["mean", "var", "count"]).unstack("strategy")
    out = pd.DataFrame({"y": g[("mean", "stratified")] - g[("mean", "random")],
                        "v": g[("var", "stratified")] / g[("count", "stratified")]
                        + g[("var", "random")] / g[("count", "random")]}).reset_index().dropna()
    avg = out.groupby(["fm", "dataset"]).agg(y=("y", "mean"), v=("v", lambda v: v.sum() / len(v) ** 2)).reset_index()
    avg["mult"] = "all"
    out["mult"] = out.mult.astype(str)
    return pd.concat([out, avg], ignore_index=True)


def dl(y, v):
    w = 1 / v
    mu_f = (w * y).sum() / w.sum()
    Q = (w * (y - mu_f) ** 2).sum()
    k = len(y)
    return max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())), Q


def reml(y, v, tau2, it=500):
    for _ in range(it):
        w = 1 / (v + tau2)
        mu = (w * y).sum() / w.sum()
        p_trace = w.sum() - (w ** 2).sum() / w.sum()
        score = 0.5 * ((w ** 2 * (y - mu) ** 2).sum() - p_trace)
        info = 0.5 * ((w ** 2).sum() - 2 * (w ** 3).sum() / w.sum() + ((w ** 2).sum() / w.sum()) ** 2)
        new = max(0.0, tau2 + score / info)
        if abs(new - tau2) < 1e-12:
            return new
        tau2 = new
    return tau2


def meta(eff: pd.DataFrame, metric: str) -> list[dict]:
    rows = []
    for (fm, m), g in eff.groupby(["fm", "mult"]):
        y, v = g.y.to_numpy(), np.maximum(g.v.to_numpy(), 1e-10)
        tau2_dl, Q = dl(y, v)
        k = len(y)
        tau2 = reml(y, v, tau2_dl)
        w = 1 / (v + tau2)
        mu = (w * y).sum() / w.sum()
        se = np.sqrt(1 / w.sum())
        half = stats.t.ppf(0.975, k - 2) * np.sqrt(tau2 + se ** 2)
        rows.append(dict(metric=metric, fm=fm, mult=m, k=k, mu=mu, se=se, ci_lo=mu - 1.96 * se,
                         ci_hi=mu + 1.96 * se, tau=np.sqrt(tau2), I2=max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0,
                         Q=Q, p_Q=float(stats.chi2.sf(Q, k - 1)), pi_lo=mu - half, pi_hi=mu + half,
                         share_pos=float(np.mean(y > 0))))
    return rows


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = load()
    rows, effs = [], {}
    for metric in METRICS:
        effs[metric] = effects(d, metric)
        rows += meta(effs[metric], metric)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "t1p_meta.csv", index=False)
    print(res.round(4).to_string())


if __name__ == "__main__":
    main()
