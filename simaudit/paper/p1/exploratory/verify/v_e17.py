"""Verification of E17 claims (d)-(h) from verify/out/fits_{t1,dev}.parquet.

Orientation: gap > 0 means stratified looks better (losses: random - stratified).
Run: OMP_NUM_THREADS=1 nice -n 15 uv run python paper/p1/exploratory/verify/v_e17.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent / "out"
lines: list[str] = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum()) if len(d) else np.nan


HIGHER = ["st_macro_f1", "st_balanced_accuracy", "st_accuracy", "kappa", "mcc",
          "auc_all", "st_auc", "log_lrp", "neg_log_lrn"]
LOWER = ["brier", "citl", "st_log_loss"]


def cell_gaps(df):
    """Per (dataset, learner, m): mean(S) - mean(R) (losses reversed). NaN draws skipped."""
    g = df.groupby(["dataset", "learner", "m", "strategy"])[HIGHER + LOWER].mean()
    s = g.xs("stratified", level="strategy")
    r = g.xs("random", level="strategy")
    k = s.index.intersection(r.index)
    out = s.loc[k, HIGHER] - r.loc[k, HIGHER]
    for c in LOWER:
        out[c] = r.loc[k, c] - s.loc[k, c]
    return out.reset_index()


res = {}
for name, fn in (("T1", "fits_t1.parquet"), ("DEV", "fits_dev.parquet")):
    df = pd.read_parquet(OUT / fn)
    df = df[df.strategy.isin(["random", "stratified"])]
    cg = cell_gaps(df)
    tv = df.groupby("dataset").tv_pool.first() if "tv_pool" in df else df.groupby("dataset").tv_full.first()
    cg["tv"] = cg.dataset.map(tv)
    cg["C"] = cg.dataset.map(df.groupby("dataset").C.first())
    res[name] = (df, cg)
    tab = cg.groupby(["dataset", "m"]).mean(numeric_only=True).reset_index()  # FM-averaged
    say(f"\n## {name}: mean gap over tables (FM-averaged), Wilcoxon over tables")
    rows = []
    for met in HIGHER + LOWER:
        for m, g in tab.groupby("m"):
            v = g[met].dropna().to_numpy()
            if len(v) < 5:
                continue
            p = stats.wilcoxon(v).pvalue if np.any(v != 0) else 1.0
            rows.append(dict(metric=met, m=m, n=len(v), mean=v.mean(), median=np.median(v), p=p,
                             rb=rank_biserial(v)))
    t = pd.DataFrame(rows)
    t["p_holm_metric"] = t.groupby("metric").p.transform(holm)
    t.to_csv(OUT / f"e17_gaps_{name}.csv", index=False)
    say(t.pivot_table(index="metric", columns="m", values="mean").round(3).to_string())
    say("p (raw):")
    say(t.pivot_table(index="metric", columns="m", values="p").map(lambda x: f"{x:.1e}").to_string())
    say("n tables:", t[t.metric == "st_balanced_accuracy"].set_index("m").n.to_dict())

# ---------------------------------------------------------------- (f) imbalanced half
df, cg = res["T1"]
tv = cg.groupby("dataset").tv.first()
med = tv.median()
hi = tv[tv > med].index
say(f"\n## (f) T1' imbalanced half: TV(pool) > {med:.3f}: {len(hi)} tables (ties at median: {(tv == med).sum()})")
for unit in ("tables(FM-mean)", "cells(table x FM)"):
    for met in ("citl", "st_accuracy", "brier", "st_balanced_accuracy"):
        out = []
        for m in (3, 10, 20):
            sub = cg[(cg.m == m) & cg.dataset.isin(hi)]
            if unit.startswith("tables"):
                sub = sub.groupby("dataset")[met].mean()
            v = np.asarray(sub[met] if not unit.startswith("tables") else sub, float)
            v = v[~np.isnan(v)]
            out.append(f"{m}C: median {np.median(v):+.3f} mean {v.mean():+.3f} rb {rank_biserial(v):+.2f} "
                       f"p {stats.wilcoxon(v).pvalue:.1e} n={len(v)}")
        say(f"{unit} {met}: " + " | ".join(out))

# ---------------------------------------------------------------- (g) binary LRs
say("\n## (g) T1' binary likelihood ratios (gap S - R; log LR+ and -log LR-; +0.5 correction)")
tab = cg[cg.C == 2].groupby(["dataset", "m"]).mean(numeric_only=True).reset_index()
pl, pn = [], []
for m, g in tab.groupby("m"):
    a, b = g.log_lrp.dropna(), g.neg_log_lrn.dropna()
    pl.append((m, a.mean(), stats.wilcoxon(a).pvalue, len(a)))
    pn.append((m, b.mean(), stats.wilcoxon(b).pvalue, len(b)))
hl = holm([x[2] for x in pl] + [x[2] for x in pn])
for i, (m, mu, p, n) in enumerate(pl):
    say(f"m={m}: log LR+ gap {mu:+.3f} p={p:.1e} holm={hl[i]:.1e} n={n} | "
        f"-log LR- gap {pn[i][1]:+.3f} p={pn[i][2]:.1e} holm={hl[len(pl) + i]:.1e}")

# ---------------------------------------------------------------- (h) flips
say("\n## (h) FM ranking flip rate (macro-F1), T1' and DEV")
for name in ("T1", "DEV"):
    df, _ = res[name]
    rr = df[df.strategy == "random"]
    nrep = rr.rep.max() + 1
    S = df[df.strategy == "stratified"].groupby(["dataset", "m", "learner"]).st_macro_f1.mean()
    R = rr.groupby(["dataset", "m", "learner"]).st_macro_f1.mean()
    if name == "T1":
        h1 = rr[rr.rep < 5].groupby(["dataset", "m", "learner"]).st_macro_f1.mean()
        h2 = rr[rr.rep >= 5].groupby(["dataset", "m", "learner"]).st_macro_f1.mean()
    else:  # DEV: 20 random draws -> two disjoint 5-draw halves (0-4, 5-9) as in E17 text
        h1 = rr[rr.rep < 5].groupby(["dataset", "m", "learner"]).st_macro_f1.mean()
        h2 = rr[(rr.rep >= 5) & (rr.rep < 10)].groupby(["dataset", "m", "learner"]).st_macro_f1.mean()

    def flip(a, b):
        fr = []
        for (d, m), ga in a.groupby(level=[0, 1]):
            if (d, m) not in b.index.droplevel(2):
                continue
            gb = b.loc[(d, m)]
            ga = ga.droplevel([0, 1])
            k = ga.index.intersection(gb.index)
            if len(k) < 4:
                continue
            x, y = ga[k].to_numpy(), gb[k].to_numpy()
            dx = np.sign(x[:, None] - x[None, :])
            dy = np.sign(y[:, None] - y[None, :])
            iu = np.triu_indices(len(k), 1)
            fr.append(np.mean(dx[iu] != dy[iu]))
        return np.array(fr)

    f_prot = flip(S, R)
    # Noise must be computed on the SAME cells (where stratified is feasible)
    keep = S.index
    f_noise_same = flip(h1[h1.index.isin(keep)], h2[h2.index.isin(keep)])
    f_noise_all = flip(h1, h2)
    say(f"{name}: random reps={nrep}; protocol flip={f_prot.mean():.3f} (cells {len(f_prot)}); "
        f"noise (5 vs 5 random, same cells)={f_noise_same.mean():.3f}; noise all cells={f_noise_all.mean():.3f}")
    if name == "T1":
        # alternative noise: 5 random draws vs the other 5 matched to 5 stratified? report also 5-vs-10 asymmetry
        h10 = rr.groupby(["dataset", "m", "learner"]).st_macro_f1.mean()
        say("   note: protocol compares 5 S draws vs 10 R draws; noise compares 5 vs 5")

# accuracy flips for completeness
(OUT / "v_e17.txt").write_text("\n".join(lines) + "\n")
