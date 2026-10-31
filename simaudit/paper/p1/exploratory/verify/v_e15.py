"""Verification of E15 claims (a), (b), (c) from verify/out/fits_t1.parquet.

Run: OMP_NUM_THREADS=1 nice -n 15 uv run python paper/p1/exploratory/verify/v_e15.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent / "out"
t1 = pd.read_parquet(OUT / "fits_t1.parquet")
lines: list[str] = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


# Higher-is-better scores
t1["auc_imp"] = t1.st_auc.fillna(0.5)           # E15 convention (NaN -> 0.5)
t1["auc_pre"] = t1.st_auc                       # prereg: NaN excluded from mean
t1["neg_ll"] = -t1.st_log_loss
t1["neg_brier"] = -t1.brier
METRICS = ["st_accuracy", "st_balanced_accuracy", "st_macro_f1", "auc_imp", "auc_pre",
           "neg_ll", "neg_brier"]


def curves(df, metric):
    return df.groupby(["dataset", "strategy", "m"])[metric].mean().unstack("m")


# --------------------------------------------------------------- (a)
say("## (a) required m per class, random, fm2 (median over 52 tables)")
f2 = t1[t1.learner == "fm2"]
res = []
for met in METRICS:
    cv = curves(f2, met).xs("random", level="strategy")
    for eps in (0.05, 0.02):
        for ref in ("own_max", "at_last_m"):
            req = []
            for key, row in cv.iterrows():
                r = row.dropna()
                target = (r.max() if ref == "own_max" else r.iloc[-1]) - eps
                req.append(float(r.index[np.argmax(r.to_numpy() >= target)]))
            req = np.array(req)
            res.append(dict(metric=met, eps=eps, ref=ref, median_m=np.median(req),
                            q25=np.percentile(req, 25), q75=np.percentile(req, 75)))
a = pd.DataFrame(res)
a.to_csv(OUT / "e15_a_required_m.csv", index=False)
say(a.pivot_table(index="metric", columns=["eps", "ref"], values="median_m").to_string())
lastm = curves(f2, "st_accuracy").xs("random", level="strategy").apply(
    lambda r: r.dropna().index.max(), axis=1)
say("max feasible m for random:", lastm.value_counts().to_dict())


# --------------------------------------------------------------- (b)
def m_eq(ms, rand, score):
    env = np.maximum.accumulate(rand)
    if score <= env[0]:
        return ms[0], "left"
    if score > env[-1]:
        return np.inf, "right"
    i = int(np.argmax(env >= score))
    t = (score - env[i - 1]) / (env[i] - env[i - 1])
    return float(np.exp(np.log(ms[i - 1]) + t * (np.log(ms[i]) - np.log(ms[i - 1])))), "pt"


def rho_values(df, fm, sel, met, m=1):
    cv = curves(df[df.learner == fm], met)
    rnd = cv.xs("random", level="strategy")
    sc = cv.xs(sel, level="strategy")
    out = []
    for key in sc.index.intersection(rnd.index):
        r = rnd.loc[key].dropna()
        if m not in sc.columns or not np.isfinite(sc.loc[key, m]) or m not in r.index:
            continue
        v, kind = m_eq(r.index.to_numpy(float), r.to_numpy(float), sc.loc[key, m])
        out.append(v / m)
    return np.array(out)


def upper_median(v):
    v = np.sort(v)
    return float(v[int(np.ceil(0.5 * len(v))) - 1])


say("\n## (b) label-equivalence rho(1) medians (upper/Turnbull-style median; plain median)")
rng = np.random.default_rng(1)
rows = []
for fm in ("fm2", "fm1", "fm3", "fm5"):
    for sel in ("kmedoids_nested", "lucos_nested", "stratified"):
        for met in METRICS:
            v = rho_values(t1, fm, sel, met)
            bs = [upper_median(rng.choice(v, len(v))) for _ in range(2000)] if fm == "fm2" else [np.nan]
            rows.append(dict(fm=fm, sel=sel, metric=met, n=len(v), med=upper_median(v),
                             plain_median=float(np.median(v)),
                             ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5),
                             share_gt1=float(np.mean(v > 1))))
b = pd.DataFrame(rows)
b.to_csv(OUT / "e15_b_rho1.csv", index=False)
say(b[b.fm == "fm2"].round(3).to_string(index=False))
say("accuracy rho(1) range over FMs (selectors):",
    b[(b.metric == "st_accuracy") & (b.sel != "stratified")].med.agg(["min", "max"]).round(2).to_dict())
say("macro-F1 rho(1) range over FMs (selectors):",
    b[(b.metric == "st_macro_f1") & (b.sel != "stratified")].med.agg(["min", "max"]).round(2).to_dict())

# --------------------------------------------------------------- (c)
say("\n## (c) mechanism at m=1, fm2 (medians of per-table means)")
g = f2[f2.m == 1].groupby(["dataset", "strategy"]).agg(
    rec_maj=("rec_maj", "mean"), rec_min=("rec_min", "mean"),
    lab_maj=("lab_maj_share", "mean"), pred_maj=("pred_maj_rate", "mean"),
    acc=("st_accuracy", "mean"), mf1=("st_macro_f1", "mean"),
    cov=("n_seen", "mean"), C=("C", "first"), ir=("ir_pool", "first"),
    ir_full=("ir_full", "first")).reset_index()
g["cov_share"] = g["cov"] / g.C
say(g.groupby("strategy")[["rec_maj", "rec_min", "lab_maj", "pred_maj", "cov_share"]]
    .median().round(3).to_string())
w = g.pivot(index="dataset", columns="strategy")
for sel in ("kmedoids_nested", "lucos_nested"):
    gap = (w["acc"][sel] - w["acc"]["random"]) - (w["mf1"][sel] - w["mf1"]["random"])
    dmaj = w["lab_maj"][sel] - w["lab_maj"]["random"]
    logir = np.log(w["ir"]["random"])
    r1 = stats.spearmanr(gap, dmaj)[0]
    r2 = stats.spearmanr(gap, logir)[0]
    r3 = stats.spearmanr(gap, np.log(w["ir_full"]["random"]))[0]
    bs = []
    idx = np.arange(len(gap))
    for _ in range(4000):
        j = rng.choice(idx, len(idx))
        bs.append(stats.spearmanr(gap.to_numpy()[j], dmaj.to_numpy()[j])[0])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    dmin = (w["rec_min"][sel] - w["rec_min"]["random"])
    say(f"{sel}: Spearman(gap, d maj share)={r1:+.3f} [{lo:+.2f},{hi:+.2f}]; "
        f"(gap, log IR pool)={r2:+.3f}; (gap, log IR full)={r3:+.3f}; "
        f"d maj share mean={dmaj.mean():+.3f} p={stats.wilcoxon(dmaj).pvalue:.3g}; "
        f"d min recall mean={dmin.mean():+.3f} p={stats.wilcoxon(dmin).pvalue:.3g}; "
        f"d maj recall p={stats.wilcoxon(w['rec_maj'][sel] - w['rec_maj']['random']).pvalue:.2g}")
# binary-only view, to check the multiclass definition of majority/minority
gb = g[g.C == 2]
say("binary-only medians:\n" + gb.groupby("strategy")[["rec_maj", "rec_min"]].median().round(3).to_string())
(OUT / "v_e15.txt").write_text("\n".join(lines) + "\n")
