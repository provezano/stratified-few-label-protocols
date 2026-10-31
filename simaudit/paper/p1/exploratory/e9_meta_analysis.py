"""E9 (exploratory, R2-W5): between-table heterogeneity of the protocol gap.

(a) Random-effects meta-analysis per FM × m (and the mean over m) for macro-F1,
accuracy and balanced accuracy. Effect y_i = per-table gap (mean stratified − mean
random, stored Study 7 Part (a)); within-table variance v_i = s²_S/5 + s²_R/20
(draw-to-draw variances; for the mean over m, v_i = Σ_m v_im / 16). Estimators:
DerSimonian–Laird and REML (Fisher scoring); pooled mean with its 95% CI (normal),
τ², I², and the 95% prediction interval μ ± t_{k−2}·√(τ² + SE²) (Higgins et al. 2009).

(b) Variance components (balanced ANOVA moment estimators, truncated at 0). Draw-level
gap d_tfr = mean_S(t, f) − MF^R(t, f, r) over the 20 random draws r, per budget:
  per FM:     d = μ + a_t + e         → σ²_table, σ²_draw
  across FMs: d = μ + a_t + b_f + (ab)_tf + e → σ²_table, σ²_model, σ²_table×model, σ²_draw
The draw component is the variance of a single random context around its table mean;
the stratified side enters only through its 5-draw mean (its draw variance is in (a)).

The forest plot (DEV and T1', all FMs) is drawn by figures/make_figures.py (fig_forest).

Writes out/e9_meta.csv, out/e9_varcomp.csv, out/E9_SUMMARY.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import common as cm  # noqa: E402

METRICS = ("macro_f1", "accuracy", "balanced_accuracy")


def draws() -> pd.DataFrame:
    return pd.concat([cm.load_a(d, fm).drop(columns=["labeled"]) for d in cm.dev_ids() for fm in cm.FMS],
                     ignore_index=True).rename(columns={"learner": "fm"})


def effects(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = d.groupby(["fm", "dataset", "mult", "strategy"])[metric].agg(["mean", "var", "count"]).unstack("strategy")
    out = pd.DataFrame({"y": g[("mean", "stratified")] - g[("mean", "random")],
                        "v": g[("var", "stratified")] / g[("count", "stratified")]
                        + g[("var", "random")] / g[("count", "random")]}).reset_index()
    avg = out.groupby(["fm", "dataset"]).agg(y=("y", "mean"), v=("v", lambda v: v.sum() / len(v) ** 2)).reset_index()
    avg["mult"] = "all"
    out["mult"] = out.mult.astype(str)
    return pd.concat([out, avg], ignore_index=True)


def dl(y, v):
    w = 1 / v
    mu_f = (w * y).sum() / w.sum()
    Q = (w * (y - mu_f) ** 2).sum()
    k = len(y)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    return tau2, Q


def reml(y, v, tau2=None, it=500):
    tau2 = np.var(y) if tau2 is None else tau2
    for _ in range(it):
        w = 1 / (v + tau2)
        mu = (w * y).sum() / w.sum()
        # REML score and expected information (Viechtbauer 2005)
        P_trace = w.sum() - (w ** 2).sum() / w.sum()
        score = 0.5 * ((w ** 2 * (y - mu) ** 2).sum() - P_trace)
        info = 0.5 * ((w ** 2).sum() - 2 * (w ** 3).sum() / w.sum() + ((w ** 2).sum() / w.sum()) ** 2)
        new = max(0.0, tau2 + score / info)
        if abs(new - tau2) < 1e-12:
            tau2 = new
            break
        tau2 = new
    return tau2


def pooled(y, v, tau2):
    w = 1 / (v + tau2)
    mu = (w * y).sum() / w.sum()
    se = np.sqrt(1 / w.sum())
    k = len(y)
    t = stats.t.ppf(0.975, k - 2)
    pi = np.sqrt(tau2 + se ** 2) * t
    return mu, se, mu - 1.96 * se, mu + 1.96 * se, mu - pi, mu + pi


def meta(eff: pd.DataFrame, metric: str) -> list[dict]:
    rows = []
    for (fm, m), g in eff.groupby(["fm", "mult"]):
        y, v = g.y.to_numpy(), np.maximum(g.v.to_numpy(), 1e-10)
        tau2_dl, Q = dl(y, v)
        k = len(y)
        I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
        tau2_r = reml(y, v, tau2_dl)
        for est, tau2 in (("DL", tau2_dl), ("REML", tau2_r)):
            mu, se, lo, hi, plo, phi = pooled(y, v, tau2)
            rows.append(dict(metric=metric, fm=fm, mult=m, estimator=est, k=k, mu=mu, se=se, ci_lo=lo, ci_hi=hi,
                             tau2=tau2, tau=np.sqrt(tau2), I2=I2, Q=Q, p_Q=float(stats.chi2.sf(Q, k - 1)),
                             pi_lo=plo, pi_hi=phi, share_pos=float(np.mean(y > 0)), median_v=float(np.median(v))))
    return rows


def varcomp(d: pd.DataFrame, metric: str) -> list[dict]:
    rows = []
    s_mean = d[d.strategy == "stratified"].groupby(["fm", "dataset", "mult"])[metric].mean().rename("s")
    r = d[d.strategy == "random"].join(s_mean, on=["fm", "dataset", "mult"])
    r["g"] = r.s - r[metric]
    for m, gm in r.groupby("mult"):
        for fm, g in gm.groupby("fm"):
            R = g.groupby("dataset").size().iloc[0]
            T = g.dataset.nunique()
            tm = g.groupby("dataset").g.mean()
            ms_t = R * ((tm - g.g.mean()) ** 2).sum() / (T - 1)
            ms_e = ((g.g - g.dataset.map(tm)) ** 2).sum() / (T * (R - 1))
            rows.append(dict(metric=metric, mult=m, fm=fm, table=max(0.0, (ms_t - ms_e) / R), draw=ms_e))
        T, F = gm.dataset.nunique(), gm.fm.nunique()
        R = len(gm) // (T * F)
        grand = gm.g.mean()
        tf = gm.groupby(["dataset", "fm"]).g.mean()
        t_ = gm.groupby("dataset").g.mean()
        f_ = gm.groupby("fm").g.mean()
        ms_t = F * R * ((t_ - grand) ** 2).sum() / (T - 1)
        ms_f = T * R * ((f_ - grand) ** 2).sum() / (F - 1)
        inter = tf - t_.reindex(tf.index, level=0).to_numpy() - f_.reindex(tf.index, level=1).to_numpy() + grand
        ms_tf = R * (inter ** 2).sum() / ((T - 1) * (F - 1))
        resid = gm.g.to_numpy() - tf.reindex(list(zip(gm.dataset, gm.fm))).to_numpy()
        ms_e = (resid ** 2).sum() / (T * F * (R - 1))
        rows.append(dict(metric=metric, mult=m, fm="all", table=max(0.0, (ms_t - ms_tf) / (F * R)),
                         model=max(0.0, (ms_f - ms_tf) / (T * R)), table_x_model=max(0.0, (ms_tf - ms_e) / R),
                         draw=ms_e))
    return rows


def main() -> None:
    d = draws()
    rows, vc, effs = [], [], {}
    for metric in METRICS:
        effs[metric] = effects(d, metric)
        rows += meta(effs[metric], metric)
        vc += varcomp(d, metric)
    res = pd.DataFrame(rows)
    res.to_csv(cm.OUT / "e9_meta.csv", index=False)
    vcd = pd.DataFrame(vc)
    vcd.to_csv(cm.OUT / "e9_varcomp.csv", index=False)
    # Forest plot: figures/make_figures.py (fig_forest) draws DEV and T1' together, for all FMs.
    L = ["# E9: random-effects meta-analysis and variance components of the protocol gap (exploratory, DEV, k = 20)", "",
         "Unit = table. REML pooled mean [95% CI], τ (SD between tables), I², 95% prediction interval (t, k−2 df). "
         "DL estimates are in e9_meta.csv (close to REML).", ""]
    for metric in METRICS:
        L += [f"## {metric}", "", "| FM | m | μ [95% CI] | τ | I² | 95% PI | share > 0 |", "|---|---|---|---|---|---|---|"]
        for fm in cm.FMS:
            for m in ("1", "3", "5", "10", "all"):
                r = res[(res.metric == metric) & (res.fm == fm) & (res.mult == m) & (res.estimator == "REML")].iloc[0]
                L.append(f"| {cm.FM_NAMES[fm]} | {m} | {r.mu:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.tau:.3f} | "
                         f"{r.I2:.2f} | [{r.pi_lo:+.3f}, {r.pi_hi:+.3f}] | {r.share_pos:.2f} |")
        L.append("")
    L += ["## Variance components (SD scale, √σ²), draw-level gap d = mean_S − MF^R(draw)", "",
          "| metric | m | σ_table | σ_model | σ_table×model | σ_draw (one random context) | share of total: table / model / t×m / draw |",
          "|---|---|---|---|---|---|---|"]
    for metric in METRICS:
        for m in cm.MULTS:
            r = vcd[(vcd.metric == metric) & (vcd.mult == m) & (vcd.fm == "all")].iloc[0]
            tot = r.table + r.model + r.table_x_model + r.draw
            L.append(f"| {metric} | {m} | {np.sqrt(r.table):.3f} | {np.sqrt(r.model):.3f} | {np.sqrt(r.table_x_model):.3f} | "
                     f"{np.sqrt(r.draw):.3f} | {r.table / tot:.2f} / {r.model / tot:.2f} / {r.table_x_model / tot:.2f} / {r.draw / tot:.2f} |")
    L += ["", "Per FM (macro-F1): σ_table and σ_draw", "", "| FM | m=1 | m=3 | m=5 | m=10 |", "|---|---|---|---|---|"]
    for fm in cm.FMS:
        cs = []
        for m in cm.MULTS:
            r = vcd[(vcd.metric == "macro_f1") & (vcd.mult == m) & (vcd.fm == fm)].iloc[0]
            cs.append(f"{np.sqrt(r.table):.3f} / {np.sqrt(r.draw):.3f}")
        L.append(f"| {cm.FM_NAMES[fm]} | " + " | ".join(cs) + " |")
    L += ["", "Forest plot (DEV and T1', all FMs): figures/fig_forest.pdf, drawn by figures/make_figures.py"]
    (cm.OUT / "E9_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
