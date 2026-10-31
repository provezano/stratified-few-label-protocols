"""E6 (exploratory, R3-W2/R3-Q3): heterogeneity of the accuracy gap and pool imbalance.

Per DEV table, FM and budget: ΔA = mean stratified accuracy − mean random accuracy
(stored Study 7 Part (a) metrics; 5 stratified and 20 random draws per cell), and
its mean over m ∈ {1,3,5,10} (the contrast-(i) unit). Pool traits from the full
table: TV = ½‖π − u‖₁ (‖π − u‖₁ = 2·TV), π_min.

Prop. 2(ii) (main.tex, H2, binary): the true prior is accuracy-optimal and random
contexts beat stratified ones once B ≳ B* = 1/(π₁π₂τ*²), τ* = log(π₂/π₁). Predicted
sign: ΔA decreases with imbalance, more so at larger budgets; for binary tables
ΔA < 0 when B > B*. Checked by (a) Spearman ρ(ΔA, TV) and ρ(ΔA, π_min) with
5000-resample bootstrap CIs over tables; (b) OLS ΔA = a + b·TV with a pairs
bootstrap CI for b; (c) on binary tables, the share of (table, FM, m) cells whose
sign agrees with sign(B* − B) among cells with |ΔA| > 0.01, and Spearman of ΔA with
log(B/B*). n = 20 tables, nothing adjusted (exploratory).

Writes out/e6_cells.csv, out/e6_assoc.csv, out/e6_threshold.csv,
out/E6_SUMMARY.md
(the figure out/e6_accuracy_imbalance.{pdf,png} is drawn by figures/make_figures.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import common as cm  # noqa: E402


def cells() -> pd.DataFrame:
    rows = []
    for did in cm.dev_ids():
        tr = cm.pool_traits(cm.dev_data(did).y)
        for fm in cm.FMS:
            df = cm.load_a(did, fm)
            g = df.groupby(["mult", "strategy"])[["accuracy", "macro_f1", "balanced_accuracy"]].mean().unstack("strategy")
            for m in cm.MULTS:
                rows.append(dict(dataset=did, fm=fm, mult=m, B=m * tr["C"], **tr,
                                 dA=g.loc[m, ("accuracy", "stratified")] - g.loc[m, ("accuracy", "random")],
                                 dM=g.loc[m, ("macro_f1", "stratified")] - g.loc[m, ("macro_f1", "random")],
                                 dBA=g.loc[m, ("balanced_accuracy", "stratified")] - g.loc[m, ("balanced_accuracy", "random")]))
    out = pd.DataFrame(rows)
    pi1 = out.pi_min
    tau = np.log((1 - pi1) / pi1)
    with np.errstate(divide="ignore"):
        out["B_star"] = np.where(out.C == 2, 1 / (pi1 * (1 - pi1) * tau ** 2), np.nan)
    return out


def slope_ci(x, y, n_boot=5000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = np.polyfit(x, y, 1)
    rng = np.random.default_rng(seed)
    bs = []
    for idx in rng.integers(0, len(x), size=(n_boot, len(x))):
        if np.ptp(x[idx]) > 0:
            bs.append(np.polyfit(x[idx], y[idx], 1)[0])
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    return float(b[0]), float(b[1]), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), r2


def assoc(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fm_avg = c.groupby(["dataset", "mult"]).agg(dA=("dA", "mean"), tv=("tv", "first"), pi_min=("pi_min", "first")).reset_index()
    fm_avg["fm"] = "mean"
    allc = pd.concat([c, fm_avg])
    for fm, g in allc.groupby("fm"):
        per_m = {m: gm.set_index("dataset") for m, gm in g.groupby("mult")}
        per_m["all"] = g.groupby("dataset").agg(dA=("dA", "mean"), tv=("tv", "first"), pi_min=("pi_min", "first"))
        drop = c.loc[c.tv.idxmax(), "dataset"]
        per_m["all_no_max_tv"] = per_m["all"].drop(index=drop)
        for m, t in per_m.items():
            for pred in ("tv", "pi_min"):
                rho, lo, hi, p = cm.spearman_ci(t[pred].to_numpy(), t.dA.to_numpy())
                b, a, blo, bhi, r2 = slope_ci(t[pred].to_numpy(), t.dA.to_numpy())
                rows.append(dict(fm=fm, mult=str(m), predictor=pred, rho=rho, lo=lo, hi=hi, p=p,
                                 slope=b, intercept=a, slope_lo=blo, slope_hi=bhi, r2=r2,
                                 dA_mean=t.dA.mean(), dA_min=t.dA.min(), dA_max=t.dA.max(),
                                 n_pos=int((t.dA > 0.01).sum()), n_neg=int((t.dA < -0.01).sum())))
    return pd.DataFrame(rows)


def threshold(c: pd.DataFrame) -> pd.DataFrame:
    b = c[(c.C == 2)].copy()
    b["pred_sign"] = np.where(b.B > b.B_star, -1, 1)
    rows = []
    for fm, g in [*b.groupby("fm"), ("all", b)]:
        dec = g[g.dA.abs() > 0.01]
        lr = np.log(g.B / g.B_star.replace(np.inf, np.nan)).fillna(-10)
        rho, p = stats.spearmanr(lr, g.dA)
        rows.append(dict(fm=fm, n_cells=len(g), n_decisive=len(dec),
                         agree=float((np.sign(dec.dA) == dec.pred_sign).mean()) if len(dec) else np.nan,
                         n_beyond=int((g.B > g.B_star).sum()),
                         dA_beyond=float(g[g.B > g.B_star].dA.mean()),
                         dA_below=float(g[g.B <= g.B_star].dA.mean()),
                         rho_logratio=float(rho), p_logratio=float(p)))
    return pd.DataFrame(rows)


def main() -> None:
    c = cells()
    c.to_csv(cm.OUT / "e6_cells.csv", index=False)
    a = assoc(c)
    a.to_csv(cm.OUT / "e6_assoc.csv", index=False)
    th = threshold(c)
    th.to_csv(cm.OUT / "e6_threshold.csv", index=False)
    # The figure (out/e6_accuracy_imbalance.pdf) is drawn by figures/make_figures.py (fig_accimb) with the
    # shared FM encodings of the paper, including the exploratory FMs.
    L = ["# E6: accuracy gap heterogeneity vs pool imbalance (exploratory, DEV, n = 20)", "",
         "ΔA = stratified − random accuracy per table (stored Study 7 Part (a)). TV = ½‖π−u‖₁. "
         "Spearman ρ [5000-resample bootstrap CI]; OLS slope of ΔA on the predictor [pairs-bootstrap CI]. "
         "Not adjusted for multiplicity. `all_no_max_tv` = mean over m without the most imbalanced table "
         f"({int(c.loc[c.tv.idxmax(), 'dataset'])}, TV {c.tv.max():.2f}), an influential point for the OLS slope.", "",
         "## Spread of ΔA (mean over m)", "",
         "| FM | mean | min | max | tables ΔA > +0.01 | tables ΔA < −0.01 |", "|---|---|---|---|---|---|"]
    for fm in (*cm.FMS, "mean"):
        r = a[(a.fm == fm) & (a.mult == "all") & (a.predictor == "tv")].iloc[0]
        L.append(f"| {cm.FM_NAMES.get(fm, 'mean over FMs')} | {r.dA_mean:+.3f} | {r.dA_min:+.3f} | {r.dA_max:+.3f} | {r.n_pos} | {r.n_neg} |")
    L += ["", "## Association with imbalance", "",
          "| FM | m | ρ(ΔA, TV) | slope on TV [CI] | R² | ρ(ΔA, π_min) |", "|---|---|---|---|---|---|"]
    for fm in (*cm.FMS, "mean"):
        for m in ("1", "3", "5", "10", "all", "all_no_max_tv"):
            r = a[(a.fm == fm) & (a.mult == m) & (a.predictor == "tv")].iloc[0]
            q = a[(a.fm == fm) & (a.mult == m) & (a.predictor == "pi_min")].iloc[0]
            L.append(f"| {cm.FM_NAMES.get(fm, 'mean')} | {m} | {r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] | "
                     f"{r.slope:+.3f} [{r.slope_lo:+.3f}, {r.slope_hi:+.3f}] | {r.r2:.2f} | {q.rho:+.2f} [{q.lo:+.2f}, {q.hi:+.2f}] |")
    L += ["", "## Prop. 2(ii) threshold on the 14 binary tables", "",
          "B* = 1/(π₁π₂τ*²); predicted ΔA < 0 iff B > B*. Agreement over cells with |ΔA| > 0.01.", "",
          th.round(3).to_markdown(index=False), "",
          "Figure: out/e6_accuracy_imbalance.pdf, drawn by figures/make_figures.py (one panel per FM: "
          "ΔA mean over m vs TV with an OLS line; last panel: mean of all seven FMs)."]
    (cm.OUT / "E6_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
