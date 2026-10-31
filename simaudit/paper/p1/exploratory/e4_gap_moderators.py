"""E4 (exploratory): why the protocol gap varies across the 20 DEV tables.

Response: per-table macro-F1 gap (stratified − random), table means over the
stored Study 7 Part (a) draws, averaged over the four FMs (per-FM values in the
CSV), per budget m and averaged over m in {1,3,5,10} (the contrast (i) unit).
Table traits (full table, read-only loader): C, minimum class share pi_min,
imbalance ratio IR = n_max / n_min, n, d, separability = 5-fold stratified CV
accuracy and balanced accuracy of 5-NN on z-scored features (fold seed 0), and
two theory quantities: the expected share of missed classes at B = m C,
(1/C) Σ q_k(B) (Prop. 1), and the prior distance ||pi − u||² (bias term of the
balance section). Spearman rho with a 5000-resample bootstrap CI over tables;
n = 20, no model fitting; many correlations, none adjusted (exploratory).
Sensitivity: without 694 (identifier column leaks the class).

Writes out/e4_tables.csv, out/e4_spearman.csv, out/E4_SUMMARY.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import gammaln  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

import common as cm  # noqa: E402


def q_mean(counts: np.ndarray, b: int) -> float:
    n = counts.sum()
    lc = lambda a, k: gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1) if k <= a else -np.inf  # noqa: E731
    return float(np.mean([np.exp(lc(n - c, b) - lc(n, b)) for c in counts]))


def traits() -> pd.DataFrame:
    rows = []
    for did in cm.dev_ids():
        d = cm.dev_data(did)
        X, y = d.X, d.y
        sd = X.std(axis=0)
        Xs = (X - X.mean(0)) / np.where(sd > 0, sd, 1)
        counts = np.bincount(y)
        C = len(counts)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        pred = cross_val_predict(KNeighborsClassifier(5), Xs, y, cv=cv)
        rec = [np.mean(pred[y == k] == k) for k in range(C)]
        pi = counts / len(y)
        row = dict(dataset=did, C=C, pi_min=pi.min(), IR=counts.max() / counts.min(), n=len(y),
                   d=X.shape[1], sep_acc=float(np.mean(pred == y)), sep_ba=float(np.mean(rec)),
                   prior_dist=float(((pi - 1 / C) ** 2).sum()))
        for m in cm.MULTS:
            row[f"qbar_m{m}"] = q_mean(counts, m * C)
        rows.append(row)
    return pd.DataFrame(rows).set_index("dataset")


def gaps() -> pd.DataFrame:
    frames = []
    for did in cm.dev_ids():
        for fm in cm.FMS:
            df = pd.read_parquet(cm.S7 / "a" / f"{did}_{fm}.parquet",
                                 columns=["dataset", "learner", "strategy", "mult", "macro_f1"])
            frames.append(df[df.strategy.isin(["stratified", "random"])])
    df = pd.concat(frames)
    w = df.groupby(["dataset", "learner", "mult", "strategy"]).macro_f1.mean().unstack()
    g = (w.stratified - w.random).rename("gap").to_frame()
    g["strat"] = w.stratified
    return g.reset_index()


def main() -> None:
    tr = traits()
    g = gaps()
    per = g.pivot_table(index="dataset", columns=["learner", "mult"], values="gap")
    out = tr.copy()
    for m in cm.MULTS:
        out[f"gap_m{m}"] = g[g.mult == m].groupby("dataset").gap.mean()
        for fm in cm.FMS:
            out[f"gap_{fm}_m{m}"] = per[(fm, m)]
    out["gap_all"] = g.groupby("dataset").gap.mean()
    for m in cm.MULTS:
        # Prop. 1(iii) proxy: share of missed classes x mean stratified macro-F1 (uses labels; descriptive)
        out[f"covterm_m{m}"] = out[f"qbar_m{m}"] * g[g.mult == m].groupby("dataset").strat.mean()
    for fm in cm.FMS:
        out[f"gap_{fm}_all"] = g[g.learner == fm].groupby("dataset").gap.mean()
    out.to_csv(cm.OUT / "e4_tables.csv")
    preds = ["C", "pi_min", "IR", "n", "d", "sep_acc", "sep_ba", "prior_dist"]
    rows = []
    for subset, data in (("all", out), ("no694", out.drop(index=694))):
        for resp in ["gap_all", "gap_m1", "gap_m3", "gap_m5", "gap_m10"]:
            m = resp.split("_m")[-1] if "_m" in resp else None
            extra = [f"qbar_m{m}", f"covterm_m{m}"] if m else ["qbar_m1", "covterm_m1"]
            for p in preds + extra:
                rho, lo, hi, pv = cm.spearman_ci(data[p].to_numpy(), data[resp].to_numpy())
                rows.append(dict(subset=subset, response=resp, predictor=p, rho=rho, lo=lo, hi=hi, p=pv,
                                 n=len(data)))
        # per-FM, averaged over m
        for fm in cm.FMS:
            for p in preds + ["qbar_m1"]:
                rho, lo, hi, pv = cm.spearman_ci(data[p].to_numpy(), data[f"gap_{fm}_all"].to_numpy())
                rows.append(dict(subset=subset, response=f"gap_{fm}_all", predictor=p, rho=rho, lo=lo, hi=hi,
                                 p=pv, n=len(data)))
    sp = pd.DataFrame(rows)
    sp.to_csv(cm.OUT / "e4_spearman.csv", index=False)
    L = ["# E4: moderators of the per-table protocol gap (exploratory, DEV, n = 20)", "",
         "Response: macro-F1 gap stratified − random per table (mean over the 4 FMs; Study 7 Part (a) stored runs). "
         "Spearman ρ [bootstrap 95% CI], unadjusted p. qbar_m = expected share of missed classes at B = mC (Prop. 1); "
         "prior_dist = ||π − u||²; sep_* = 5-fold 5-NN accuracy / balanced accuracy on the full table.", ""]
    for subset in ("all", "no694"):
        L += [f"## Tables: {subset}", "", "| predictor | mean over m | m=1 | m=3 | m=5 | m=10 |", "|---|---|---|---|---|---|"]
        s = sp[sp.subset == subset]
        for p in preds + ["qbar", "covterm"]:
            cells = []
            for resp in ["gap_all", "gap_m1", "gap_m3", "gap_m5", "gap_m10"]:
                pp = p if p not in ("qbar", "covterm") else (
                    f"{p}_m{resp.split('_m')[-1]}" if "_m" in resp else f"{p}_m1")
                r = s[(s.response == resp) & (s.predictor == pp)].iloc[0]
                star = " *" if r.lo > 0 or r.hi < 0 else ""
                cells.append(f"{r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]{star}")
            L.append(f"| {p} | " + " | ".join(cells) + " |")
        L += ["", "`*` CI excludes 0. qbar and covterm use the budget of the column (m = 1 for the mean over m); "
              "covterm = qbar × mean stratified macro-F1 (the Prop. 1(iii) coverage term with f_k replaced by its mean; uses labels).", ""]
    L += ["## Per FM (mean over m, all 20 tables)", "", "| predictor | " + " | ".join(cm.FMS) + " |",
          "|---|" + "---|" * len(cm.FMS)]
    s = sp[sp.subset == "all"]
    for p in preds + ["qbar_m1"]:
        L.append(f"| {p} | " + " | ".join(
            (lambda r: f"{r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]")(s[(s.response == f'gap_{fm}_all') & (s.predictor == p)].iloc[0])
            for fm in cm.FMS) + " |")
    L += ["", "## Table traits and gaps", "", out[preds + ["qbar_m1", "gap_m1", "gap_m3", "gap_m10", "gap_all"]].round(3).to_markdown()]
    (cm.OUT / "E4_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
