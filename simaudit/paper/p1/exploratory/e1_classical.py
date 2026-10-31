"""E1 (exploratory): classical and plug-in learners under the stratified and random protocols.

Reuses the exact labelled sets of Study 7 Part (a) (results/study7/selections/
{did}_{stratified,random}.parquet: 5 stratified and 20 random draws per budget,
mult in {1,3,5,10}); these are the same rows the four FMs saw. Pools are rebuilt
read-only with the project loader. Metrics use the frozen Study 7 function
``complement_metrics`` (macro-F1 over all C classes with unseen F1 = 0,
accuracy, balanced accuracy over classes present in the complement).

Conventions
- Features are z-scored with the mean/sd of the whole pool (label-free, what a
  transductive practitioner can compute). Constant columns get sd 1.
- Missing classes: ``selection.strategies.predict_with_budget`` (the FM wrapper):
  fit on the seen classes, unseen classes get probability 0; with a single seen
  class every row is predicted as that class (constant probability 1).
- Learners: lr = LogisticRegression(C=1, lbfgs); knn1 = 1-NN; knn3 = k-NN with
  k = min(3, B); gnb = GaussianNB(var_smoothing=0.1) (the default 1e-9 gives a
  near-zero variance when a class has one row); lda = Gaussian plug-in with a
  pooled within-class covariance shrunk by Ledoit-Wolf (identity when the
  within-class scatter is zero, i.e. one row per class), prior = context shares;
  hgb = HistGradientBoosting(max_iter=100, max_depth=3, min_samples_leaf=1,
  learning_rate=0.1, no early stopping). No tuning (there are no spare labels).
- pc_*: the same metrics after the p / pi_hat prior correction of Study 7.

Check: the 2000 cached selections equal the `labeled` column of results/study7/a (verified, 0 mismatches).

Writes out/e1_runs.parquet, out/e1_gaps.csv, out/e1_theory.csv, out/E1_SUMMARY.md.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.covariance import ledoit_wolf  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

import common as cm  # noqa: E402
from simaudit.analysis.study7 import complement_metrics, prior_correct  # noqa: E402
from simaudit.selection.strategies import predict_with_budget  # noqa: E402

LEARNERS = ("lr", "knn1", "knn3", "gnb", "lda", "hgb")
PLUGIN = ("gnb", "lda")
WORKERS = 5


def _lda_proba(Xf, yf, Xq):
    k = int(yf.max()) + 1
    means = np.array([Xf[yf == c].mean(axis=0) for c in range(k)])
    resid = Xf - means[yf]
    d = Xf.shape[1]
    if np.allclose(resid, 0):
        cov = np.eye(d)
    else:
        cov, _ = ledoit_wolf(resid, assume_centered=True)
        cov = cov + 1e-6 * np.eye(d)
    prec = np.linalg.pinv(cov)
    prior = np.bincount(yf, minlength=k) / len(yf)
    W = means @ prec
    b = -0.5 * np.einsum("kd,kd->k", W, means) + np.log(prior)
    s = Xq @ W.T + b
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    return p / p.sum(axis=1, keepdims=True)


def predictor(name: str):
    def predict(Xf, yf, Xq, seed):
        if name == "lda":
            return _lda_proba(Xf, yf, Xq)
        if name == "lr":
            model = LogisticRegression(C=1.0, max_iter=5000)
        elif name == "knn1":
            model = KNeighborsClassifier(n_neighbors=1)
        elif name == "knn3":
            model = KNeighborsClassifier(n_neighbors=min(3, len(yf)))
        elif name == "gnb":
            model = GaussianNB(var_smoothing=0.1)
        elif name == "hgb":
            model = HistGradientBoostingClassifier(
                max_iter=100, max_depth=3, min_samples_leaf=1, learning_rate=0.1,
                early_stopping=False, random_state=int(seed) % (2**31 - 1),
            )
        else:
            raise ValueError(name)
        model.fit(Xf, yf)
        proba = np.zeros((len(Xq), int(yf.max()) + 1))
        proba[:, model.classes_] = model.predict_proba(Xq)
        return proba

    return predict


def run_dataset(did: int) -> list[dict]:
    data = cm.dev_data(did)
    X, y = data.X, data.y
    sd = X.std(axis=0)
    Xs = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    C = int(y.max()) + 1
    rows = []
    for strategy in ("stratified", "random"):
        sel = pd.read_parquet(cm.S7 / "selections" / f"{did}_{strategy}.parquet")
        for r in sel.itertuples():
            lab = np.asarray(r.labeled, dtype=np.int64)
            rest = np.setdiff1d(np.arange(len(y)), lab)
            for name in LEARNERS:
                proba = predict_with_budget(predictor(name), Xs, lab, y[lab], C, int(r.seed))
                met = complement_metrics(y, proba, lab)
                pc = complement_metrics(y, prior_correct(proba, y[lab]), lab)
                pcf1 = f1_score(y[rest], proba[rest].argmax(1), labels=list(range(C)),
                                average=None, zero_division=0)
                shares = np.bincount(y[lab], minlength=C) / len(lab)
                rows.append(dict(
                    dataset=did, learner=name, strategy=strategy, mult=int(r.mult),
                    rep=int(r.rep), budget=len(lab), n_classes=C,
                    coverage=float((shares > 0).mean()),
                    D=float(((shares - 1 / C) ** 2).sum()),
                    macro_f1=met["macro_f1"], accuracy=met["accuracy"],
                    balanced_accuracy=met["balanced_accuracy"],
                    pc_macro_f1=pc["macro_f1"], pc_accuracy=pc["accuracy"],
                    pc_balanced_accuracy=pc["balanced_accuracy"],
                    per_class_f1=pcf1.tolist(),
                ))
    return rows


def fm_runs() -> pd.DataFrame:
    keep = ["dataset", "learner", "strategy", "mult", "rep", "n_classes", "coverage",
            "macro_f1", "accuracy", "balanced_accuracy"]
    frames = []
    for did in cm.dev_ids():
        for fm in cm.FMS:
            df = pd.read_parquet(cm.S7 / "a" / f"{did}_{fm}.parquet", columns=keep)
            frames.append(df[df.strategy.isin(["stratified", "random"])])
    return pd.concat(frames, ignore_index=True)


def gap_table(runs: pd.DataFrame, metrics=("macro_f1", "accuracy", "balanced_accuracy")):
    means = runs.groupby(["learner", "dataset", "mult", "strategy"])[list(metrics)].mean()
    wide = means.unstack("strategy")
    rows = []
    for metric in metrics:
        delta = (wide[(metric, "stratified")] - wide[(metric, "random")]).rename("d")
        d = delta.reset_index()
        for (lr, mult), g in d.groupby(["learner", "mult"]):
            rows.append(dict(learner=lr, metric=metric, mult=mult, **cm.summarize(g.d.to_numpy())))
        pooled = d.groupby(["learner", "dataset"]).d.mean().reset_index()
        for lr, g in pooled.groupby("learner"):
            rows.append(dict(learner=lr, metric=metric, mult="all", **cm.summarize(g.d.to_numpy())))
    out = pd.DataFrame(rows)
    # Holm within metric x mult over the classical learners (FMs listed for context)
    parts = []
    for _, g in out.groupby(["metric", "mult"]):
        parts.append(cm.add_holm(g))
    return pd.concat(parts).sort_values(["metric", "mult", "learner"], key=lambda s: s.astype(str))


def theory_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Coverage part (observed vs Prop. 1 formulas) and balance-part checks per learner."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "theory_checks", cm.PROJECT_ROOT / "reports" / "theory_checks.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    out = []
    for did in cm.dev_ids():
        y = cm.dev_data(did).y
        counts = np.bincount(y)
        n, C = len(y), len(counts)
        pi = counts / n
        for (lr, mult), g in runs[runs.dataset == did].groupby(["learner", "mult"]):
            b = int(mult) * C
            s, r = g[g.strategy == "stratified"], g[g.strategy == "random"]
            miss = (r.coverage < 1).to_numpy()
            q = tc.p_miss(counts, b)
            row = dict(dataset=did, learner=lr, mult=int(mult), C=C)
            for met in ("macro_f1", "accuracy", "balanced_accuracy"):
                ms = s[met].mean()
                row[f"{met}_gap"] = ms - r[met].mean()
                row[f"{met}_miss_obs"] = float(((ms - r[met]) * miss).mean())
                row[f"{met}_bal_obs"] = float(((ms - r[met]) * ~miss).mean())
            if C == 2:
                other = (counts[::-1] - b) / (n - b)
                row["f1_cov_pred"] = float((q * (s.macro_f1.mean() - other / (1 + other))).sum())
                row["acc_cov_pred"] = float((q * (s.accuracy.mean() - other)).sum())
                tau = np.log(pi.max() / pi.min())
                row["B_cross"] = 1 / (pi[0] * pi[1] * tau**2) if tau > 0 else np.inf
            else:
                f1k = np.mean(np.vstack(s.per_class_f1.to_numpy()), axis=0)
                row["f1_cov_pred"] = float((q * f1k).mean())
                row["acc_cov_pred"] = float((q * counts / (n - b) * f1k).sum())
                row["B_cross"] = np.nan
            row["budget"] = b
            out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    path = cm.OUT / "e1_runs.parquet"
    if path.exists():
        runs = pd.read_parquet(path)
    else:
        with ProcessPoolExecutor(WORKERS) as pool:
            rows = [r for rows in pool.map(run_dataset, cm.dev_ids()) for r in rows]
        runs = pd.DataFrame(rows)
        runs.to_parquet(path, index=False)
    fms = fm_runs()
    both = pd.concat([runs.drop(columns=["per_class_f1"]), fms], ignore_index=True)
    gaps = gap_table(both)
    pcg = gap_table(runs[runs.learner.isin(PLUGIN + ("lr", "hgb"))].assign(
        macro_f1=lambda d: d.pc_macro_f1, accuracy=lambda d: d.pc_accuracy,
        balanced_accuracy=lambda d: d.pc_balanced_accuracy,
        learner=lambda d: d.learner + "_pc"))
    gaps = pd.concat([gaps, pcg], ignore_index=True)
    gaps.to_csv(cm.OUT / "e1_gaps.csv", index=False)
    th = theory_table(runs)
    th.to_csv(cm.OUT / "e1_theory.csv", index=False)
    write_summary(gaps, th, runs)


def _fmt(r) -> str:
    return f"{r['mean']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"


def write_summary(gaps: pd.DataFrame, th: pd.DataFrame, runs: pd.DataFrame) -> None:
    L = ["# E1: classical and plug-in learners under the two protocols (exploratory, DEV)", "",
         "Same labelled sets as Study 7 Part (a) (5 stratified, 20 random draws per budget). "
         "Gap = stratified − random, table means over draws; mean over 20 tables, bootstrap 95% CI "
         "(10k), win rate, two-sided Wilcoxon, Holm within metric × budget over the 6 classical + 4 FM "
         "learners (FMs are context only). Conventions in `e1_classical.py`.", ""]
    for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
        g = gaps[(gaps.metric == metric) & ~gaps.learner.str.endswith("_pc")]
        L += [f"## {metric}: stratified − random", "",
              "| learner | m=1 | m=3 | m=5 | m=10 | mean over m (p_holm, win) |", "|---|---|---|---|---|---|"]
        for lr in list(LEARNERS) + list(cm.FMS):
            cells = []
            for m in (1, 3, 5, 10, "all"):
                r = g[(g.learner == lr) & (g.mult.astype(str) == str(m))].iloc[0]
                c = _fmt(r)
                if m == "all":
                    c += f" (p={r.p_holm:.2g}, win={r.win:.2f})"
                elif r.p_holm < 0.05:
                    c += " *"
                cells.append(c)
            L.append(f"| {lr} | " + " | ".join(cells) + " |")
        L += ["", "`*` = Holm-adjusted p < 0.05 within metric × budget.", ""]
    g = gaps[gaps.learner.str.endswith("_pc")]
    L += ["## After the p / π̂ prior correction (macro-F1, stratified − random)", "",
          "| learner | m=1 | m=3 | m=5 | m=10 | mean |", "|---|---|---|---|---|---|"]
    for lr in sorted(g.learner.unique()):
        gg = g[(g.learner == lr) & (g.metric == "macro_f1")]
        L.append(f"| {lr} | " + " | ".join(_fmt(gg[gg.mult.astype(str) == str(m)].iloc[0])
                                           for m in (1, 3, 5, 10, "all")) + " |")
    # theory
    L += ["", "## Theory checks (per learner; means over the 20 tables)", "",
          "Coverage part: observed P(miss)·(M^S − M^R|miss) against Prop. 1 (binary: exact absorption "
          "formula; multiclass: first-order (1/C)Σ q_k f_k). Pearson r across tables per budget.", "",
          "| learner | m | F1 cov obs | F1 cov pred | r(F1) | acc cov obs | acc cov pred | F1 bal obs | BA bal obs | acc bal obs |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for (lr, m), g2 in th.groupby(["learner", "mult"]):
        r = np.corrcoef(g2.macro_f1_miss_obs, g2.f1_cov_pred)[0, 1] if g2.f1_cov_pred.std() > 0 else np.nan
        L.append(f"| {lr} | {m} | {g2.macro_f1_miss_obs.mean():.3f} | {g2.f1_cov_pred.mean():.3f} | {r:.2f} | "
                 f"{g2.accuracy_miss_obs.mean():.3f} | {g2.acc_cov_pred.mean():.3f} | {g2.macro_f1_bal_obs.mean():.3f} | "
                 f"{g2.balanced_accuracy_bal_obs.mean():.3f} | {g2.accuracy_bal_obs.mean():.3f} |")
    # Prop. balance (H2) checks for plug-ins, m >= 3
    L += ["", "## Prop. 2 (balance, H2) on the plug-in learners (GNB, LDA), covered-draw part, m ≥ 3", ""]
    sub = th[th.learner.isin(PLUGIN) & (th.mult >= 3)]
    ba_pos = (sub.balanced_accuracy_bal_obs > 0).mean()
    order = ((sub.accuracy_bal_obs <= sub.macro_f1_bal_obs) & (sub.macro_f1_bal_obs <= sub.balanced_accuracy_bal_obs)).mean()
    order_ab = (sub.accuracy_bal_obs <= sub.balanced_accuracy_bal_obs).mean()
    binsub = sub[sub.C == 2]
    pred_neg = binsub.budget >= binsub.B_cross
    agree = ((binsub.accuracy_bal_obs < 0) == pred_neg).mean()
    L += [f"- (i) BA balance part > 0 in {ba_pos:.2f} of {len(sub)} table × budget × learner cells.",
          f"- (iv) ordering Δacc ≤ ΔF1 ≤ ΔBA (balance parts) holds in {order:.2f} of cells; Δacc ≤ ΔBA in {order_ab:.2f}.",
          f"- (ii) binary: sign of the accuracy balance part agrees with the crossover rule B ≥ 1/(π1π2τ*²) in "
          f"{agree:.2f} of {len(binsub)} binary cells (predicted random-better in {pred_neg.mean():.2f}).", ""]
    for lr in PLUGIN:
        s2 = sub[sub.learner == lr]
        L.append(f"  - {lr}: BA>0 {(s2.balanced_accuracy_bal_obs > 0).mean():.2f}; ordering "
                 f"{((s2.accuracy_bal_obs <= s2.macro_f1_bal_obs) & (s2.macro_f1_bal_obs <= s2.balanced_accuracy_bal_obs)).mean():.2f}")
    L.append(f"- Cells with a BA balance part below −0.01 (beyond noise of 5 vs 20 draws): "
             f"{(sub.balanced_accuracy_bal_obs < -0.01).mean():.2f}.")
    L += ["", "Mean balance parts over the 20 tables (ordering of the means, Prop. 2(iv) reading):", "",
          "| learner | m | Δacc | ΔF1 | ΔBA | acc ≤ F1 ≤ BA | BA bal: mean [CI], Wilcoxon p |", "|---|---|---|---|---|---|---|"]
    for (lr, m), g2 in th[th.mult >= 3].groupby(["learner", "mult"]):
        a, f, b = g2.accuracy_bal_obs.mean(), g2.macro_f1_bal_obs.mean(), g2.balanced_accuracy_bal_obs.mean()
        s = cm.summarize(g2.balanced_accuracy_bal_obs.to_numpy())
        L.append(f"| {lr} | {m} | {a:+.3f} | {f:+.3f} | {b:+.3f} | {a <= f <= b} | "
                 f"{s['mean']:+.3f} [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}], p={s['p']:.2g} |")
    L += ["", "Note: knn3 at m = 1 uses k = B = C, i.e. a vote over the whole context; under the stratified "
          "protocol this is a tie broken towards the lowest class index, so its m = 1 row is degenerate."]
    (cm.OUT / "E1_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
