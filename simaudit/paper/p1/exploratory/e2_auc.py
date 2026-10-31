"""E2 (exploratory): macro AUC over ALL classes, from the stored Study 7 probabilities.

Convention ("realistic AUC"): one-vs-rest AUC for every class k = 1..C on the
complement of the labelled rows, averaged with equal weights. An unseen class
has a constant score (its stored probability column is 0 under the Study 7
wrapper), so its AUC is 0.5; a single-class context gives a constant matrix,
hence macro AUC 0.5 rather than NaN. A class with no row left in the complement
(all its rows labelled) has no defined AUC and is dropped from that draw's
average (counted and reported). Ties are handled by sklearn's roc_auc_score
(average rank).

Alignment check first: for every stored row of results/study7/a (all strategies),
the probability matrix is reshaped to (n, C) with n from the rebuilt pool, and
the frozen ``complement_metrics`` must reproduce the stored macro_f1 to 1e-9.
If any row fails, E2 stops and writes the reason.

Writes out/e2_auc_runs.parquet, out/e2_auc_gaps.csv, out/E2_SUMMARY.md.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import common as cm  # noqa: E402
from simaudit.analysis.study7 import complement_metrics  # noqa: E402

TOL = 1e-9


def all_class_auc(y_true: np.ndarray, P: np.ndarray) -> tuple[float, int]:
    scores, dropped = [], 0
    for k in range(P.shape[1]):
        pos = y_true == k
        if 0 < pos.sum() < len(y_true):
            scores.append(float(roc_auc_score(pos, P[:, k])))
        else:
            dropped += 1
    return float(np.mean(scores)), dropped


def run(job: tuple[int, str]) -> tuple[list[dict], list[str]]:
    did, fm = job
    y = cm.dev_data(did).y
    df = pd.read_parquet(cm.S7 / "a" / f"{did}_{fm}.parquet")
    rows, errors = [], []
    for r in df.itertuples():
        proba = np.asarray(r.proba, dtype=np.float64)
        if proba.size != len(y) * r.n_classes:
            errors.append(f"{did} {fm} {r.strategy} m={r.mult} rep={r.rep}: proba size {proba.size} != n*C {len(y)}*{r.n_classes}")
            continue
        P = proba.reshape(len(y), r.n_classes)
        lab = np.asarray(r.labeled, dtype=np.int64)
        m = complement_metrics(y, P, lab)
        if abs(m["macro_f1"] - r.macro_f1) > TOL:
            errors.append(f"{did} {fm} {r.strategy} m={r.mult} rep={r.rep}: macro_f1 {m['macro_f1']} vs stored {r.macro_f1}")
        if r.strategy not in ("stratified", "random"):
            continue
        rest = np.setdiff1d(np.arange(len(y)), lab)
        auc, dropped = all_class_auc(y[rest], P[rest])
        rows.append(dict(dataset=did, learner=fm, strategy=r.strategy, mult=int(r.mult),
                         rep=int(r.rep), coverage=float(r.coverage), auc_all=auc,
                         auc_dropped=dropped, auc_seen=float(r.auc)))
    return rows, errors


def main() -> None:
    jobs = [(d, fm) for d in cm.dev_ids() for fm in cm.FMS]
    with ProcessPoolExecutor(5) as pool:
        res = list(pool.map(run, jobs))
    errors = [e for _, es in res for e in es]
    n_checked = sum(len(pd.read_parquet(cm.S7 / "a" / f"{d}_{fm}.parquet", columns=["rep"])) for d, fm in jobs)
    if errors:
        (cm.OUT / "E2_SUMMARY.md").write_text(
            "# E2 stopped: stored probabilities do not align with the rebuilt labels\n\n"
            + "\n".join(f"- {e}" for e in errors[:50]) + "\n")
        raise SystemExit(f"{len(errors)} alignment failures")
    runs = pd.DataFrame([r for rs, _ in res for r in rs])
    runs.to_parquet(cm.OUT / "e2_auc_runs.parquet", index=False)
    means = runs.groupby(["learner", "dataset", "mult", "strategy"])[["auc_all", "auc_seen"]].mean().unstack("strategy")
    out = []
    for metric in ("auc_all", "auc_seen"):
        d = (means[(metric, "stratified")] - means[(metric, "random")]).rename("d").reset_index()
        d = d.dropna()
        for (lr, m), g in d.groupby(["learner", "mult"]):
            out.append(dict(metric=metric, learner=lr, mult=m, **cm.summarize(g.d.to_numpy())))
        pooled = d.groupby(["learner", "dataset"]).d.mean().reset_index()
        for lr, g in pooled.groupby("learner"):
            out.append(dict(metric=metric, learner=lr, mult="all", **cm.summarize(g.d.to_numpy())))
    gaps = pd.concat([cm.add_holm(g) for _, g in pd.DataFrame(out).groupby(["metric", "mult"], sort=False)])
    gaps.to_csv(cm.OUT / "e2_auc_gaps.csv", index=False)
    nan_seen = int(runs.auc_seen.isna().sum())
    L = ["# E2: macro AUC over all classes (exploratory, DEV, stored Study 7 Part (a) probabilities)", "",
         f"Alignment: {n_checked} stored rows (all strategies, 4 FMs, 20 tables) re-scored with the frozen "
         f"`complement_metrics`; every recomputed macro-F1 equals the stored value to {TOL:g} (0 failures).", "",
         "Convention: one-vs-rest AUC for all C classes on the complement, equal weights; unseen class = constant "
         "score ⇒ AUC 0.5; single-class context ⇒ 0.5. "
         f"Classes with no complement row dropped: {int(runs.auc_dropped.sum())} class-draws in total. "
         f"The stored (seen-class) AUC is NaN in {nan_seen} of {len(runs)} stratified/random draws; its gap below is "
         "computed on table means of the defined draws only (as in the paper).", "",
         "Gap = stratified − random; mean over 20 tables [bootstrap 95% CI]; `*` Holm p < 0.05 over the 4 FMs within metric × budget.", "",
         "| FM | metric | m=1 | m=3 | m=5 | m=10 | mean over m (p_holm, win) |", "|---|---|---|---|---|---|---|"]
    for lr in cm.FMS:
        for metric in ("auc_all", "auc_seen"):
            cells = []
            for m in (1, 3, 5, 10, "all"):
                r = gaps[(gaps.learner == lr) & (gaps.metric == metric) & (gaps.mult.astype(str) == str(m))].iloc[0]
                c = f"{r['mean']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
                c += f" (p={r.p_holm:.2g}, win={r.win:.2f})" if m == "all" else (" *" if r.p_holm < 0.05 else "")
                cells.append(c)
            L.append(f"| {lr} | {metric} | " + " | ".join(cells) + " |")
    (cm.OUT / "E2_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
