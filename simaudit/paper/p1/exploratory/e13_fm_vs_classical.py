"""E13 (exploratory, R3-W8): do in-context learners amplify the protocol gap?

Per DEV table and budget, the protocol gap Δ = stratified − random (mean over the
same stored Study 7 Part (a) labelled sets) of each FM (out/e6_cells.csv) is compared
with the gap of the classical learners of E1 (out/e1_runs.parquet) on the same tables,
budgets and labelled rows. Reference "classical" = mean of the gaps of LR, 1-NN, GNB,
LDA and HGB (3-NN left out as in E1: k = B at m = 1 is degenerate); every single
classical learner is also reported. Paired difference per table
D = Δ_FM − Δ_classical, mean over the 20 tables with bootstrap 95% CI (10k) and
Wilcoxon p (unadjusted; Holm over the 4 FMs within metric × m in the CSV).
D > 0 means the FM shows a larger gap than the classical learners.

Writes out/e13_diff.csv, out/E13_SUMMARY.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import common as cm  # noqa: E402

CLASSICAL = ("lr", "knn1", "gnb", "lda", "hgb")
METRICS = dict(macro_f1="dM", accuracy="dA", balanced_accuracy="dBA")


def classical_gaps() -> pd.DataFrame:
    r = pd.read_parquet(cm.OUT / "e1_runs.parquet")
    g = r.groupby(["learner", "dataset", "mult", "strategy"])[list(METRICS)].mean().unstack("strategy")
    out = pd.DataFrame({METRICS[k]: g[(k, "stratified")] - g[(k, "random")] for k in METRICS}).reset_index()
    return out


def main() -> None:
    fm = pd.read_csv(cm.OUT / "e6_cells.csv")[["dataset", "fm", "mult", *METRICS.values()]]
    cl = classical_gaps()
    ref = cl[cl.learner.isin(CLASSICAL)].groupby(["dataset", "mult"])[list(METRICS.values())].mean().reset_index()
    ref["learner"] = "classical_mean"
    cl = pd.concat([cl, ref])
    rows = []
    for lr, gl in cl.groupby("learner"):
        j = fm.merge(gl, on=["dataset", "mult"], suffixes=("_fm", "_cl"))
        for col in METRICS.values():
            j["D"] = j[f"{col}_fm"] - j[f"{col}_cl"]
            avg = j.groupby(["fm", "dataset"]).agg(D=("D", "mean"), a=(f"{col}_fm", "mean"), b=(f"{col}_cl", "mean")).reset_index()
            avg["mult"] = "all"
            jj = pd.concat([j.assign(mult=j.mult.astype(str), a=j[f"{col}_fm"], b=j[f"{col}_cl"]), avg])
            for (f, m), g in jj.groupby(["fm", "mult"]):
                s = cm.summarize(g.D.to_numpy())
                rows.append(dict(classical=lr, metric=col, fm=f, mult=m, gap_fm=g.a.mean(), gap_cl=g.b.mean(), **s))
    d = pd.DataFrame(rows)
    d = pd.concat([cm.add_holm(g) for _, g in d.groupby(["classical", "metric", "mult"], sort=False)])
    d.to_csv(cm.OUT / "e13_diff.csv", index=False)
    L = ["# E13: FM gap vs classical-learner gap on the same labelled sets (exploratory, DEV, n = 20)", "",
         "D = Δ_FM − Δ_classical per table (Δ = stratified − random); mean [bootstrap 95% CI]; `*` Holm over the 4 FMs < 0.05. "
         "Reference = mean over LR, 1-NN, GNB, LDA, HGB.", ""]
    for col, name in (("dM", "macro-F1"), ("dA", "accuracy"), ("dBA", "balanced accuracy")):
        L += [f"## {name}", "", "| FM | m | Δ_FM | Δ_classical | D [95% CI] | win |", "|---|---|---|---|---|---|"]
        for f in cm.FMS:
            for m in ("1", "3", "5", "10", "all"):
                r = d[(d.classical == "classical_mean") & (d.metric == col) & (d.fm == f) & (d.mult == m)].iloc[0]
                L.append(f"| {cm.FM_NAMES[f]} | {m} | {r.gap_fm:+.3f} | {r.gap_cl:+.3f} | {r['mean']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
                         f"{' *' if r.p_holm < 0.05 else ''} | {r.win:.2f} |")
        L.append("")
    L += ["## Per classical learner, macro-F1, mean over m: D [CI]", "", "| FM | " + " | ".join(CLASSICAL) + " |",
          "|---|" + "---|" * len(CLASSICAL)]
    for f in cm.FMS:
        cs = []
        for lr in CLASSICAL:
            r = d[(d.classical == lr) & (d.metric == "dM") & (d.fm == f) & (d.mult == "all")].iloc[0]
            cs.append(f"{r['mean']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
        L.append(f"| {cm.FM_NAMES[f]} | " + " | ".join(cs) + " |")
    (cm.OUT / "E13_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
