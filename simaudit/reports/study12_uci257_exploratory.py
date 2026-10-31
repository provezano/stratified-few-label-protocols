"""Study 12, EXPLORATORY (not pre-registered): UCI 257 label-spelling sensitivity.

T1' table pmlb_1000000257 (UCI 257) is stored with C = 5, but "Very Low" (code 3) and
"very_low" (code 4) are one class (method and label check:
paper/p1/exploratory/uci257/, commit R042). Two variants, re-scored from the stored
Study 12 probabilities (nothing refitted):
- DROP: every analysis on the other 51 tables;
- MERGE: columns 3 and 4 summed, true labels merged (C = 4). Valid for random draws;
  approximate for stratified (it stratified over 5 classes, so the merged class got 2m
  of 5m labels) and for redraw (coverage required 5 classes).
The ``_bal`` learners are recomputed with p / pi-hat on the merged context.
Writes reports/STUDY12_UCI257_EXPLORATORY.md.
"""

import numpy as np
import pandas as pd

from simaudit.analysis import study12 as a12
from simaudit.analysis.study7 import prior_correct
from simaudit.data.openml_loader import load
from simaudit.settings import PROJECT_ROOT, settings

OUT_DIR = settings.results_dir / "study12"
REPORT = PROJECT_ROOT / "reports" / "STUDY12_UCI257_EXPLORATORY.md"
KEY, DID = "pmlb_1000000257", -1000000257
MAPPING = np.array([0, 1, 2, 3, 3], dtype=np.int64)


def merged_rows() -> pd.DataFrame:
    """Every stored fit of table 257 re-scored with C = 4 (plus fm*_bal rows)."""
    y = MAPPING[load(DID, settings.data_cache_dir).y]
    rows = []
    for fm in a12.FMS:
        frame = pd.read_parquet(OUT_DIR / f"{KEY}_{fm}.parquet")
        for r in frame.to_dict("records"):
            q = np.asarray(r["query"], dtype=np.int64)
            lab = np.asarray(r["labeled"], dtype=np.int64)
            P5 = np.asarray(r["proba"], dtype=np.float64).reshape(len(q), 5)
            P = np.zeros((len(q), 4))
            np.add.at(P.T, MAPPING, P5.T)
            seen = np.unique(y[lab]).astype(np.int64)
            base = dict(dataset=KEY, strategy=r["strategy"], mult=r["mult"], rep=r["rep"],
                        attempt=r["attempt"], reused=r["reused"], n_classes=4,
                        coverage=len(seen) / 4, seconds=r["seconds"])
            rows.append(dict(base, learner=fm, **a12.panel_metrics(y[q], P, seen)))
            if fm in ("fm1", "fm5"):
                pc = a12.panel_metrics(y[q], prior_correct(P, y[lab]), seen)
                rows.append(dict(base, learner=f"{fm}_bal", **pc))
    return pd.DataFrame(rows)


def summary(runs: pd.DataFrame, label: str) -> list[str]:
    """Confirmatory family (Holm recomputed), S-1, S-2 at 1C, S-3 and S-4 for a variant."""
    table = a12.confirmatory(a12.cell_means(runs, "macro_f1"))
    reading = a12.overall(table)
    n_classes = runs.groupby("dataset")["n_classes"].first()
    out = [f"## {label} (n = {runs.dataset.nunique()}; "
           f"{int((n_classes == 2).sum())} binary / {int((n_classes > 2).sum())} multiclass)",
           "", "| ID | Mean Δ [95% CI] | p (TOST for ii-c) | Wilcoxon p | Holm p | Verdict |",
           "|---|---|---|---|---|---|"]
    for r in table.to_dict("records"):
        out.append(f"| {r['id']} | {r['mean']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                   f"{r['p']:.2g} | {r['p_wilcoxon']:.2g} | {r['p_holm']:.2g} | {r['verdict']} |")
    out += ["", f"Reading rules: C1 {reading['c1']}; ranking claim {reading['ranking']}.", ""]
    split = a12.split_analysis(runs, n_classes)
    out += ["Binary / multiclass split (mean [95% CI], Wilcoxon p; Mann–Whitney p):", "",
            "| Contrast | binary | multiclass | MW p |", "|---|---|---|---|"]
    for r in split.to_dict("records"):
        out.append(f"| {r['id']} | {r['binary_mean']:+.3f} [{r['binary_ci_low']:+.3f}, "
                   f"{r['binary_ci_high']:+.3f}], {r['binary_p']:.2g} | {r['multi_mean']:+.3f} "
                   f"[{r['multi_ci_low']:+.3f}, {r['multi_ci_high']:+.3f}], {r['multi_p']:.2g} | "
                   f"{r['mw_p']:.2g} |")
    red = a12.redraw_analysis(runs)
    auc = a12.auc_all_analysis(runs)["auc_all"]
    flips = a12.flip_rate(runs, a12.FMS + ("fm1_bal", "fm5_bal"))
    out += ["", "S-1 redraw (m 3–10): " + "; ".join(
        f"{r['id']} {r['mean']:+.3f} (Holm {r['p_holm']:.2g})" for r in red["table"].to_dict("records"))
        + f"; fm2 − fm5 under redraw {red['rank']['mean']:+.3f} (p {red['rank']['p']:.2g}).",
        "S-2 auc_all, strat − random at 1C: " + "; ".join(
        f"{r['learner']} {r['m1_mean']:+.3f} (Holm {r['m1_p_holm']:.2g})" for r in auc.to_dict("records")) + ".",
        f"S-3 flip rate {flips['rate']:.3f} vs noise {flips['noise_rate']:.3f} "
        f"(diff {flips['diff']:+.3f} [{flips['diff_ci'][0]:+.3f}, {flips['diff_ci'][1]:+.3f}]).", ""]
    return out


def main() -> None:
    """Write the exploratory report (base, drop, merge)."""
    runs = a12.load_runs(OUT_DIR)
    merged = merged_rows()
    variants = {
        "BASE (frozen analysis, as in STUDY12_REPORT.md)": runs,
        "DROP: without pmlb_1000000257": runs[runs.dataset != KEY],
        "MERGE: pmlb_1000000257 re-scored with C = 4": pd.concat(
            [runs[runs.dataset != KEY], merged], ignore_index=True),
    }
    own = merged[merged.learner.isin(a12.FMS)]
    lines = [
        "# Study 12, EXPLORATORY: UCI 257 label-spelling sensitivity",
        "",
        "**Not pre-registered; no confirmatory weight.** The confirmatory result is "
        "reports/STUDY12_REPORT.md (frozen analysis) and is not changed by this file. "
        "Table pmlb_1000000257 (UCI 257, User Knowledge Modeling) is stored with C = 5, but "
        "'Very Low' (26 rows) and 'very_low' (24 rows) are one class (label check and method: "
        "paper/p1/exploratory/uci257/, commit R042). Re-scored from the stored Study 12 "
        "probabilities, nothing refitted. MERGE is exact for random draws, approximate for "
        "stratified (stratified over 5 classes: the merged class got 2m of 5m labels) and for "
        "redraw (coverage required both spellings). Generated by "
        "`reports/study12_uci257_exploratory.py`.",
        "",
        f"Table 257 itself, merged vs stored (C = 4 vs 5), mean macro-F1 over FMs and reps: "
        + ", ".join(
            f"{s} {own[own.strategy == s].macro_f1.mean():.3f} vs "
            f"{runs[(runs.dataset == KEY) & (runs.strategy == s) & runs.learner.isin(a12.FMS)].macro_f1.mean():.3f}"
            for s in ("stratified", "random", "redraw")) + ".",
        "",
    ]
    for label, frame in variants.items():
        lines += summary(frame, label)
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
