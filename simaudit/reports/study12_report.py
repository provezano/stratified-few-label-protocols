"""Generate reports/STUDY12_REPORT.md from results/study12/ (prereg Study 12).

Usage: .venv/bin/python reports/study12_report.py
Runs the pre-registered analysis exactly once over the finished runs: the
confirmatory Holm family (Sec. 3.1), robustness (3.2), secondary S-1 to S-4
(3.3) and the Addendum 1 sensitivities (SENS-INT, SENS-ID).
"""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from simaudit.analysis import study12 as a12
from simaudit.runner.study9 import dataset_key
from simaudit.settings import PROJECT_ROOT, settings

OUT_DIR = settings.results_dir / "study12"
REPORT = PROJECT_ROOT / "reports" / "STUDY12_REPORT.md"
SECONDARY_METRICS = (
    "accuracy", "balanced_accuracy", "log_loss", "auc", "auc_all",
    "seen_macro_f1", "seen_accuracy",
)
COUNTER = ("pmlb:cloud", "pmlb:haberman", "uci:519", "pmlb:backache")
DUPLICATES = ("pmlb:breast_cancer", "pmlb:spambase", "pmlb:hypothyroid",
              "pmlb:splice", "uci:563")
CONTRAST_TEXT = {
    "(i)-fm1": "stratified − random, TabPFN v2 (m 1–10)",
    "(i)-fm2": "stratified − random, TabICL (m 1–10)",
    "(i)-fm3": "stratified − random, Kumo (m 1–10)",
    "(i)-fm5": "stratified − random, TabPFN v3.5 (m 1–10)",
    "(ii-a)": "TabICL − TabPFN v3.5, random (m 3–10)",
    "(ii-c)": "TabICL − TabPFN v3.5, stratified (m 3–10)",
}


def f(x: float, d: int = 3) -> str:
    """Signed float."""
    return f"{x:+.{d}f}"


def p(x: float) -> str:
    """Compact p-value."""
    return f"{x:.2g}"


def pool_keys() -> dict[str, str]:
    """Result key -> pool key."""
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())
    return {dataset_key(int(d["did"])): str(d["key"]) for d in pool["datasets"]}


def confirmatory_lines(table: pd.DataFrame) -> list[str]:
    """Markdown table of the Holm family."""
    lines = [
        "| ID | Contrast | Test | n | Mean Δ [95% CI] | Median | Win-rate | p | Holm p "
        "| TOST p | Bayes P(+)/ROPE/(−) | LODO mean range, max p | DEV mean (in CI?) "
        "| Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.to_dict("records"):
        lines.append(
            f"| {r['id']} | {CONTRAST_TEXT[r['id']]} | {r['test']} | {r['n']} | "
            f"{f(r['mean'])} [{f(r['ci_low'])}, {f(r['ci_high'])}] | {f(r['median'])} | "
            f"{r['win_rate']:.2f} | {p(r['p'])} | **{p(r['p_holm'])}** | {p(r['p_tost'])} | "
            f"{r['p_bayes_pos']:.2f}/{r['p_bayes_rope']:.2f}/{r['p_bayes_neg']:.2f} | "
            f"[{f(r['lodo_min'])}, {f(r['lodo_max'])}], {p(r['lodo_p_max'])} | "
            f"{f(r['dev_mean'])} ({'yes' if r['dev_in_ci'] else 'no'}) | "
            f"**{r['verdict']}** |"
        )
    return lines


def per_budget_lines(runs: pd.DataFrame, metric: str) -> list[str]:
    """Per-mult Δ for the six contrasts plus the balance option."""
    means = a12.cell_means(runs, metric)
    pairs = [(c[0], c[1], c[2]) for c in a12.CONFIRMATORY] + [
        ("(ii-b) fm2 − fm5_bal, random", ("fm2", "random"), ("fm5_bal", "random")),
        ("fm5_bal − fm5, random", ("fm5_bal", "random"), ("fm5", "random")),
        ("fm1_bal − fm1, random", ("fm1_bal", "random"), ("fm1", "random")),
    ]
    lines = ["| contrast | 1C | 3C | 5C | 10C |", "|---|---|---|---|---|"]
    for name, a, b in pairs:
        cells = a12.per_budget(means, a, b, metric)
        lines.append(
            f"| {name} | "
            + " | ".join(
                f"{f(c['mean'])} (p={c['p']:.3f}, TOST p={c['p_tost']:.2f})"
                for c in cells.values()
            )
            + " |"
        )
    return lines


def secondary_summary_lines(runs: pd.DataFrame) -> list[str]:
    """The six contrasts (Sec. 3.2) in the secondary metrics, unadjusted."""
    lines = ["| metric | " + " | ".join(c[0] for c in a12.CONFIRMATORY) + " |",
             "|---|" + "---|" * len(a12.CONFIRMATORY)]
    for metric in SECONDARY_METRICS:
        means = a12.cell_means(runs, metric)
        cells = []
        for cid, a, b, mults, _ in a12.CONFIRMATORY:
            v = a12.delta(means, a, b, mults, metric).to_numpy(float)
            low, high = a12.bootstrap_ci(v)
            cells.append(f"{f(v.mean())} [{f(low)}, {f(high)}] p={a12.wilcoxon_p(v):.2g}")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    return lines


def redraw_lines(res: dict[str, Any]) -> list[str]:
    """S-1 tables."""
    lines = [
        "Share of random draws covering every class (all FMs share draws): "
        + ", ".join(f"{m}C {v:.2f}" for m, v in res["covered"].items()),
        "",
        "| Contrast (m 3–10) | n | Mean Δ [95% CI] | Win-rate | p | Holm p (4) |",
        "|---|---|---|---|---|---|",
    ]
    for r in res["table"].to_dict("records"):
        lines.append(
            f"| {r['id']} | {r['n']} | {f(r['mean'])} [{f(r['ci_low'])}, {f(r['ci_high'])}] "
            f"| {r['win_rate']:.2f} | {p(r['p'])} | **{p(r['p_holm'])}** |"
        )
    rk = res["rank"]
    lines += [
        f"| fm2 − fm5 under redraw | {rk['n']} | {f(rk['mean'])} [{f(rk['ci_low'])}, "
        f"{f(rk['ci_high'])}] | {rk['win_rate']:.2f} | {p(rk['p'])} | — |",
        "",
        "Check at m = 1 (redraw ≡ stratified in distribution): "
        + "; ".join(f"{fm} {f(s['mean'])} (p={s['p']:.2g})"
                    for fm, s in res["check_m1"].items()),
        "",
        "Gap closed by conditioning on coverage, (strat − random) − (strat − redraw):",
        "",
        res["closed"].pivot(index="learner", columns="mult", values="closed")
        .round(3).to_markdown(),
        "",
        "(strat − redraw) per m:",
        "",
        res["closed"].pivot(index="learner", columns="mult", values="gap_redraw")
        .round(3).to_markdown(),
    ]
    return lines


def auc_lines(res: dict[str, Any]) -> list[str]:
    """S-2 tables."""
    lines = []
    for metric, label in (("auc_all", "auc_all (realistic: unseen class = fixed score)"),
                          ("auc", "auc (Study 7 convention: seen-class columns)")):
        t = res[metric]
        lines += [
            f"**{label}**, stratified − random:",
            "",
            "| FM | m=1: n, mean Δ [95% CI] | p | Holm p (4) | m 1–10: n, mean Δ [95% CI] | p |",
            "|---|---|---|---|---|---|",
        ]
        for r in t.to_dict("records"):
            lines.append(
                f"| {r['learner']} | {r['m1_n']}, {f(r['m1_mean'])} [{f(r['m1_ci_low'])}, "
                f"{f(r['m1_ci_high'])}] | {p(r['m1_p'])} | **{p(r['m1_p_holm'])}** | "
                f"{r['all_n']}, {f(r['all_mean'])} [{f(r['all_ci_low'])}, "
                f"{f(r['all_ci_high'])}] | {p(r['all_p'])} |"
            )
        lines.append("")
    share = res["auc_nan_share"].unstack("mult").round(3)
    lines += ["Share of fits whose Study 7 AUC is undefined (one seen class):", "",
              share.to_markdown()]
    return lines


def flip_lines(runs: pd.DataFrame) -> list[str]:
    """S-3 for six and four learners, macro-F1 and accuracy."""
    lines = []
    sets = (("6 learners", a12.FMS + ("fm1_bal", "fm5_bal")), ("4 FMs", a12.FMS))
    for metric in ("macro_f1", "accuracy"):
        for name, learners in sets:
            r = a12.flip_rate(runs, learners, metric)
            lines += [
                f"- **{metric}, {name}.** Protocol flip rate {r['rate']:.3f} "
                f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] ({r['flips']}/{r['decisive']} decisive "
                f"pairs); noise baseline (random reps 0–9 vs 10–19) {r['noise_rate']:.3f} "
                f"[{r['noise_ci'][0]:.3f}, {r['noise_ci'][1]:.3f}] ({r['noise_decisive']} "
                f"decisive); difference {f(r['diff'])} [{f(r['diff_ci'][0])}, "
                f"{f(r['diff_ci'][1])}]. Kendall τ per m: "
                + ", ".join(f"{m}C {t:.2f}" for m, t in r["tau"].items()),
            ]
            if metric == "macro_f1" and name == "6 learners":
                for m, o in r["orders"].items():
                    lines.append(f"  - {m}C stratified: {' > '.join(o['stratified'])}; "
                                 f"random: {' > '.join(o['random'])}")
    return lines


def split_lines(table: pd.DataFrame) -> list[str]:
    """S-4 table."""
    lines = ["| Contrast | binary n, mean [95% CI], p | multiclass n, mean [95% CI], p "
             "| Mann–Whitney p |", "|---|---|---|---|"]
    for r in table.to_dict("records"):
        lines.append(
            f"| {r['id']} | {r['binary_n']}, {f(r['binary_mean'])} [{f(r['binary_ci_low'])}, "
            f"{f(r['binary_ci_high'])}], {p(r['binary_p'])} | {r['multi_n']}, "
            f"{f(r['multi_mean'])} [{f(r['multi_ci_low'])}, {f(r['multi_ci_high'])}], "
            f"{p(r['multi_p'])} | {p(r['mw_p'])} |"
        )
    return lines


def sensitivity_lines(runs: pd.DataFrame, drop: set[str], label: str) -> list[str]:
    """Confirmatory family and S-1 recomputed without the flagged tables."""
    sub = runs[~runs.dataset.isin(drop)]
    table = a12.confirmatory(a12.cell_means(sub, "macro_f1"))
    red = a12.redraw_analysis(sub)
    flips = a12.flip_rate(sub, a12.FMS + ("fm1_bal", "fm5_bal"))
    out = [f"#### {label} (n = {sub.dataset.nunique()})", ""]
    out += confirmatory_lines(table)
    out += ["", f"Overall: C1 {a12.overall(table)['c1']}; ranking claim "
            f"{a12.overall(table)['ranking']}.", "",
            "S-1 (redraw, m 3–10): " + "; ".join(
                f"{r['id']} {f(r['mean'])} (Holm {p(r['p_holm'])})"
                for r in red["table"].to_dict("records"))
            + f"; fm2 − fm5 under redraw {f(red['rank']['mean'])} (p {p(red['rank']['p'])}).",
            f"S-3 flip rate {flips['rate']:.3f} vs noise {flips['noise_rate']:.3f} "
            f"(diff {f(flips['diff'])} [{f(flips['diff_ci'][0])}, {f(flips['diff_ci'][1])}]).",
            ""]
    return out


def jobs_lines(runs: pd.DataFrame) -> list[str]:
    """Completed jobs, fits and compute."""
    base = runs[runs.learner.isin(a12.FMS)]
    fitted = base[~base.reused.astype(bool)]
    by = base.groupby("learner").agg(datasets=("dataset", "nunique"), rows=("dataset", "size"))
    by["fits"] = fitted.groupby("learner").size()
    by["worker_hours"] = (fitted.groupby("learner")["seconds"].sum() / 3600).round(1)
    return [by.to_markdown(), "",
            f"Total: {len(fitted)} fits (incl. constant one-class contexts), "
            f"{fitted['seconds'].sum() / 3600:.1f} worker-hours of FM time; "
            f"redraw reps that needed a redraw: {int((base.attempt > 0).sum() / 4)} "
            f"(of {int((base.strategy == 'redraw').sum() / 4)}).", "",
            "Infeasible redraw reps (no covering draw in 100,000 attempts; Amendment 2): "
            + infeasible_redraws()]


def infeasible_redraws() -> str:
    """Count of attempt = -1 redraw reps per (table, m) from the selection cache."""
    frames = []
    for path in sorted((OUT_DIR / "selections").glob("*.parquet")):
        sel = pd.read_parquet(path, columns=["strategy", "mult", "attempt"])
        bad = sel[(sel.strategy == "redraw") & (sel.attempt < 0)]
        if len(bad):
            frames.append(f"{path.stem} " + ", ".join(
                f"{m}C {n}/20" for m, n in bad.groupby("mult").size().items()))
    return "; ".join(frames) if frames else "none"


def main() -> None:
    """Build the report."""
    runs = a12.load_runs(OUT_DIR)
    keys = pool_keys()
    runs["pool_key"] = runs.dataset.map(keys)
    expected = set(keys)
    missing = {(k, lr) for k in expected for lr in a12.FMS} - set(
        zip(runs.dataset, runs.learner))
    n_classes = runs.groupby("dataset")["n_classes"].first()
    means = a12.cell_means(runs, "macro_f1")
    table = a12.confirmatory(means)
    reading = a12.overall(table)
    lines = [
        "# Study 12: Study 7 Part (a) replicated on T1′ — report",
        "",
        "Pre-registration: prereg/STUDY12_PROTOCOL_T1PRIME.md (frozen R007 before any "
        "run; Addendum 1 at R008 before any outcome). Generated by "
        "`reports/study12_report.py` from `results/study12/`. T1′ = "
        f"{len(n_classes)} tables ({int((n_classes == 2).sum())} binary, "
        f"{int((n_classes > 2).sum())} multiclass, from the stored n_classes; the "
        "prereg prose said 37/15, corrected by its Erratum 1). HELD-OUT not touched.",
        "",
        "## Completed jobs",
        "",
        *jobs_lines(runs),
        "",
        f"Missing (dataset, FM) jobs: {sorted(missing) if missing else 'none'}.",
        "",
        "## Confirmatory family (6 contrasts, Holm; macro-F1)",
        "",
        "Per-dataset values; (i) mean over m ∈ {1,3,5,10}, (ii) over m ∈ {3,5,10}. "
        "(ii-c)'s p is the TOST p (margin 0.01). Bayes: baycomp signed-rank, ROPE 0.01.",
        "",
        *confirmatory_lines(table),
        "",
        f"**Overall (Sec. 3.1 rules):** C1 {reading['c1']}; ranking claim "
        f"**{reading['ranking']}**.",
        "",
        "Magnitude: T1′ mean / DEV mean = "
        + ", ".join(f"{r['id']} {r['ratio_to_dev']:.2f}"
                    for r in table.to_dict("records") if r["id"] != "(ii-c)"),
        "",
        "## Robustness: per budget (macro-F1; unadjusted)",
        "",
        *per_budget_lines(runs, "macro_f1"),
        "",
        "(ii-b) and the `_bal` rows use the post-hoc p/π̂ transform of fm1/fm5 "
        "(prereg Sec. 2.2); they are secondary.",
        "",
        "### Per budget, accuracy",
        "",
        *per_budget_lines(runs, "accuracy"),
        "",
        "### The six contrasts in secondary metrics (mean Δ [95% CI], Wilcoxon p, unadjusted)",
        "",
        *secondary_summary_lines(runs),
        "",
        "## S-1: redraw protocol (pre-declared secondary)",
        "",
        *redraw_lines(a12.redraw_analysis(runs)),
        "",
        "## S-2: AUC with a realistic convention (pre-declared secondary)",
        "",
        *auc_lines(a12.auc_all_analysis(runs)),
        "",
        "## S-3: ranking flip rate (pre-declared secondary)",
        "",
        *flip_lines(runs),
        "",
        "## S-4: binary vs multiclass (pre-declared secondary; unadjusted)",
        "",
        *split_lines(a12.split_analysis(runs, n_classes)),
        "",
        "## Addendum 1 sensitivities (no confirmatory weight)",
        "",
    ]
    inverse = {v: k for k, v in keys.items()}
    lines += sensitivity_lines(
        runs, {inverse[k] for k in COUNTER + DUPLICATES}, "SENS-INT: without the 9 flagged tables")
    lines += sensitivity_lines(
        runs, {inverse[k] for k in COUNTER}, "SENS-ID: without the 4 counter-column tables")
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
