"""Pilot report: descriptive tables, tests and the pre-registered decision."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from simaudit.analysis.decision import (
    BIG_GAIN,
    DIAGNOSTICS,
    KILL_SHARE,
    ROPE_KILL_MEAN,
    classify_outcome,
    lodo_rule,
    paired_deltas,
)

FMS = ("FM1", "FM2", "FM3")
HYBRIDS = ("H1", "H2", "H3", "H4")
GRAPHS = ("G1", "G2")
EXT_FMS = ("FM4", "FM5")
EXT_HYBRIDS = ("H5", "H6")


def load_results(out_dir: Path, pattern: str = "*_m*.parquet") -> pd.DataFrame:
    """Concatenate every per-(dataset, m) result file."""
    return pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob(pattern))])


def dataset_means(raw: pd.DataFrame) -> pd.DataFrame:
    """Average repetitions: one row per (dataset, m, arm)."""
    cols = ["macro_f1", "accuracy", "fallback", *DIAGNOSTICS]
    return raw.groupby(["dataset", "m", "arm"], as_index=False)[cols].mean()


def holm(pvalues: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_ci(
    values: np.ndarray, seed: int = 0, n: int = 10_000
) -> tuple[float, float]:
    """Percentile 95% CI of the mean, resampling datasets."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, values.size)).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def per_m_tests(deltas: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank of Δ per m, Holm-adjusted, with bootstrap CIs."""
    rows = []
    for m, group in deltas.groupby("m"):
        values = group.delta.to_numpy()
        p = wilcoxon(values).pvalue if np.any(values != 0) else 1.0
        low, high = bootstrap_ci(values)
        rows.append(dict(m=m, mean_delta=values.mean(), ci_low=low, ci_high=high, p=p))
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table.p.tolist())
    return table


def rule_cells(means: pd.DataFrame, hybrid: str) -> pd.DataFrame:
    """Per (dataset, m): diagnostics, FM1 score and best-hybrid score."""
    wide = means.pivot_table(index=["dataset", "m"], columns="arm", values="macro_f1")
    diag = means[means.arm == "FM1"].set_index(["dataset", "m"])[list(DIAGNOSTICS)]
    cells = diag.assign(f1_fm=wide["FM1"], f1_hybrid=wide[hybrid])
    return cells.reset_index()


def secondary_section(raw: pd.DataFrame, ext_dir: Path) -> list[str]:
    """Amendment 2: the same Δ with the extended FM and hybrid sets.

    Only (dataset, m) cells where every extension arm is present are compared,
    so a partially finished extension never mixes cell sets.
    """
    ext = load_results(ext_dir, "*/*_m*.parquet")
    keys = ["dataset", "m", "rep", "arm"]
    scores = pd.concat([raw[[*keys, "macro_f1"]], ext[[*keys, "macro_f1"]]])
    means = scores.groupby(["dataset", "m", "arm"], as_index=False).macro_f1.mean()
    ext_arms = sorted(set(ext.arm))
    wide = means.pivot_table(index=["dataset", "m"], columns="arm", values="macro_f1")
    complete = wide.dropna(subset=ext_arms).index
    means = means.set_index(["dataset", "m"]).loc[complete].reset_index()
    fms = tuple(a for a in (*FMS, *EXT_FMS) if a in wide)
    hybrids = tuple(a for a in (*HYBRIDS, *EXT_HYBRIDS) if a in wide)
    deltas, hybrid = paired_deltas(means, fms, hybrids)
    share_big = (deltas.groupby("dataset").delta.max() > BIG_GAIN).mean()
    kill = deltas.delta.mean() <= ROPE_KILL_MEAN and share_big < KILL_SHARE
    return [
        "",
        "## Secondary analysis (Amendment 2; cannot change the outcome above)",
        "",
        f"- extension arms: {', '.join(ext_arms)}; cells compared: {len(complete)}",
        f"- FM set {', '.join(fms)}; hybrid set {', '.join(hybrids)}; "
        f"best hybrid: {hybrid}",
        f"- mean Δ (best hybrid − best FM): {deltas.delta.mean():+.4f}",
        f"- share of datasets with Δ > {BIG_GAIN} at some m: {share_big:.2f}",
        f"- KILL criterion met with the extended sets: {'yes' if kill else 'no'}",
        "",
        means.pivot_table(index="arm", columns="m", values="macro_f1")
        .round(4)
        .to_markdown(),
        "",
        "Best FM per dataset (extended set):",
        "",
        deltas.groupby("dataset").best_fm.first().value_counts().to_markdown(),
    ]


def build_report(out_dir: Path, ext_dir: Path | None = None) -> str:
    """Return the markdown report and write it to ``out_dir/REPORT.md``."""
    raw = load_results(out_dir)
    means = dataset_means(raw)
    by_arm = means.pivot_table(index="arm", columns="m", values="macro_f1").round(4)
    deltas, hybrid = paired_deltas(means, FMS, HYBRIDS)
    lodo = lodo_rule(rule_cells(means, hybrid))
    outcome = classify_outcome(deltas, lodo.beats_baselines, lodo.auc)
    best_graph_vs_fm, _ = paired_deltas(means, FMS, GRAPHS)
    share_big = (deltas.groupby("dataset").delta.max() > 0.02).mean()
    lines = [
        "# Pilot I2 report (DEV)",
        "",
        f"**Outcome: {outcome.value}** (rules in prereg/PILOT_I2.md, Sec. 6)",
        "",
        f"- datasets: {means.dataset.nunique()}, best hybrid: {hybrid}",
        f"- mean Δ (best hybrid − best FM): {deltas.delta.mean():+.4f}",
        f"- share of datasets with Δ > 0.02 at some m: {share_big:.2f}",
        f"- LODO rule: AUC {lodo.auc:.3f}, regret rule {lodo.regret_rule:.4f}, "
        f"always-FM {lodo.regret_always_fm:.4f}, always-hybrid "
        f"{lodo.regret_always_hybrid:.4f}",
        f"- H2a (best graph − best FM), mean: {best_graph_vs_fm.delta.mean():+.4f}; "
        f"share of cells where graph wins by > 0.01: "
        f"{(best_graph_vs_fm.delta > 0.01).mean():.2f}",
        "",
        "## Mean macro-F1 by arm and m",
        "",
        by_arm.to_markdown(),
        "",
        "## Δ per m (best hybrid − best FM)",
        "",
        per_m_tests(deltas).round(4).to_markdown(index=False),
        "",
        "## Best FM per dataset",
        "",
        deltas.groupby("dataset").best_fm.first().value_counts().to_markdown(),
    ]
    if ext_dir is not None and any(ext_dir.glob("*/*_m*.parquet")):
        lines += secondary_section(raw, ext_dir)
    text = "\n".join(lines) + "\n"
    (out_dir / "REPORT.md").write_text(text)
    return text
