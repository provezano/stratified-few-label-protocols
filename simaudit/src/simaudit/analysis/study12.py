"""Study 12 metrics and analysis (prereg/STUDY12_PROTOCOL_T1PRIME.md).

Metrics reuse Study 7's definitions on the evaluation rows E minus L, and add
``auc_all``: macro one-vs-rest AUC over every class present in the evaluation
rows, with the stored matrix (an unseen class has the fixed score 0, so its
AUC is 0.5 and no draw is dropped). The analysis computes the confirmatory
Holm family, the robustness summaries of Study 7 and the four pre-declared
secondary analyses (redraw, auc_all, flip rate, binary vs multiclass).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.metrics import roc_auc_score

from simaudit.analysis.report import bootstrap_ci, holm
from simaudit.analysis.study7 import (
    MARGIN,
    bayes_signed_rank,
    lodo,
    metric_set,
    tost_p,
    wilcoxon_p,
)

FMS = ("fm1", "fm2", "fm3", "fm5")
FM_NAMES = {
    "fm1": "TabPFN v2",
    "fm2": "TabICL",
    "fm3": "Kumo",
    "fm5": "TabPFN v3.5",
    "fm1_bal": "TabPFN v2 (bal)",
    "fm5_bal": "TabPFN v3.5 (bal)",
}
MULTS = (1, 3, 5, 10)
RANK_MULTS = (3, 5, 10)
METRIC_COLUMNS = (
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "log_loss",
    "auc",
    "auc_all",
    "seen_macro_f1",
    "seen_accuracy",
    "seen_balanced_accuracy",
    "seen_log_loss",
    "seen_auc",
)
FLIP_THRESHOLD = 0.01


# --------------------------------------------------------------------- metrics
def auc_all(y_true: NDArray[np.int64], proba: NDArray[np.float64]) -> float:
    """Macro one-vs-rest AUC over every class present (both outcomes) in y_true.

    Columns of classes absent from the context are constant (0), so their AUC
    is 0.5; NaN only if no class has both outcomes.
    """
    scores = [
        float(roc_auc_score(y_true == c, proba[:, c]))
        for c in np.unique(y_true)
        if 0 < int((y_true == c).sum()) < len(y_true)
    ]
    return float(np.mean(scores)) if scores else float("nan")


def panel_metrics(
    y_query: NDArray[np.int64], proba: NDArray[np.float64], seen: NDArray[np.int64]
) -> dict[str, float]:
    """Study 7 metrics (+ ``seen_*``) on the query rows, plus ``auc_all``.

    Equals ``study7.complement_metrics`` when the query rows are the complement.
    """
    all_cols = np.arange(proba.shape[1], dtype=np.int64)
    full = metric_set(y_query, proba, all_cols, seen)
    keep = np.isin(y_query, seen)
    seen_only = metric_set(y_query[keep], proba[keep], seen, seen)
    return {
        **full,
        **{f"seen_{k}": v for k, v in seen_only.items()},
        "auc_all": auc_all(y_query, proba) if len(y_query) else float("nan"),
    }


# --------------------------------------------------------------------- loading
def load_runs(out_dir: Path) -> pd.DataFrame:
    """All fit rows without the heavy columns, with fm1_bal / fm5_bal added.

    The ``*_bal`` learners are the ``pc_*`` metrics of fm1 and fm5 (prereg
    Sec. 2.2: the post-hoc transform equals ``balance_probabilities=True``).
    """
    frames = []
    for path in sorted(out_dir.glob("*_fm*.parquet")):
        if ".partial" in path.name:
            continue
        frame = pd.read_parquet(path)
        frames.append(frame.drop(columns=["proba", "labeled", "query", "seen"]))
    runs = pd.concat(frames, ignore_index=True)
    bal = runs[runs.learner.isin(["fm1", "fm5"])].copy()
    for name in METRIC_COLUMNS:
        if f"pc_{name}" in bal:
            bal[name] = bal[f"pc_{name}"]
    bal["learner"] = bal["learner"] + "_bal"
    return pd.concat([runs, bal], ignore_index=True)


def cell_means(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean over reps per (dataset, learner, strategy, mult)."""
    return (
        runs.groupby(["dataset", "learner", "strategy", "mult"])[metric]
        .mean()
        .reset_index()
    )


def per_dataset(
    means: pd.DataFrame,
    learner: str,
    strategy: str,
    mults: tuple[int, ...],
    metric: str,
) -> pd.Series:
    """Per-dataset mean over ``mults`` of one learner and strategy."""
    sub = means[
        (means.learner == learner)
        & (means.strategy == strategy)
        & (means.mult.isin(mults))
    ]
    return sub.groupby("dataset")[metric].mean()


def delta(
    means: pd.DataFrame,
    a: tuple[str, str],
    b: tuple[str, str],
    mults: tuple[int, ...],
    metric: str,
) -> pd.Series:
    """Per-dataset Δ = (learner, strategy) a − b, complete cases."""
    left = per_dataset(means, a[0], a[1], mults, metric)
    right = per_dataset(means, b[0], b[1], mults, metric)
    return (left - right).dropna()


# --------------------------------------------------------------------- summaries
def summarize(values: pd.Series, test: str = "wilcoxon") -> dict[str, Any]:
    """Study 7 Sec. 4.2 summary; ``test`` = wilcoxon (two-sided) or tost."""
    arr = values.to_numpy(dtype=float)
    low, high = bootstrap_ci(arr)
    pos, mid, neg = bayes_signed_rank(arr)
    fn: Callable[[NDArray[np.float64]], float] = (
        tost_p if test == "tost" else wilcoxon_p
    )
    return dict(
        n=len(arr),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        win_rate=float(np.mean(arr > 0)),
        ci_low=low,
        ci_high=high,
        p_wilcoxon=wilcoxon_p(arr),
        p_tost=tost_p(arr),
        p=fn(arr),
        sd=float(arr.std(ddof=1)),
        p_bayes_pos=pos,
        p_bayes_rope=mid,
        p_bayes_neg=neg,
        **lodo(values, fn),
    )


CONFIRMATORY = (
    ("(i)-fm1", ("fm1", "stratified"), ("fm1", "random"), MULTS, "wilcoxon"),
    ("(i)-fm2", ("fm2", "stratified"), ("fm2", "random"), MULTS, "wilcoxon"),
    ("(i)-fm3", ("fm3", "stratified"), ("fm3", "random"), MULTS, "wilcoxon"),
    ("(i)-fm5", ("fm5", "stratified"), ("fm5", "random"), MULTS, "wilcoxon"),
    ("(ii-a)", ("fm2", "random"), ("fm5", "random"), RANK_MULTS, "wilcoxon"),
    ("(ii-c)", ("fm2", "stratified"), ("fm5", "stratified"), RANK_MULTS, "tost"),
)
# Study 7 DEV estimates (reports/STUDY7_REPORT.md), for magnitude consistency.
DEV_MEANS = {
    "(i)-fm1": 0.059,
    "(i)-fm2": 0.040,
    "(i)-fm3": 0.048,
    "(i)-fm5": 0.082,
    "(ii-a)": 0.046,
    "(ii-c)": -0.003,
}


def verdict(cid: str, row: dict[str, Any]) -> str:
    """Pre-registered decision rule of Sec. 3.1."""
    if cid == "(ii-c)":
        if row["p_holm"] < 0.05:
            return "tie established"
        if row["p_wilcoxon"] < 0.05:
            return "difference"
        return "unresolved"
    if row["p_holm"] < 0.05:
        return "replicated" if row["mean"] > 0 else "reversed"
    return "not replicated"


def confirmatory(means: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """The 6-contrast Holm family with robustness and verdicts."""
    rows = []
    for cid, a, b, mults, test in CONFIRMATORY:
        values = delta(means, a, b, mults, metric)
        rows.append(dict(id=cid, test=test, **summarize(values, test)))
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].tolist())
    table["dev_mean"] = table["id"].map(DEV_MEANS)
    table["dev_in_ci"] = (table.dev_mean >= table.ci_low) & (
        table.dev_mean <= table.ci_high
    )
    table["ratio_to_dev"] = table["mean"] / table["dev_mean"]
    table["verdict"] = [
        verdict(r["id"], r)
        for r in cast(list[dict[str, Any]], table.to_dict("records"))
    ]
    return table


def overall(table: pd.DataFrame) -> dict[str, str]:
    """C1 and ranking-claim readings of Sec. 3.1."""
    v = dict(zip(table["id"], table["verdict"]))
    c1 = sum(v[f"(i)-{fm}"] == "replicated" for fm in FMS)
    if v["(ii-a)"] != "replicated":
        ranking = "not supported"
    elif v["(ii-c)"] == "tie established":
        ranking = "supported"
    else:
        ranking = "half supported"
    return dict(
        c1=f"{'replicates' if c1 >= 3 else 'does not replicate'} ({c1}/4 FMs)",
        ranking=ranking,
    )


def per_budget(
    means: pd.DataFrame, a: tuple[str, str], b: tuple[str, str], metric: str
) -> dict[int, dict[str, float]]:
    """Mean Δ, Wilcoxon p and TOST p per mult (unadjusted)."""
    out = {}
    for m in MULTS:
        values = delta(means, a, b, (m,), metric).to_numpy(dtype=float)
        out[m] = dict(
            mean=float(values.mean()), p=wilcoxon_p(values), p_tost=tost_p(values)
        )
    return out


# --------------------------------------------------------------------- secondary
def redraw_analysis(runs: pd.DataFrame) -> dict[str, Any]:
    """S-1: stratified − redraw per FM and fm2 − fm5 under redraw (m 3-10)."""
    means = cell_means(runs, "macro_f1")
    rows = []
    for fm in FMS:
        values = delta(
            means, (fm, "stratified"), (fm, "redraw"), RANK_MULTS, "macro_f1"
        )
        rows.append(dict(id=f"strat−redraw {fm}", **summarize(values)))
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].tolist())
    rank = summarize(
        delta(means, ("fm2", "redraw"), ("fm5", "redraw"), RANK_MULTS, "macro_f1")
    )
    rnd = runs[(runs.strategy == "random") & runs.learner.isin(FMS)]
    covered = rnd.groupby("mult")["coverage"].apply(lambda c: float((c >= 1).mean()))
    attempts = runs[(runs.strategy == "redraw") & (runs.learner == "fm2")]
    redrawn = attempts.groupby("mult")["attempt"].apply(lambda a: float((a > 0).mean()))
    closed = []
    for fm in FMS:
        for m in MULTS:
            gap = delta(means, (fm, "stratified"), (fm, "random"), (m,), "macro_f1")
            rest = delta(means, (fm, "stratified"), (fm, "redraw"), (m,), "macro_f1")
            closed.append(
                dict(
                    learner=fm,
                    mult=m,
                    gap_random=float(gap.mean()),
                    gap_redraw=float(rest.mean()),
                    closed=float(gap.mean() - rest.mean()),
                )
            )
    check_m1 = {
        fm: summarize(
            delta(means, (fm, "stratified"), (fm, "redraw"), (1,), "macro_f1")
        )
        for fm in FMS
    }
    return dict(
        table=table,
        rank=rank,
        covered=covered,
        redrawn=redrawn,
        closed=pd.DataFrame(closed),
        check_m1=check_m1,
    )


def auc_all_analysis(runs: pd.DataFrame) -> dict[str, Any]:
    """S-2: stratified − random in auc_all at m = 1 (Holm over 4) and over mults."""
    out: dict[str, Any] = {}
    for metric in ("auc_all", "auc"):
        means = cell_means(runs, metric)
        rows = []
        for fm in FMS:
            m1 = delta(means, (fm, "stratified"), (fm, "random"), (1,), metric)
            allm = delta(means, (fm, "stratified"), (fm, "random"), MULTS, metric)
            s1, sa = summarize(m1), summarize(allm)
            rows.append(
                dict(
                    learner=fm,
                    m1_mean=s1["mean"],
                    m1_ci_low=s1["ci_low"],
                    m1_ci_high=s1["ci_high"],
                    m1_p=s1["p"],
                    m1_n=s1["n"],
                    all_mean=sa["mean"],
                    all_ci_low=sa["ci_low"],
                    all_ci_high=sa["ci_high"],
                    all_p=sa["p"],
                    all_n=sa["n"],
                )
            )
        table = pd.DataFrame(rows)
        table["m1_p_holm"] = holm(table["m1_p"].tolist())
        out[metric] = table
    nan_share = (
        runs[runs.learner.isin(FMS)]
        .groupby(["strategy", "mult"])["auc"]
        .apply(lambda a: float(a.isna().mean()))
    )
    out["auc_nan_share"] = nan_share
    return out


def _flips(cells: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Decisive pairs and flips per dataset between columns ``a`` and ``b``."""
    out = []
    for (dataset, _), g in cells.groupby(level=[0, 1]):
        s, r = g[a].to_numpy(), g[b].to_numpy()
        ds, dr = s[:, None] - s[None, :], r[:, None] - r[None, :]
        upper = np.triu(np.ones_like(ds, dtype=bool), 1)
        decisive = upper & (np.abs(ds) > FLIP_THRESHOLD) & (np.abs(dr) > FLIP_THRESHOLD)
        flips = decisive & (np.sign(ds) != np.sign(dr))
        out.append(
            dict(dataset=dataset, decisive=int(decisive.sum()), flips=int(flips.sum()))
        )
    return pd.DataFrame(out).groupby("dataset").sum()


def _cluster_ci(
    num: NDArray[np.float64], den: NDArray[np.float64], seed: int = 0, n: int = 2000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n, len(num)))
    ratios = num[idx].sum(axis=1) / np.maximum(den[idx].sum(axis=1), 1)
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def flip_rate(
    runs: pd.DataFrame, learners: tuple[str, ...], metric: str = "macro_f1"
) -> dict[str, Any]:
    """S-3: protocol flip rate, the random-halves noise baseline, their difference."""
    sub = runs[runs.learner.isin(learners)].copy()
    rnd = sub[sub.strategy == "random"].copy()
    rnd["strategy"] = np.where(rnd.rep < 10, "half_a", "half_b")
    both = pd.concat([sub[sub.strategy.isin(["stratified", "random"])], rnd])
    cells = both.pivot_table(
        index=["dataset", "mult", "learner"], columns="strategy", values=metric
    )
    proto = _flips(cells, "stratified", "random")
    noise = _flips(cells, "half_a", "half_b")
    joined = proto.join(noise, rsuffix="_noise")
    rate = joined.flips.sum() / joined.decisive.sum()
    base = joined.flips_noise.sum() / max(joined.decisive_noise.sum(), 1)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(joined), size=(2000, len(joined)))
    f, d = joined.flips.to_numpy(float), joined.decisive.to_numpy(float)
    fn, dn = joined.flips_noise.to_numpy(float), joined.decisive_noise.to_numpy(float)
    diff = f[idx].sum(1) / np.maximum(d[idx].sum(1), 1) - fn[idx].sum(1) / np.maximum(
        dn[idx].sum(1), 1
    )
    means = cells.groupby(level=["mult", "learner"]).mean()
    taus = {
        int(cast(int, m)): float(stats.kendalltau(g.stratified, g.random)[0])
        for m, g in means.groupby(level=0)
    }
    orders = {
        int(cast(int, m)): dict(
            stratified=list(
                g.stratified.droplevel(0).sort_values(ascending=False).index
            ),
            random=list(g.random.droplevel(0).sort_values(ascending=False).index),
        )
        for m, g in means.groupby(level=0)
    }
    return dict(
        rate=float(rate),
        ci=_cluster_ci(f, d),
        flips=int(f.sum()),
        decisive=int(d.sum()),
        noise_rate=float(base),
        noise_ci=_cluster_ci(fn, dn),
        noise_decisive=int(dn.sum()),
        diff=float(rate - base),
        diff_ci=(float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))),
        tau=taus,
        orders=orders,
    )


def split_analysis(runs: pd.DataFrame, n_classes: pd.Series) -> pd.DataFrame:
    """S-4: confirmatory contrasts within binary and multiclass, Mann-Whitney."""
    means = cell_means(runs, "macro_f1")
    rows = []
    extra = (
        ("(i)-fm1 @1C", ("fm1", "stratified"), ("fm1", "random"), (1,), "wilcoxon"),
        ("(i)-fm2 @1C", ("fm2", "stratified"), ("fm2", "random"), (1,), "wilcoxon"),
        ("(i)-fm3 @1C", ("fm3", "stratified"), ("fm3", "random"), (1,), "wilcoxon"),
        ("(i)-fm5 @1C", ("fm5", "stratified"), ("fm5", "random"), (1,), "wilcoxon"),
    )
    for cid, a, b, mults, _ in CONFIRMATORY + extra:
        values = delta(means, a, b, mults, "macro_f1")
        binary = values[n_classes.reindex(values.index) == 2]
        multi = values[n_classes.reindex(values.index) > 2]
        row: dict[str, Any] = dict(id=cid)
        for name, part in (("binary", binary), ("multi", multi)):
            arr = part.to_numpy(dtype=float)
            low, high = bootstrap_ci(arr)
            row.update(
                {
                    f"{name}_n": len(arr),
                    f"{name}_mean": float(arr.mean()),
                    f"{name}_ci_low": low,
                    f"{name}_ci_high": high,
                    f"{name}_p": wilcoxon_p(arr),
                }
            )
        row["mw_p"] = float(stats.mannwhitneyu(binary, multi).pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["MARGIN", "panel_metrics", "auc_all", "confirmatory"]
