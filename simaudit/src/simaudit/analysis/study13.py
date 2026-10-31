"""Study 13 analysis (prereg/STUDY13_HELDOUT_CONFIRMATION.md, DRAFT until frozen).

Confirmatory family (Sec. 3-4): Study 12's six contrasts on arm T, with the
same tests, Holm family, TOST margin and ROPE, plus the HELD-OUT readings of
Sec. 4.2 (C1 needs >= 3/4 FMs replicated and none reversed) and an optional
cross-study Holm (Sec. 4.4). Secondary analyses S-1 to S-9 (Sec. 5) and the
integrity sensitivities (Sec. 1). Study 7 and Study 12 functions are reused.

Amendment 1 (2026-10-28, before any HELD-OUT access), all secondary:
SENS-LABEL (tables whose labels were merged dropped), the secondary FMs fm6
and fm7 (S-10 gap, Holm over 2; S-11 FM minus classical, Holm over 2), and
the E17 metric families (S-12 kappa/MCC, Holm over 8; S-13 all-class minus
seen-class AUC, Holm over 4; S-14 CITL on the more imbalanced half, Holm
over 4). The confirmatory family is unchanged.

Amendment 3 (2026-10-29, before any HELD-OUT access): fm8 (LimiX-16M) joins
S-10 and S-11, so each is a Holm family of 3 (fm6, fm7, fm8). An FM with no
table is kept in the family with n = 0 and p = 1, so the family size stays 3.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize, stats

from simaudit.analysis import study12 as a12
from simaudit.analysis.report import bootstrap_ci, holm
from simaudit.analysis.study7 import MARGIN, tost_p, wilcoxon_p

FMS = a12.FMS
EXTRA_FMS = ("fm6", "fm7", "fm8")  # Amendments 1 (B) and 3
CLASSICAL = ("lr", "knn1", "knn3", "gnb", "lda", "hgb")
E13_REFERENCE = ("lr", "knn1", "gnb", "lda", "hgb")  # knn3 excluded as in E13
MULTS = a12.MULTS
RANK_MULTS = a12.RANK_MULTS
E17_COLUMNS = ("kappa", "mcc", "citl")
MAX_UNVERIFIED_SHARE = 0.5  # Amendment 2: SENS-LABEL+ only up to half
FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------- E17 panel
def e17_panel(y_query: NDArray[np.int64], proba: FloatArray) -> dict[str, float]:
    """Kappa, MCC (argmax over all C columns) and CITL of one fit (E17 code).

    CITL = Σ_k |mean_rows p_k − share_k(query)| (L1 calibration-in-the-large).
    Degenerate cases follow E17: kappa 0 when p_e = 1, MCC 0 when undefined.
    """
    n, k = proba.shape
    if n == 0:
        return dict(kappa=float("nan"), mcc=float("nan"), citl=float("nan"))
    pred = proba.argmax(axis=1)
    cmx = np.bincount(y_query * k + pred, minlength=k * k).reshape(k, k)
    cmx = cmx.astype(np.float64)
    true, said = cmx.sum(axis=1), cmx.sum(axis=0)
    acc = float(np.trace(cmx)) / n
    pe = float((true * said).sum()) / n**2
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0
    den = np.sqrt((n**2 - (said**2).sum()) * (n**2 - (true**2).sum()))
    mcc = float(np.trace(cmx) * n - (true * said).sum()) / den if den > 0 else 0.0
    citl = float(np.abs(proba.mean(axis=0) - true / n).sum())
    return dict(kappa=float(kappa), mcc=float(mcc), citl=citl)


# --------------------------------------------------------------------- loading
def load_runs(out_dir: Path) -> pd.DataFrame:
    """All fit rows of one set without heavy columns; fm1_bal / fm5_bal added."""
    frames = []
    for path in sorted(out_dir.glob("*_*.parquet")):
        if ".partial" in path.name or path.parent.name == "selections":
            continue
        frame = pd.read_parquet(path)
        frames.append(
            frame.drop(columns=["proba", "labeled", "query", "seen"], errors="ignore")
        )
    runs = pd.concat(frames, ignore_index=True)
    bal = runs[runs.learner.isin(["fm1", "fm5"])].copy()
    for name in (*a12.METRIC_COLUMNS, *E17_COLUMNS):
        if f"pc_{name}" in bal:
            bal[name] = bal[f"pc_{name}"]
    bal["learner"] = bal["learner"] + "_bal"
    return pd.concat([runs, bal], ignore_index=True)


def arm(runs: pd.DataFrame, name: str) -> pd.DataFrame:
    """Rows of arm ``T`` or ``F``."""
    return runs[runs.arm == name]


def drop_tables(runs: pd.DataFrame, tables: Sequence[str]) -> pd.DataFrame:
    """Runs without the given tables (SENS-INT, SENS-ID; prereg Sec. 1)."""
    return runs[~runs.dataset.isin(list(tables))]


# --------------------------------------------------------------------- primary
def confirmatory(runs: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """Study 12's six-contrast Holm family on arm T, with verdicts."""
    return a12.confirmatory(a12.cell_means(arm(runs, "T"), metric), metric)


def overall(table: pd.DataFrame) -> dict[str, str]:
    """C1 (>= 3/4 replicated and none reversed) and the ranking claim."""
    v = dict(zip(table["id"], table["verdict"]))
    fm = [v[f"(i)-{f}"] for f in FMS]
    replicated = sum(x == "replicated" for x in fm)
    reversed_ = sum(x == "reversed" for x in fm)
    c1 = replicated >= 3 and reversed_ == 0
    if v["(ii-a)"] != "replicated":
        ranking = "not supported"
    elif v["(ii-c)"] == "tie established":
        ranking = "supported"
    else:
        ranking = "half supported"
    return dict(
        c1=f"{'replicates' if c1 else 'does not replicate'} "
        f"({replicated}/4 replicated, {reversed_} reversed)",
        ranking=ranking,
    )


def attenuation(table: pd.DataFrame) -> pd.Series:
    """'replicated, attenuated' when the HO/DEV ratio is below 0.5 (Sec. 4.2)."""
    out = table["verdict"].copy()
    weak = (out == "replicated") & (table["ratio_to_dev"] < 0.5)
    out[weak] = "replicated, attenuated"
    return out


def cross_study_holm(families: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Holm over the union of several studies' confirmatory p-values (Sec. 4.4)."""
    flat = [(name, i, p) for name, ps in families.items() for i, p in enumerate(ps)]
    adjusted = holm([p for _, _, p in flat])
    out: dict[str, list[float]] = {
        name: [0.0] * len(ps) for name, ps in families.items()
    }
    for (name, i, _), q in zip(flat, adjusted):
        out[name][i] = q
    return out


# --------------------------------------------------------------------- helpers
def _gap(
    means: pd.DataFrame, learner: str, mults: tuple[int, ...], metric: str
) -> pd.Series:
    return a12.delta(means, (learner, "stratified"), (learner, "random"), mults, metric)


def _row(values: pd.Series, **extra: Any) -> dict[str, Any]:
    arr = values.to_numpy(dtype=float)
    low, high = bootstrap_ci(arr)
    return dict(
        extra,
        n=len(arr),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        ci_low=low,
        ci_high=high,
        win_rate=float(np.mean(arr > 0)),
        p=wilcoxon_p(arr),
        p_tost=tost_p(arr),
    )


def _row_or_empty(values: pd.Series, **extra: Any) -> dict[str, Any]:
    """``_row``, or a not-evaluable row (n = 0, p = 1) when no table is left."""
    if len(values):
        return _row(values, **extra)
    nan = float("nan")
    return dict(
        extra, n=0, mean=nan, median=nan, ci_low=nan, ci_high=nan,
        win_rate=nan, p=1.0, p_tost=1.0,
    )  # fmt: skip


def _family(rows: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].tolist())
    return table


# --------------------------------------------------------------------- S-1, S-7
def classical_gaps(
    runs: pd.DataFrame, arm_name: str = "T", metric: str = "macro_f1"
) -> pd.DataFrame:
    """S-1: stratified − random per classical learner, mean over m, Holm over 6."""
    means = a12.cell_means(arm(runs, arm_name), metric)
    return _family(
        [_row(_gap(means, lr, MULTS, metric), learner=lr) for lr in CLASSICAL]
    )


def learner_agnostic(table: pd.DataFrame) -> bool:
    """S-1 reading: >= 5 of 6 classical learners Holm-significant with Δ > 0."""
    return int(((table.p_holm < 0.05) & (table["mean"] > 0)).sum()) >= 5


def fm_minus_classical(
    runs: pd.DataFrame,
    metric: str = "macro_f1",
    learners: Sequence[str] = FMS,
) -> pd.DataFrame:
    """S-7 (S-11 for fm6-fm8): D = Δ_FM − mean classical Δ (E13 set), Holm."""
    means = a12.cell_means(arm(runs, "T"), metric)
    ref = pd.concat(
        [_gap(means, lr, MULTS, metric) for lr in E13_REFERENCE], axis=1
    ).mean(axis=1)
    rows = []
    for fm in learners:
        d = (_gap(means, fm, MULTS, metric) - ref).dropna()
        rows.append(_row_or_empty(d, learner=fm))
    return _family(rows)


# --------------------------------------------------------------------- S-3
def total_variation(counts: Sequence[int]) -> float:
    """TV = ½‖π − u‖₁ of a class-count vector."""
    pi = np.asarray(counts, dtype=float) / float(np.sum(counts))
    return float(0.5 * np.abs(pi - 1.0 / len(pi)).sum())


def imbalance(runs: pd.DataFrame, tables: pd.DataFrame) -> dict[str, Any]:
    """S-3: Spearman ρ(ΔA, TV) (one-sided ρ < 0 for the FM mean), OLS slopes.

    ``tables`` has columns key and class_counts (';'-joined), as written by the
    runner's integrity screen.
    """
    counts = tables.set_index("key")["class_counts"].map(
        lambda s: [int(c) for c in str(s).split(";")]
    )
    tv = counts.map(total_variation)
    pmin = counts.map(lambda c: min(c) / sum(c))
    means = a12.cell_means(arm(runs, "T"), "accuracy")
    gaps = pd.concat({fm: _gap(means, fm, MULTS, "accuracy") for fm in FMS}, axis=1)
    gaps["mean"] = gaps[list(FMS)].mean(axis=1)
    rows = []
    for col in gaps:
        g = gaps[col].dropna()
        x = tv.reindex(g.index).to_numpy(float)
        rho, p_two = stats.spearmanr(g.to_numpy(float), x)
        fit = stats.linregress(x, g.to_numpy(float))
        rows.append(
            dict(
                learner=col,
                n=len(g),
                rho=float(rho),
                p_one_sided=float(p_two / 2 if rho < 0 else 1 - p_two / 2),
                slope=float(fit.slope),
                slope_ci=_pairs_ci(x, g.to_numpy(float)),
                rho_pmin=float(
                    stats.spearmanr(g.to_numpy(float), pmin.reindex(g.index))[0]
                ),
                spread=(float(g.min()), float(g.max())),
            )
        )
    return dict(table=pd.DataFrame(rows), tv=tv)


def _pairs_ci(
    x: FloatArray, y: FloatArray, n: int = 5000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    slopes = []
    for row in idx:
        xs = x[row]
        if np.ptp(xs) > 0:
            slopes.append(np.polyfit(xs, y[row], 1)[0])
    return float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))


# --------------------------------------------------------------------- S-4
def fixed_split(runs: pd.DataFrame, metric: str = "macro_f1") -> dict[str, Any]:
    """S-4: (i) per FM on arm F (Holm over 4), (ii-a)/(ii-c), T − F difference."""
    mf = a12.cell_means(arm(runs, "F"), metric)
    mt = a12.cell_means(arm(runs, "T"), metric)
    gaps = _family([_row(_gap(mf, fm, MULTS, metric), learner=fm) for fm in FMS])
    ii_a = _row(a12.delta(mf, ("fm2", "random"), ("fm5", "random"), RANK_MULTS, metric))
    ii_c = _row(
        a12.delta(mf, ("fm2", "stratified"), ("fm5", "stratified"), RANK_MULTS, metric)
    )
    diff = [
        _row(
            (_gap(mt, fm, MULTS, metric) - _gap(mf, fm, MULTS, metric)).dropna(),
            learner=fm,
        )
        for fm in FMS
    ]
    classical = pd.DataFrame(
        [_row(_gap(mf, lr, MULTS, metric), learner=lr) for lr in CLASSICAL]
    )
    return dict(
        gaps=gaps,
        ii_a=ii_a,
        ii_c=ii_c,
        t_minus_f=pd.DataFrame(diff),
        classical=classical,
    )


# --------------------------------------------------------------------- S-6
def reml(effects: FloatArray, variances: FloatArray) -> dict[str, Any]:
    """REML random-effects estimate: μ, its CI, τ, I² and the 95% PI (t, k−2)."""
    y = np.asarray(effects, float)
    v = np.maximum(np.asarray(variances, float), 1e-12)
    k = len(y)

    def neg_restricted_loglik(tau2: float) -> float:
        w = 1.0 / (v + tau2)
        mu = float((w * y).sum() / w.sum())
        return 0.5 * float(
            np.log(v + tau2).sum() + np.log(w.sum()) + (w * (y - mu) ** 2).sum()
        )

    upper = max(float(np.var(y)) * 10, 1e-8)
    tau2 = float(
        optimize.minimize_scalar(
            neg_restricted_loglik, bounds=(0, upper), method="bounded"
        ).x
    )
    w = 1.0 / (v + tau2)
    mu = float((w * y).sum() / w.sum())
    se = float(np.sqrt(1.0 / w.sum()))
    w0 = 1.0 / v
    mu0 = (w0 * y).sum() / w0.sum()
    q = float((w0 * (y - mu0) ** 2).sum())
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    t = stats.t.ppf(0.975, max(k - 2, 1))
    half = t * np.sqrt(tau2 + se**2)
    return dict(
        k=k,
        mu=mu,
        ci=(mu - 1.96 * se, mu + 1.96 * se),
        tau=float(np.sqrt(tau2)),
        i2=float(i2),
        pi=(float(mu - half), float(mu + half)),
        share_pos=float(np.mean(y > 0)),
    )


def table_gaps(
    runs: pd.DataFrame, learner: str, metric: str, mults: tuple[int, ...]
) -> pd.DataFrame:
    """Per-table gap (mean over ``mults``) and its sampling variance (Sec. S-6).

    Var of a per-m gap = var(stratified reps)/n_S + var(random reps)/n_R; the
    per-m gaps come from independent draws, so the mean over m has variance
    Σ v_m / |mults|².
    """
    sub = arm(runs, "T")
    sub = sub[
        (sub.learner == learner)
        & sub.mult.isin(mults)
        & sub.strategy.isin(["stratified", "random"])
    ]
    stat = sub.groupby(["dataset", "mult", "strategy"])[metric].agg(
        ["mean", "var", "count"]
    )
    stat["v"] = stat["var"] / stat["count"]
    wide = stat.unstack("strategy")
    gap = wide[("mean", "stratified")] - wide[("mean", "random")]
    var = wide[("v", "stratified")] + wide[("v", "random")]
    per = pd.DataFrame(dict(gap=gap, var=var)).reset_index()
    out = per.groupby("dataset").agg(gap=("gap", "mean"), var=("var", "sum"))
    out["var"] = out["var"] / len(mults) ** 2
    return out.dropna()


def meta_analysis(
    runs: pd.DataFrame, metrics: Sequence[str] = ("macro_f1", "accuracy")
) -> pd.DataFrame:
    """S-6: REML per FM × m (and the mean over m) for each metric."""
    rows = []
    for metric in metrics:
        for fm in FMS:
            for label, mults in [*((str(m), (m,)) for m in MULTS), ("all", MULTS)]:
                g = table_gaps(runs, fm, metric, mults)
                res = reml(g["gap"].to_numpy(), g["var"].to_numpy())
                rows.append(dict(metric=metric, learner=fm, m=label, **res))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- Amendment 1
def new_fm_gaps(runs: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """S-10: stratified − random per secondary FM, mean over m, Holm over 3."""
    means = a12.cell_means(arm(runs, "T"), metric)
    return _family(
        [_row_or_empty(_gap(means, fm, MULTS, metric), learner=fm) for fm in EXTRA_FMS]
    )


def kappa_mcc(runs: pd.DataFrame) -> pd.DataFrame:
    """S-12: stratified − random in kappa and MCC per FM, mean over m; Holm over 8."""
    rows = []
    for metric in ("kappa", "mcc"):
        means = a12.cell_means(arm(runs, "T"), metric)
        rows += [
            _row(_gap(means, fm, MULTS, metric), learner=fm, metric=metric)
            for fm in FMS
        ]
    return _family(rows)


def auc_all_minus_seen(runs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """S-13: Δ auc_all − Δ auc (seen classes) per FM at m = 1; Holm over 4.

    Per table: the stratified − random gap under the all-class convention
    minus the same gap under the seen-class convention (Study 7; table means
    over the draws where it is defined), complete cases. The mean over m is
    reported unadjusted.
    """
    t_runs = arm(runs, "T")
    every, seen = a12.cell_means(t_runs, "auc_all"), a12.cell_means(t_runs, "auc")

    def diff(fm: str, mults: tuple[int, ...]) -> pd.Series:
        a = _gap(every, fm, mults, "auc_all")
        b = _gap(seen, fm, mults, "auc")
        return (a - b).dropna()

    table = _family([_row(diff(fm, (1,)), learner=fm, m="1") for fm in FMS])
    over_m = pd.DataFrame([_row(diff(fm, MULTS), learner=fm, m="mean") for fm in FMS])
    return dict(table=table, mean_over_m=over_m)


def imbalance_halves(tables: pd.DataFrame) -> tuple[pd.Series, float]:
    """Pool TV per table (from the run's class counts) and its median."""
    counts = tables.set_index("key")["class_counts"].map(
        lambda s: [int(c) for c in str(s).split(";")]
    )
    tv = counts.map(total_variation).astype(float)
    return tv, float(tv.median())


def citl_imbalanced(runs: pd.DataFrame, tables: pd.DataFrame) -> dict[str, Any]:
    """S-14: CITL(stratified) − CITL(random), m ∈ {3, 5, 10}, imbalanced half.

    The more imbalanced half is TV > median TV over the analysed tables (ties
    go to the other half), with TV from the class counts the run recorded at
    load time. Holm over 4 FMs; the other half is reported unadjusted.
    """
    tv, median = imbalance_halves(tables)
    high = set(tv[tv > median].index)
    means = a12.cell_means(arm(runs, "T"), "citl")
    rows, other = [], []
    for fm in FMS:
        gap = _gap(means, fm, RANK_MULTS, "citl")
        rows.append(_row(gap[gap.index.isin(high)], learner=fm, half="imbalanced"))
        other.append(_row(gap[~gap.index.isin(high)], learner=fm, half="balanced"))
    return dict(
        table=_family(rows),
        other_half=pd.DataFrame(other),
        median_tv=median,
        imbalanced=sorted(high),
    )


def amendment_families(runs: pd.DataFrame, tables: pd.DataFrame) -> dict[str, Any]:
    """S-10 to S-14 (Amendment 1), each computed only if its inputs exist."""
    learners = set(runs.learner)
    out: dict[str, Any] = {}
    if set(EXTRA_FMS) & learners:
        out["s10_new_fms"] = new_fm_gaps(runs)
        out["s11_new_fms_minus_classical"] = fm_minus_classical(
            runs, learners=EXTRA_FMS
        )
    if set(E17_COLUMNS) <= set(runs.columns):
        out["s12_kappa_mcc"] = kappa_mcc(runs)
        out["s14_citl_imbalanced"] = citl_imbalanced(runs, tables)
    out["s13_auc_all_minus_seen"] = auc_all_minus_seen(runs)
    return out


def label_sensitivity(runs: pd.DataFrame, label_log: pd.DataFrame) -> dict[str, Any]:
    """SENS-LABEL: the confirmatory family without tables whose labels merged.

    Tables whose raw labels could not be verified are listed. If any exist,
    SENS-LABEL+ also drops them, but only when they number at most half of
    the tables in the label log (Amendment 2); otherwise it is not reported
    and ``sens_label_plus_note`` says why.
    """
    merged = list(label_log[label_log.merged.astype(str).eq("True")]["key"])
    unverified = list(label_log[label_log.status != "verified"]["key"])
    out: dict[str, Any] = dict(
        merged_tables=merged,
        unverified_tables=unverified,
        sens_label=confirmatory(drop_tables(runs, merged)),
    )
    if not unverified:
        return out
    if len(unverified) <= MAX_UNVERIFIED_SHARE * len(label_log):
        out["sens_label_plus"] = confirmatory(drop_tables(runs, merged + unverified))
    else:
        out["sens_label_plus_note"] = (
            f"SENS-LABEL+ not reported: {len(unverified)} of {len(label_log)} "
            "tables have unverified labels, more than half (Amendment 2)"
        )
    return out


# --------------------------------------------------------------------- report
def analyse(
    runs: pd.DataFrame,
    tables: pd.DataFrame,
    screen: pd.DataFrame | None = None,
    label_log: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Every pre-declared analysis of one set (no file output)."""
    t_runs = arm(runs, "T")
    fm_runs = t_runs[t_runs.learner.isin([*FMS, "fm1_bal", "fm5_bal"])]
    table = confirmatory(runs)
    table["reading"] = attenuation(table)
    n_classes = tables.set_index("key")["C"].astype(int)
    out: dict[str, Any] = dict(
        confirmatory=table,
        overall=overall(table),
        s1_classical=classical_gaps(runs),
        s2_auc_all=a12.auc_all_analysis(fm_runs),
        s3_imbalance=imbalance(runs, tables),
        s4_fixed_split=fixed_split(runs),
        s5_split=a12.split_analysis(fm_runs, n_classes),
        s6_meta=meta_analysis(runs),
        s7_fm_minus_classical=fm_minus_classical(runs),
        s8_redraw=a12.redraw_analysis(fm_runs),
        s9_flips=a12.flip_rate(fm_runs, (*FMS, "fm1_bal", "fm5_bal")),
        **amendment_families(runs, tables),
    )
    if label_log is not None:
        out["labels"] = label_sensitivity(runs, label_log)
        out["fragile_label"] = [
            cid
            for cid, a, b in zip(
                table.id, table.verdict, out["labels"]["sens_label"].verdict
            )
            if a != b
        ]
    if screen is not None:
        flagged = screen[screen.flagged.astype(str).eq("True")]["key"]
        id_tables = screen[screen.flag_id.astype(str).eq("True")]["key"]
        out["sens_int"] = confirmatory(drop_tables(runs, list(flagged)))
        out["sens_id"] = confirmatory(drop_tables(runs, list(id_tables)))
        out["fragile"] = [
            cid
            for cid, a, b in zip(table.id, table.verdict, out["sens_int"].verdict)
            if a != b
        ]
    return out


__all__ = ["MARGIN", "analyse", "confirmatory", "cross_study_holm", "overall", "reml"]
