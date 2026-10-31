"""Study 10 families: H1 (pooled and per set), H2-H10, Sec. 7.3-7.4 and Sec. 8.

prereg/STUDY10_SELECTOR_CONFIRMATION.md, written 2026-10-30, after the freeze
and before any HELD-OUT or T1' Study 10 output exists, for the joint HELD-OUT
analysis (prereg/JOINT_OPENING_2026-10-30.md). It builds on the frozen-time
module ``analysis/study10.py`` (``ni_test``, ``ni_family``, ``decide``), which
is not changed. Choices that the prereg leaves open are listed in ``READINGS``
and reported with every analysis.

Inputs are the runner's job files ``results/study10/{set}/{key}_{learner}.parquet``.
The secondaries that rescore stored probabilities (the N2 rule, the
common-set sensitivity, the coverage decomposition, the ring case and the
moderators) also need each dataset's labels in the runner's permuted order,
passed in as ``labels``; without them those parts are skipped and say so.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from simaudit.analysis.report import bootstrap_ci, holm
from simaudit.analysis.study7 import (
    avg_ranks,
    bayes_signed_rank,
    friedman_p,
    tost_p,
    wilcoxon_p,
)
from simaudit.analysis.study10 import ALPHA, MARGIN, decide, ni_test

PRIMARY_LEARNER = "fm2"
F1, F0 = "f1_e5", "d22_v3"
H1_SPEC = {
    "H1a": ("lucos_fps", "macro_f1"),
    "H1b": ("lucos_fps", "accuracy"),
    "H1c": ("kmedoids", "macro_f1"),
    "H1d": ("kmedoids", "accuracy"),
}
METRICS = ("macro_f1", "accuracy")
MULTS = (1, 3, 5, 10)
NO_1C = (3, 5, 10)
MEMBERS = ("f1_sub", "f3_e7", "e9_ik", "e9_ikproj")
SECONDARY = (
    "kmeans_central",
    "typiclust",
    "kcenter",
    "probcover",
    "usl",
    "rdss",
    "zcore",
    "random",
)
CD_ARMS = (F1, F0, "lucos_fps", "lucos", "kmedoids", *SECONDARY)
N2_ARMS = ("typiclust", "kcenter")
N2_MAX_N = 160
OPEN_LEARNERS = ("fm1", "fm3", "logreg")
SENSITIVITY_LEARNERS = ("fm5",)
LOST_SHARE = 0.10  # Sec. 3.4
MAX_UNVERIFIED_SHARE = 0.5  # Sec. 3.5, as Study 13 Amendment 2
RHO_GRID = (1, 3, 5, 10)  # H8: random's budgets C, 3C, 5C, 10C
LIGHT_DROP = ["labeled_pos", "labeled", "query_pos", "proba"]
READINGS = (
    "R10-1 complete cells (Sec. 3.4): an (arm, dataset) enters a contrast only "
    "if every (mult, rep) cell of the arm has a finite metric; the "
    "repaired-cell sensitivity (Sec. 4.3) drops repaired rows and relaxes this.",
    "R10-2 per-set condition (Sec. 8): mean_ho and mean_t1 are the means on the "
    "'heldout' and 't1prime' sets; on a single-set dry run both are that set.",
    "R10-3 not evaluable (Sec. 3.4): n < 0.9 x the datasets of the primary set; "
    "such a contrast is never established and never triggers 'refuted'.",
    "R10-4 pairwise common set (Sec. 3.3): rep r of arm a is paired with rep r "
    "of arm b, reps present in both (random keeps reps 0-4 against 5-rep arms).",
    "R10-5 H8 censoring: below random's 1C score rho <= 1/m (B_rand <= C), "
    "above its 10C score rho > 10/m; medians and HL use the bounds (rank-based).",
    "R10-6 P(delta < -0.01) is baycomp's P(-) at ROPE 0.01 (random_state 0).",
    "R10-7 the HO label sensitivity (Sec. 3.5) recomputes H1 with the Holm "
    "family of 4 on the reduced pooled set.",
)
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# --------------------------------------------------------------------- loading
def result_files(directory: Path, learners: Iterable[str] | None = None) -> list[Path]:
    """Finished job files ``{key}_{learner}.parquet`` (partials skipped)."""
    files = sorted(p for p in directory.glob("*.parquet") if "partial" not in p.name)
    if learners is None:
        return files
    wanted = tuple(f"_{lr}.parquet" for lr in learners)
    return [p for p in files if p.name.endswith(wanted)]


def load_runs(
    dirs: Mapping[str, Path], learners: Iterable[str] | None = None
) -> pd.DataFrame:
    """Light rows (no index lists, no probabilities) of every set."""
    chosen = None if learners is None else list(learners)
    frames = [
        pd.read_parquet(path).drop(columns=LIGHT_DROP, errors="ignore")
        for directory in dirs.values()
        for path in result_files(directory, chosen)
    ]
    if not frames:
        raise FileNotFoundError(f"no Study 10 result files in {list(dirs.values())}")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------- scores
def expected_reps(arm: str) -> int:
    """Repetitions per (arm, mult) cell (Sec. 4.1: random 10, the others 5)."""
    return 10 if arm == "random" else 5


def complete_mask(runs: pd.DataFrame, learner: str, metric: str) -> pd.DataFrame:
    """(dataset x arm) True where every cell of the arm has a finite metric."""
    sub = runs[(runs.learner == learner) & runs["mult"].isin(MULTS)]
    ok = sub[sub[metric].notna()]
    counts = ok.groupby(["dataset", "arm"]).size().unstack("arm")
    want = pd.Series({a: expected_reps(a) * len(MULTS) for a in counts.columns})
    return counts.eq(want, axis=1)


def scores(
    runs: pd.DataFrame,
    learner: str,
    metric: str,
    mults: Sequence[int] = MULTS,
    strict: bool = True,
) -> pd.DataFrame:
    """Per-dataset score per arm: mean over reps per mult, then over ``mults``.

    With ``strict`` (R10-1), an arm with any incomplete cell is NaN there.
    """
    sub = runs[(runs.learner == learner) & runs["mult"].isin(list(mults))]
    if sub.empty:
        return pd.DataFrame()
    per_m = sub.groupby(["dataset", "arm", "mult"])[metric].mean()
    table = per_m.groupby(level=["dataset", "arm"]).mean().unstack("arm")
    if strict:
        mask = complete_mask(runs, learner, metric)
        mask = mask.reindex(index=table.index, columns=table.columns)
        table = table.where(mask.fillna(False).astype(bool))
    return table


def delta(table: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Δ_d = a − b per dataset, pairwise complete."""
    if a not in table or b not in table:
        return pd.Series(dtype=float)
    return (table[a] - table[b]).dropna()


# --------------------------------------------------------------------- summaries
def hodges_lehmann(
    values: FloatArray, alpha: float = 0.05
) -> tuple[float, float, float]:
    """HL estimate (median of Walsh averages) and its distribution-free CI."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    i, j = np.triu_indices(n)
    walsh = np.sort((v[i] + v[j]) / 2)
    z = stats.norm.ppf(1 - alpha / 2)
    k = int(np.floor(n * (n + 1) / 4 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = min(max(k, 0), len(walsh) - 1)
    return float(np.median(walsh)), float(walsh[k]), float(walsh[-1 - k])


def sign_p(values: FloatArray) -> float:
    """Exact two-sided sign test p (zeros dropped)."""
    pos, neg = int(np.sum(values > 0)), int(np.sum(values < 0))
    if pos + neg == 0:
        return 1.0
    return float(stats.binomtest(pos, pos + neg).pvalue)


SUMMARY_KEYS = (
    "mean", "median", "ci_low", "ci_high", "hl", "hl_low", "hl_high",
    "win_rate", "lower", "upper", "p_bayes_pos", "p_bayes_rope", "p_worse",
    "lodo_min", "lodo_max",
)  # fmt: skip


def empty_summary() -> dict[str, Any]:
    """A contrast with no dataset: NaN estimates, p = 1."""
    out: dict[str, Any] = {k: float("nan") for k in SUMMARY_KEYS}
    out.update(n=0, p=1.0, p_sign=1.0, p_tost=1.0, p_two_sided=1.0)
    return out


def summary(values: pd.Series, margin: float = MARGIN) -> dict[str, Any]:
    """Every Sec. 7 summary of one contrast; ``p`` is the NI p (Δ + δ > 0)."""
    arr = values.to_numpy(dtype=np.float64)
    if len(arr) < 2:
        return empty_summary()
    ni = ni_test(arr, margin)
    low, high = bootstrap_ci(arr)
    hl, hl_low, hl_high = hodges_lehmann(arr)
    pos, rope, neg = bayes_signed_rank(arr)
    loo = [float(values.drop(k).mean()) for k in values.index]
    return dict(
        n=len(arr),
        mean=ni["mean"],
        median=float(np.median(arr)),
        ci_low=low,
        ci_high=high,
        hl=hl,
        hl_low=hl_low,
        hl_high=hl_high,
        win_rate=float(np.mean(arr > 0)),
        p_sign=sign_p(arr),
        p_tost=tost_p(arr, margin),
        p_two_sided=wilcoxon_p(arr),
        p=ni["p"],
        lower=ni["lower"],
        upper=ni["upper"],
        p_bayes_pos=pos,
        p_bayes_rope=rope,
        p_worse=neg,
        lodo_min=min(loo),
        lodo_max=max(loo),
    )


def ni_table(
    contrasts: Mapping[str, pd.Series], holm_family: bool = True
) -> pd.DataFrame:
    """NI summaries of a family; Holm over it and ``established`` (Sec. 7)."""
    table = pd.DataFrame({k: summary(v) for k, v in contrasts.items()}).T
    table.index.name = "id"
    p = table["p"].astype(float)
    table["p_holm"] = holm(p.tolist()) if holm_family else p
    table["established"] = (table["p_holm"].astype(float) < ALPHA) & (
        table["lower"].astype(float) > -MARGIN
    )
    return table


# --------------------------------------------------------------------- context
@dataclass
class Study10Data:
    """Light runs, the dataset -> set map and the size of the primary set."""

    runs: pd.DataFrame
    n_primary: int
    labels: Mapping[str, IntArray] | None = None

    def sets(self) -> dict[str, list[str]]:
        """Dataset keys per set."""
        pairs = self.runs[["set", "dataset"]].drop_duplicates()
        return {str(s): sorted(g.dataset) for s, g in pairs.groupby("set")}

    def learners(self) -> list[str]:
        """Learners present in the runs."""
        return sorted(set(self.runs.learner))


def _h1_contrasts(
    runs: pd.DataFrame,
    learner: str = PRIMARY_LEARNER,
    mults: Sequence[int] = MULTS,
    strict: bool = True,
) -> dict[str, pd.Series]:
    tables = {m: scores(runs, learner, m, mults, strict) for m in METRICS}
    return {cid: delta(tables[m], F1, comp) for cid, (comp, m) in H1_SPEC.items()}


def h1_family(
    data: Study10Data,
    runs: pd.DataFrame | None = None,
    mults: Sequence[int] = MULTS,
    strict: bool = True,
) -> pd.DataFrame:
    """H1 on the pooled set (Holm over 4), per-set means and NI p, verdicts."""
    runs = data.runs if runs is None else runs
    contrasts = _h1_contrasts(runs, mults=mults, strict=strict)
    table = ni_table(contrasts)
    by_set = data.sets()
    for name, keys in by_set.items():
        for cid, values in contrasts.items():
            part = values[values.index.isin(keys)]
            row = summary(part)
            table.loc[cid, f"mean_{name}"] = row["mean"]
            table.loc[cid, f"n_{name}"] = row["n"]
            table.loc[cid, f"p_ni_{name}"] = row["p"]
            table.loc[cid, f"lower_{name}"] = row["lower"]
    present = list(by_set)
    for col, key in (("mean_ho", "heldout"), ("mean_t1", "t1prime")):
        source = f"mean_{key}" if key in by_set else None
        table[col] = (
            table[source]
            if source
            else table[[f"mean_{s}" for s in present]].astype(float).min(axis=1)
        )
    floor = (1 - LOST_SHARE) * data.n_primary
    table["evaluable"] = table["n"].astype(float) >= floor
    table.loc[~table["evaluable"], "established"] = False
    return table


def sec8_decision(table: pd.DataFrame) -> str:
    """Sec. 8 outcome; not-evaluable contrasts cannot trigger 'refuted' (R10-3)."""
    rows: dict[str, dict[str, Any]] = {}
    for cid in H1_SPEC:
        row = dict(table.loc[cid])
        if not bool(row["evaluable"]):
            row.update(established=False, p_worse=0.0, upper=np.inf, lodo_min=-np.inf)
        rows[cid] = row
    outcome = decide(rows)
    lost = [cid for cid in H1_SPEC if not bool(table.loc[cid, "evaluable"])]
    return f"{outcome} (not evaluable: {', '.join(lost)})" if lost else outcome


def h10_lodo(data: Study10Data) -> pd.DataFrame:
    """H10: leave one dataset out; Holm over 4 recomputed in each left-out set."""
    contrasts = _h1_contrasts(data.runs)
    keys = sorted(set().union(*(set(v.index) for v in contrasts.values())))
    rows = []
    for key in keys:
        ps = [
            ni_test(v.drop(key, errors="ignore").to_numpy(float))["p"]
            for v in contrasts.values()
        ]
        rows.append(dict(zip(contrasts, holm(ps))))
    worst = pd.DataFrame(rows).max() if rows else pd.Series(dtype=float)
    out = []
    for cid, values in contrasts.items():
        means = pd.Series({k: float(values.drop(k).mean()) for k in values.index})
        if means.empty:
            continue
        shift = (means - float(values.mean())).abs()
        top = str(shift.idxmax())
        out.append(
            dict(
                id=cid,
                lodo_min=float(means.min()),
                lodo_max=float(means.max()),
                max_holm_p=float(worst.get(cid, np.nan)),
                stays_above=bool(means.min() > -MARGIN),
                most_influential=top,
                mean_without_it=float(means[top]),
            )
        )
    return pd.DataFrame(out)


def h9_shares(data: Study10Data) -> pd.DataFrame:
    """H9: shares Δ > −δ and Δ > 0 (Clopper-Pearson), median and HL CI."""
    rows = []
    for cid, values in _h1_contrasts(data.runs).items():
        arr = values.to_numpy(float)
        row: dict[str, Any] = dict(id=cid, n=len(arr))
        for name, hit in (("above_margin", arr > -MARGIN), ("positive", arr > 0)):
            k, n = int(hit.sum()), len(arr)
            ci = stats.binomtest(k, n).proportion_ci() if n else None
            row[f"share_{name}"] = k / n if n else np.nan
            row[f"{name}_ci"] = (ci.low, ci.high) if ci else (np.nan, np.nan)
        hl, lo, hi = hodges_lehmann(arr)
        row.update(median=float(np.median(arr)) if len(arr) else np.nan)
        row.update(hl=hl, hl_low=lo, hl_high=hi)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs["robust_to_mean"] = bool((table.hl_low > -MARGIN).all())
    return table


# --------------------------------------------------------------------- H2-H6
def _family(
    runs: pd.DataFrame,
    pairs: Mapping[str, tuple[str, str, str]],
    learner: str = PRIMARY_LEARNER,
) -> pd.DataFrame:
    tables = {m: scores(runs, learner, m) for m in METRICS}
    return ni_table({cid: delta(tables[m], a, b) for cid, (a, b, m) in pairs.items()})


def h2_family(data: Study10Data) -> pd.DataFrame:
    """H2: F1 − LUCoS and F1 − LUCoS-10, both metrics; Holm over 4."""
    pairs = {
        f"F1-{comp}|{m}": (F1, comp, m)
        for comp in ("lucos", "lucos_n10")
        for m in METRICS
    }
    return _family(data.runs, pairs)


def _landmark_datasets(runs: pd.DataFrame, arm: str) -> list[str]:
    sub = runs[(runs.arm == arm) & runs.landmark_active.astype(bool)]
    return sorted(set(sub.dataset))


def h3_landmark(data: Study10Data) -> pd.DataFrame:
    """H3: F1 − F0 equivalence (TOST ±δ, Holm over 2) where s < n for F1.

    Δ_d is averaged over the mults with s < n only; Bayesian P(ROPE) beside.
    """
    runs = data.runs[data.runs.learner == PRIMARY_LEARNER]
    keys = _landmark_datasets(runs, F1)
    active = runs[(runs.arm == F1) & runs.landmark_active.astype(bool)]
    cells = set(zip(active.dataset, active["mult"]))
    rows = []
    for m in METRICS:
        sub = runs[runs.dataset.isin(keys) & runs.arm.isin([F1, F0])]
        sub = sub[[(d, k) in cells for d, k in zip(sub.dataset, sub["mult"])]]
        if sub.empty:
            rows.append(dict(id=f"H3|{m}", n=0, mean=np.nan, p=1.0))
            continue
        per = sub.groupby(["dataset", "arm", "mult"])[m].mean()
        table = per.groupby(level=["dataset", "arm"]).mean().unstack("arm")
        values = delta(table, F1, F0).to_numpy(float)
        enough = len(values) > 1
        pos, rope, neg = bayes_signed_rank(values) if enough else (np.nan,) * 3
        rows.append(
            dict(
                id=f"H3|{m}",
                n=len(values),
                mean=float(values.mean()) if len(values) else np.nan,
                p=tost_p(values) if enough else 1.0,
                p_bayes_rope=rope,
                p_worse=neg,
            )
        )
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].tolist())
    table["equivalent"] = table["p_holm"] < ALPHA
    return table


def h4_members(data: Study10Data) -> dict[str, pd.DataFrame]:
    """H4: per member, X − LUCoS+FPS and X − F1 in both metrics; Holm over 4."""
    out = {}
    for x in MEMBERS:
        pairs = {
            f"{x}-{comp}|{m}": (x, comp, m)
            for comp in ("lucos_fps", F1)
            for m in METRICS
        }
        table = _family(data.runs, pairs)
        table.attrs["keeps_quality"] = bool(table["established"].all())
        out[x] = table
    sub_keys = _landmark_datasets(data.runs, "f1_sub")
    small = data.runs[data.runs.dataset.isin(sub_keys)]
    pairs = {
        f"f1_sub-{comp}|{m}": ("f1_sub", comp, m)
        for comp in ("lucos_fps", F1)
        for m in METRICS
    }
    out["f1_sub|s<n"] = _family(small, pairs) if sub_keys else pd.DataFrame()
    return out


def h5_superiority(data: Study10Data) -> pd.DataFrame:
    """H5: F1 − k-medoids > 0, one-sided Wilcoxon, Holm over 2; others reported."""
    rows = []
    for m in METRICS:
        table = scores(data.runs, PRIMARY_LEARNER, m)
        for comp in ("kmedoids", "lucos_fps", "lucos"):
            values = delta(table, F1, comp)
            arr = values.to_numpy(float)
            low, high = bootstrap_ci(arr) if len(arr) else (np.nan, np.nan)
            rows.append(
                dict(
                    id=f"F1-{comp}|{m}",
                    n=len(arr),
                    mean=float(arr.mean()) if len(arr) else np.nan,
                    ci_low=low,
                    ci_high=high,
                    p_greater=wilcoxon_p(arr, "greater") if len(arr) else 1.0,
                    p_two_sided=wilcoxon_p(arr) if len(arr) else 1.0,
                )
            )
    table = pd.DataFrame(rows)
    fam = table.id.str.startswith("F1-kmedoids|")
    table.loc[fam, "p_holm"] = holm(table.loc[fam, "p_greater"].tolist())
    return table


def h6_profile(data: Study10Data) -> dict[str, pd.DataFrame]:
    """H6: per-mult Δ and NI (Holm over 4 mults per comparator x metric), H6-KM."""
    rows = []
    for m in METRICS:
        for comp in ("lucos_fps", "lucos", "kmedoids"):
            family = {
                mult: delta(scores(data.runs, PRIMARY_LEARNER, m, (mult,)), F1, comp)
                for mult in MULTS
            }
            table = ni_table({str(k): v for k, v in family.items()})
            for mult, row in table.iterrows():
                rows.append(dict(metric=m, comparator=comp, mult=int(str(mult)), **row))
    km = []
    for m in METRICS:
        per = {
            mult: delta(scores(data.runs, PRIMARY_LEARNER, m, (mult,)), F1, "kmedoids")
            for mult in MULTS
        }
        g = (per[1] - pd.concat([per[k] for k in NO_1C], axis=1).mean(axis=1)).dropna()
        arr = g.to_numpy(float)
        km.append(
            dict(
                metric=m,
                n=len(arr),
                mean=float(arr.mean()) if len(arr) else np.nan,
                p=wilcoxon_p(arr, "greater") if len(arr) else 1.0,
            )
        )
    km_table = pd.DataFrame(km)
    km_table["p_holm"] = holm(km_table["p"].tolist())
    return dict(per_mult=pd.DataFrame(rows), h6_km=km_table)


def h7_learners(data: Study10Data) -> dict[str, Any]:
    """H7: H1 contrasts with each open learner (Holm over 12); fm5 outside."""
    present = set(data.learners())
    if not present & {*OPEN_LEARNERS, *SENSITIVITY_LEARNERS}:
        return dict(note="secondary learners not run yet (analysed after the joint)")
    contrasts = {}
    for lr in OPEN_LEARNERS:
        for cid, values in _h1_contrasts(data.runs, lr).items():
            contrasts[f"{lr}|{cid}"] = (
                values if lr in present else pd.Series([], dtype=float)
            )
    family = ni_table(contrasts)
    out: dict[str, Any] = dict(family=family)
    for lr in OPEN_LEARNERS:
        part = family[family.index.str.startswith(f"{lr}|")]
        out[f"generalises_{lr}"] = bool(part["established"].all())
    logreg = family[family.index.str.startswith("logreg|")]
    fm_ok = any(out[f"generalises_{lr}"] for lr in ("fm1", "fm3"))
    means_ok = bool((logreg["mean"].astype(float) > -MARGIN).all())
    out["not_fm_specific"] = out["generalises_logreg"] or (fm_ok and means_ok)
    if "fm5" in present:
        out["fm5"] = ni_table(_h1_contrasts(data.runs, "fm5"), holm_family=False)
    return out


# --------------------------------------------------------------------- H8
def rho_interval(
    random_scores: Sequence[float], score: float, mult: int
) -> tuple[float, str]:
    """ρ = B_rand / (m C) by isotonic interpolation over log B (R10-5)."""
    logb = np.log(np.asarray(RHO_GRID, dtype=float))
    fit = IsotonicRegression().fit(logb, np.asarray(random_scores, dtype=float))
    curve = fit.predict(logb)
    if score <= curve[0]:
        return 1.0 / mult, "<="
    if score > curve[-1]:
        return 10.0 / mult, ">"
    log_b = float(np.interp(score, curve, logb))
    return float(np.exp(log_b)) / mult, "="


def _rho_cell(
    per: pd.Series, keys: Sequence[str], arm: str, mult: int
) -> dict[str, Any]:
    rhos, flags = [], []
    for key in keys:
        rnd = [per.get((key, "random", k)) for k in RHO_GRID]
        score = per.get((key, arm, mult))
        if score is None or any(v is None for v in rnd):
            continue
        rho, flag = rho_interval(
            [float(v) for v in rnd if v is not None], float(score), mult
        )
        rhos.append(rho)
        flags.append(flag)
    arr = np.asarray(rhos, dtype=float)
    low, high = _median_ci(arr)
    return dict(
        n=len(arr),
        median=float(np.median(arr)) if len(arr) else np.nan,
        ci_low=low,
        ci_high=high,
        n_low_censored=flags.count("<="),
        n_high_censored=flags.count(">"),
    )


def h8_labels_saved(data: Study10Data) -> pd.DataFrame:
    """H8: median ρ per learner x arm x mult, bootstrap CI; censored counts."""
    rows = []
    for lr in data.learners():
        sub = data.runs[data.runs.learner == lr]
        per = sub.groupby(["dataset", "arm", "mult"])["macro_f1"].mean()
        keys = sorted(set(sub.dataset))
        for arm in sorted(set(sub.arm) - {"random"}):
            for mult in MULTS:
                cell = _rho_cell(per, keys, arm, mult)
                rows.append(dict(learner=lr, arm=arm, mult=mult, **cell))
    return pd.DataFrame(rows)


def _median_ci(arr: FloatArray, n_boot: int = 2000) -> tuple[float, float]:
    if len(arr) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    meds = np.median(rng.choice(arr, size=(n_boot, len(arr))), axis=1)
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


# --------------------------------------------------------------------- Sec. 7.3
def secondary_comparators(
    data: Study10Data, n2: Mapping[tuple[str, str, str], pd.Series] | None = None
) -> pd.DataFrame:
    """Sec. 7.3: F1 against each secondary comparator, per learner and metric.

    Two-sided Wilcoxon with Holm over the 8 comparators per learner x metric;
    the NI p unadjusted; the Bayesian P(+/ROPE/-).

    ``n2[(learner, comparator, metric)]`` replaces Δ_d on the n <= 160 datasets
    by its pairwise-common-set value (N2 rule, Sec. 3.3).
    """
    rows = []
    for lr in data.learners():
        for m in METRICS:
            table = scores(data.runs, lr, m)
            block = []
            for comp in SECONDARY:
                values = delta(table, F1, comp)
                fix = (n2 or {}).get((lr, comp, m))
                if fix is not None and len(fix):
                    values = values.copy()
                    common = values.index.intersection(fix.index)
                    values.loc[common] = fix.loc[common]
                row = summary(values)
                block.append(dict(learner=lr, metric=m, comparator=comp, **row))
            ps = [b["p_two_sided"] for b in block]
            for b, q in zip(block, holm(ps)):
                b["p_two_sided_holm"] = q
            rows += block
    return pd.DataFrame(rows)


def nemenyi_q(k: int) -> float:
    """Studentized-range critical value / √2 at α = 0.05 (any k)."""
    return float(stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2))


def cd_tables(data: Study10Data) -> pd.DataFrame:
    """Friedman + Nemenyi per learner x metric, and per mult for fm2."""
    rows = []
    specs: list[tuple[str, str, tuple[int, ...]]] = [
        (lr, m, MULTS) for lr in data.learners() for m in METRICS
    ]
    specs += [(PRIMARY_LEARNER, m, (k,)) for m in METRICS for k in MULTS]
    for lr, m, mults in specs:
        table = scores(data.runs, lr, m, mults)
        cols = [a for a in CD_ARMS if a in table]
        matrix = table[cols].dropna()
        if len(matrix) < 2 or len(cols) < 3:
            continue
        k, n = len(cols), len(matrix)
        ranks = avg_ranks(matrix)
        rows.append(
            dict(
                learner=lr,
                metric=m,
                mults=",".join(map(str, mults)),
                n=n,
                friedman_p=friedman_p(matrix),
                cd=float(nemenyi_q(k) * np.sqrt(k * (k + 1) / (6 * n))),
                ranks=ranks.round(3).to_dict(),
            )
        )
    return pd.DataFrame(rows)


def lucos_upgrade(data: Study10Data) -> pd.DataFrame:
    """LUCoS+FPS − LUCoS, both metrics, bootstrap CI (descriptive)."""
    rows = []
    for m in METRICS:
        values = delta(scores(data.runs, PRIMARY_LEARNER, m), "lucos_fps", "lucos")
        arr = values.to_numpy(float)
        low, high = bootstrap_ci(arr) if len(arr) else (np.nan, np.nan)
        rows.append(
            dict(
                metric=m,
                n=len(arr),
                mean=float(np.mean(arr)) if len(arr) else np.nan,
                ci_low=low,
                ci_high=high,
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- sensitivities
def sensitivities(
    data: Study10Data, label_log: pd.DataFrame | None = None
) -> dict[str, Any]:
    """1C excluded (Sec. 8), repaired cells dropped (Sec. 4.3), HO labels (3.5)."""
    no_1c = h1_family(data, mults=NO_1C)
    repaired = data.runs[~data.runs.repaired.astype(bool)]
    out: dict[str, Any] = dict(
        no_1c=no_1c,
        no_1c_lost=[cid for cid in H1_SPEC if not bool(no_1c.loc[cid, "established"])],
        repaired_counts=data.runs[data.runs.repaired.astype(bool)]
        .groupby("arm")
        .size()
        .to_dict(),
        repaired_dropped=h1_family(data, runs=repaired, strict=False),
    )
    if label_log is not None:
        out["labels"] = label_sensitivity(data, label_log)
    return out


def label_sensitivity(data: Study10Data, label_log: pd.DataFrame) -> dict[str, Any]:
    """H1 without merged (and, if at most half, unverified) HO tables (Sec. 3.5)."""
    merged = list(label_log[label_log.merged.astype(str).eq("True")]["key"])
    unverified = list(label_log[label_log.status != "verified"]["key"])
    drop = merged + unverified
    out: dict[str, Any] = dict(merged_tables=merged, unverified_tables=unverified)
    if len(unverified) > MAX_UNVERIFIED_SHARE * len(label_log):
        out["note"] = (
            f"not reported: {len(unverified)} of {len(label_log)} HO tables have "
            "unverified labels, more than half (Sec. 3.5)"
        )
        return out
    kept = data.runs[~data.runs.dataset.isin(drop)]
    out["h1"] = h1_family(data, runs=kept)
    if not drop:
        out["note"] = "no merged or unverified table: equals the primary analysis"
    return out


# --------------------------------------------------------------------- rescoring
def macro_f1_accuracy(
    y: IntArray, pred: IntArray, n_classes: int
) -> tuple[float, float]:
    """Macro-F1 over all C classes (zero_division 0) and accuracy."""
    if len(y) == 0:
        return float("nan"), float("nan")
    cm = np.bincount(y * n_classes + pred, minlength=n_classes**2).reshape(
        n_classes, n_classes
    )
    tp = np.diag(cm).astype(float)
    den = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    return float(f1.mean()), float(tp.sum() / len(y))


def _cells(frame: pd.DataFrame) -> dict[tuple[str, int, int], dict[str, Any]]:
    out = {}
    row: Any
    for row in frame.itertuples(index=False):
        k = int(row.n_classes)
        query = np.asarray(row.query_pos, dtype=np.int64)
        proba = np.asarray(row.proba, dtype=np.float32).reshape(len(query), k)
        out[(str(row.arm), int(row.mult), int(row.rep))] = dict(
            labeled=np.asarray(row.labeled_pos, dtype=np.int64),
            query=query,
            pred=proba.argmax(axis=1).astype(np.int64),
            k=k,
        )
    return out


def _score_on(cell: dict[str, Any], y: IntArray, rows: IntArray) -> tuple[float, float]:
    keep = np.isin(cell["query"], rows)
    q = cell["query"][keep]
    return macro_f1_accuracy(y[q], cell["pred"][keep], cell["k"])


def rescore_dataset(
    frame: pd.DataFrame, y: IntArray, pairs: Sequence[tuple[str, str]]
) -> pd.DataFrame:
    """Pairwise common set E ∖ (L_a ∪ L_b) per paired (mult, rep) cell (R10-4).

    One job file (one dataset, one learner) with its heavy columns. Returns,
    per (a, b, mult, rep) and metric, a's and b's score on the common set,
    and a's score on its own complement (a check of the stored metrics).
    """
    cells = _cells(frame)
    panel_n = min(len(y), 2000)
    rows = []
    for a, b in pairs:
        for (arm, mult, rep), ca in cells.items():
            cb = cells.get((b, mult, rep))
            if arm != a or cb is None:
                continue
            common = np.setdiff1d(
                np.arange(panel_n), np.union1d(ca["labeled"], cb["labeled"])
            )
            fa, aa = _score_on(ca, y, common)
            fb, ab = _score_on(cb, y, common)
            own_f, own_a = _score_on(ca, y, ca["query"])
            rows.append(
                dict(
                    a=a, b=b, mult=mult, rep=rep,
                    common_macro_f1=fa - fb, common_accuracy=aa - ab,
                    own_macro_f1_a=own_f, own_accuracy_a=own_a,
                )
            )  # fmt: skip
    out = pd.DataFrame(rows)
    if len(out):
        out.insert(0, "dataset", str(frame.dataset.iloc[0]))
        out.insert(1, "learner", str(frame.learner.iloc[0]))
        out.insert(2, "n", int(frame.n.iloc[0]))
    return out


def common_pairs() -> list[tuple[str, str]]:
    """Every contrast of H1, H2, H4, H5 and Sec. 7.3 (a, b)."""
    pairs = [(F1, c) for c in ("lucos_fps", "kmedoids", "lucos", "lucos_n10")]
    pairs += [(x, c) for x in MEMBERS for c in ("lucos_fps", F1)]
    pairs += [(F1, c) for c in SECONDARY]
    return list(dict.fromkeys(pairs))


def common_deltas(rescored: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per (learner, a, b, dataset): common-set Δ, mean over reps then mults."""
    col = f"common_{metric}"
    per = rescored.groupby(["learner", "a", "b", "dataset", "mult"])[col].mean()
    return (
        per.groupby(level=["learner", "a", "b", "dataset"])
        .mean()
        .rename("delta")
        .reset_index()
    )


def n2_replacements(rescored: pd.DataFrame) -> dict[tuple[str, str, str], pd.Series]:
    """N2 rule: common-set Δ for TypiClust/k-center contrasts at n <= 160."""
    small = rescored[
        (rescored.n <= N2_MAX_N) & rescored.b.isin(N2_ARMS) & (rescored.a == F1)
    ]
    out = {}
    for m in METRICS:
        table = common_deltas(small, m) if len(small) else pd.DataFrame()
        for (lr, b), g in table.groupby(["learner", "b"]) if len(table) else []:
            out[(str(lr), str(b), m)] = g.set_index("dataset")["delta"]
    return out


def common_set_sensitivity(data: Study10Data, rescored: pd.DataFrame) -> pd.DataFrame:
    """Every contrast on the pairwise common set against own-complement scoring."""
    rows = []
    for m in METRICS:
        common = common_deltas(rescored, m)
        for (lr, a, b), g in common.groupby(["learner", "a", "b"]):
            own = delta(scores(data.runs, str(lr), m), str(a), str(b))
            com = g.set_index("dataset")["delta"].reindex(own.index).dropna()
            rows.append(
                dict(
                    learner=lr, metric=m, a=a, b=b, n=len(com),
                    mean_own=float(own.mean()) if len(own) else np.nan,
                    mean_common=float(com.mean()) if len(com) else np.nan,
                    difference=_mean(com - own.reindex(com.index)),
                )
            )  # fmt: skip
    return pd.DataFrame(rows)


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


# --------------------------------------------------------------------- H6 coverage, 7.4
def coverage_cells(frame: pd.DataFrame, y: IntArray) -> pd.DataFrame:
    """Paired (mult, rep) cells of F1 and k-medoids in one fm2 job file (H6).

    ``differs`` is True when the two arms label different class sets (N1).
    """
    key = str(frame.dataset.iloc[0])
    sub = frame[frame.arm.isin([F1, "kmedoids"])].set_index(["arm", "mult", "rep"])
    rows = []
    for mult, rep in sorted({(m, r) for _, m, r in sub.index}):
        if (F1, mult, rep) not in sub.index or ("kmedoids", mult, rep) not in sub.index:
            continue
        a: Any = sub.loc[(F1, mult, rep)]  # type: ignore[index]
        b: Any = sub.loc[("kmedoids", mult, rep)]  # type: ignore[index]
        seen_a = set(y[np.asarray(a.labeled_pos, dtype=np.int64)].tolist())
        seen_b = set(y[np.asarray(b.labeled_pos, dtype=np.int64)].tolist())
        rows.append(
            dict(
                dataset=key, mult=int(mult), rep=int(rep), differs=seen_a != seen_b,
                macro_f1=float(a.macro_f1 - b.macro_f1),
                accuracy=float(a.accuracy - b.accuracy),
            )
        )  # fmt: skip
    return pd.DataFrame(rows)


def coverage_decomposition(cells: pd.DataFrame) -> pd.DataFrame:
    """N1-style split of F1 − k-medoids into coverage-differs and same cells.

    The other group's cells are set to 0, so the parts add up to the headline
    (mean over m of the mean over datasets of the mean over reps).
    """
    if cells.empty:
        return cells
    out = []
    for m in METRICS:
        total = _headline(cells, m)
        for name, mask in (
            ("coverage differs", cells.differs),
            ("coverage differs, 1C", cells.differs & (cells["mult"] == 1)),
            ("same coverage", ~cells.differs),
        ):
            value = _headline(cells.assign(**{m: cells[m].where(mask, 0.0)}), m)
            share = value / total if total else np.nan
            out.append(dict(metric=m, part=name, value=value, share=share, total=total))
    return pd.DataFrame(out)


def _headline(cells: pd.DataFrame, metric: str) -> float:
    per = cells.groupby(["mult", "dataset"])[metric].mean()
    return float(per.groupby(level="mult").mean().mean())


def ring_cases(frame: pd.DataFrame, y: IntArray) -> pd.DataFrame:
    """Sec. 7.4: B >= 3C cells with >= 90 % of L in a class holding < 60 %."""
    key = str(frame.dataset.iloc[0])
    shares = np.bincount(y) / len(y)
    sub = frame[
        frame.arm.isin(["lucos_fps", "lucos", "kmedoids"]) & (frame["mult"] >= 3)
    ]
    rows = []
    row: Any
    for row in sub.itertuples(index=False):
        lab = np.asarray(row.labeled_pos, dtype=np.int64)
        counts = np.bincount(y[lab], minlength=len(shares))
        top = int(np.argmax(counts))
        if counts[top] >= 0.9 * counts.sum() and shares[top] < 0.6:
            rows.append(
                dict(dataset=key, arm=row.arm, mult=int(row.mult), rep=int(row.rep))
            )
    return pd.DataFrame(rows, columns=["dataset", "arm", "mult", "rep"])


def class_sortedness(y_stored: IntArray) -> float:
    """Share of adjacent equal labels (stored order) over its random-order value."""
    n = len(y_stored)
    observed = float(np.mean(y_stored[1:] == y_stored[:-1]))
    counts = np.bincount(y_stored).astype(float)
    expected = float((counts * (counts - 1)).sum() / (n * (n - 1)))
    return observed / expected if expected > 0 else float("nan")


def ols_hc3(X: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """OLS coefficients and HC3 standard errors (intercept in X)."""
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    lev = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    w = (resid / np.clip(1 - lev, 1e-12, None)) ** 2
    cov = xtx_inv @ (X.T * w) @ X @ xtx_inv
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def moderators(data: Study10Data, traits: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Sec. 7.4: Kendall τ of Δ_d with C, IR, n, d, s<n, sortedness; OLS HC3."""
    taus, fits = [], []
    contrasts = _h1_contrasts(data.runs)
    for cid, values in contrasts.items():
        t = traits.reindex(values.index)
        for col in ("C", "IR", "n", "d", "landmark", "sortedness"):
            ok = t[col].notna()
            if ok.sum() > 2 and t.loc[ok, col].nunique() > 1:
                tau, p = stats.kendalltau(values[ok], t.loc[ok, col])
                taus.append(dict(id=cid, moderator=col, tau=float(tau), p=float(p)))
        X = np.column_stack(
            [np.ones(len(t)), np.log(t.C.astype(float)), np.log(t.IR.astype(float))]
        )
        if len(t) > 3:
            beta, se = ols_hc3(X, values.to_numpy(float))
            for name, b, s in zip(("const", "log_C", "log_IR"), beta, se):
                fits.append(dict(id=cid, term=name, coef=float(b), se_hc3=float(s)))
    return dict(kendall=pd.DataFrame(taus), ols=pd.DataFrame(fits))


def coverage_at_1c(data: Study10Data) -> pd.DataFrame:
    """Sec. 7.4: Δ coverage and Δ seen-class metrics at 1C (F1 − comparator)."""
    rows = []
    for col in ("coverage", "seen_macro_f1", "seen_accuracy"):
        table = scores(data.runs, PRIMARY_LEARNER, col, (1,), strict=False)
        for comp in ("lucos_fps", "kmedoids", "lucos"):
            values = delta(table, F1, comp)
            rows.append(
                dict(
                    column=col,
                    comparator=comp,
                    n=len(values),
                    mean=float(values.mean()) if len(values) else np.nan,
                )
            )
    return pd.DataFrame(rows)


def selection_cost(data: Study10Data) -> pd.DataFrame:
    """Sec. 7.4: median selection CPU seconds per arm and size bin (descriptive)."""
    runs = data.runs[data.runs.learner == PRIMARY_LEARNER].copy()
    runs["size"] = pd.cut(
        runs.n, [0, 1000, 2000, np.inf], labels=["<=1000", "1001-2000", ">2000"]
    )
    return (
        runs.groupby(["arm", "size"], observed=True)["select_cpu_s"]
        .median()
        .unstack("size")
    )


# --------------------------------------------------------------------- report
def analyse(
    data: Study10Data,
    label_log: pd.DataFrame | None = None,
    rescored: pd.DataFrame | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The H1 family and Sec. 8 decision, then every fm2 secondary.

    ``rescored`` (from ``rescore_dataset``) and ``extras`` (coverage
    decomposition, ring cases, traits) need labels; absent, they are skipped.
    """
    h1 = h1_family(data)
    out: dict[str, Any] = dict(
        readings=list(READINGS),
        h1=h1,
        decision=sec8_decision(h1),
        h10=h10_lodo(data),
        h9=h9_shares(data),
        sensitivities=sensitivities(data, label_log),
    )
    out["decision_checks"] = dict(
        c1=bool(h1["established"].all()),
        sets_above=bool(
            (h1[["mean_ho", "mean_t1"]].astype(float).min(axis=1) > -MARGIN).all()
        ),
        h10_above=bool(out["h10"]["stays_above"].all()) if len(out["h10"]) else False,
        p_worse_below=bool((h1["p_worse"].astype(float) < 0.05).all()),
    )
    secondary: dict[str, Any] = dict(
        h2=h2_family,
        h3=h3_landmark,
        h4=h4_members,
        h5=h5_superiority,
        h6=h6_profile,
        h7=h7_learners,
        h8=h8_labels_saved,
        cd=cd_tables,
        lucos_upgrade=lucos_upgrade,
        coverage_1c=coverage_at_1c,
        selection_cost=selection_cost,
    )
    for name, fn in secondary.items():
        out[name] = guarded(fn, data)
    n2 = n2_replacements(rescored) if rescored is not None and len(rescored) else None
    out["sec73"] = guarded(secondary_comparators, data, n2)
    out["n2_applied"] = sorted({f"{k[0]}|{k[1]}" for k in (n2 or {})})
    if rescored is not None and len(rescored):
        out["common_set"] = guarded(common_set_sensitivity, data, rescored)
    else:
        out["common_set"] = "skipped: no labels for rescoring"
    for key, value in (extras or {}).items():
        out[key] = value
    if extras and "traits" in extras:
        out["moderators"] = guarded(moderators, data, extras["traits"])
    return out


def guarded(fn: Any, *args: Any) -> Any:
    """Run a secondary analysis; an error is recorded instead of raised."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - secondaries must not stop the report
        return f"ERROR in {getattr(fn, '__name__', fn)}: {exc!r}"
