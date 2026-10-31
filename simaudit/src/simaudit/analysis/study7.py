"""Study 7 metrics and statistics (prereg/STUDY7_PROTOCOL_MECHANISM.md).

Metrics are computed on the complement of the labeled rows from a full
probability matrix, so every score can be recomputed from stored outputs. The
statistics cover the pre-registered analysis plan: Wilcoxon with Holm, TOST,
the Bayesian signed-rank test, leave-one-dataset-out, a stratified
Jonckheere-Terpstra trend test, Page's L test, Friedman/Nemenyi ranks and the
protocol rank-reversal summary.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

from simaudit.analysis.report import bootstrap_ci, holm

CLIP = 1e-6
ROPE = 0.01
MARGIN = 0.01
# Studentized range q_{0.05, k} / sqrt(2) for the Nemenyi test (Demsar 2006).
NEMENYI_Q05 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}
METRICS = ("macro_f1", "accuracy", "balanced_accuracy", "log_loss", "auc")


def prior_correct(
    proba: NDArray[np.float64], y_lab: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Divide by the context class shares and renormalise (TabPFN's balance option).

    Unseen classes keep probability 0; rows are renormalised over seen classes.
    """
    n_classes = proba.shape[1]
    share = np.bincount(y_lab, minlength=n_classes) / len(y_lab)
    seen = share > 0
    out = np.zeros_like(proba, dtype=np.float64)
    out[:, seen] = proba[:, seen] / share[seen]
    total = out.sum(axis=1, keepdims=True)
    return np.asarray(out / np.where(total > 0, total, 1.0))


def _log_loss(y_pos: NDArray[np.int64], P: NDArray[np.float64]) -> float:
    """Mean negative log-probability of the true column after clipping."""
    Q = np.clip(P, CLIP, 1.0)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(Q[np.arange(len(y_pos)), y_pos])))


def _auc(
    y_true: NDArray[np.int64], proba: NDArray[np.float64], seen: NDArray[np.int64]
) -> float:
    """Macro one-vs-rest AUC over seen-class columns with both outcomes present."""
    scores = [
        float(roc_auc_score(y_true == c, proba[:, c]))
        for c in seen
        if 0 < int((y_true == c).sum()) < len(y_true)
    ]
    return float(np.mean(scores)) if scores and len(seen) > 1 else float("nan")


def metric_set(
    y_true: NDArray[np.int64],
    proba: NDArray[np.float64],
    cols: NDArray[np.int64],
    seen: NDArray[np.int64],
) -> dict[str, float]:
    """Return the five metrics with predictions restricted to the columns ``cols``.

    ``proba`` has one column per class; every value of ``y_true`` is in ``cols``.
    """
    if len(y_true) == 0:
        return {name: float("nan") for name in METRICS}
    P = proba[:, cols]
    pred = cols[P.argmax(axis=1)]
    present = np.unique(y_true)
    recall = [float(np.mean(pred[y_true == c] == c)) for c in present]
    return dict(
        macro_f1=float(
            f1_score(
                y_true, pred, labels=cols.tolist(), average="macro", zero_division=0
            )
        ),
        accuracy=float(np.mean(pred == y_true)),
        balanced_accuracy=float(np.mean(recall)),
        log_loss=_log_loss(np.searchsorted(cols, y_true).astype(np.int64), P),
        auc=_auc(y_true, proba, seen),
    )


def complement_metrics(
    y: NDArray[np.int64], proba: NDArray[np.float64], labeled: NDArray[np.int64]
) -> dict[str, float]:
    """All metrics on the complement, plus ``seen_*`` on rows of seen classes."""
    rest = np.setdiff1d(np.arange(len(y)), labeled)
    seen = np.unique(y[labeled]).astype(np.int64)
    all_cols = np.arange(proba.shape[1], dtype=np.int64)
    full = metric_set(y[rest], proba[rest], all_cols, seen)
    keep = rest[np.isin(y[rest], seen)]
    seen_only = metric_set(y[keep], proba[keep], seen, seen)
    return {**full, **{f"seen_{k}": v for k, v in seen_only.items()}}


# --------------------------------------------------------------------------- tests
def wilcoxon_p(
    values: NDArray[np.float64], alternative: str = "two-sided", shift: float = 0.0
) -> float:
    """Wilcoxon signed-rank p of ``values + shift``; 1.0 when every value is zero.

    The defaults give the two-sided test of earlier studies; ``shift`` = δ with
    ``alternative`` = "greater" is the one-sided non-inferiority test.
    """
    shifted = np.asarray(values, dtype=np.float64) + shift
    if not np.any(shifted != 0):
        return 1.0
    return float(stats.wilcoxon(shifted, alternative=alternative).pvalue)


def tost_p(values: NDArray[np.float64], margin: float = MARGIN) -> float:
    """Paired TOST p for |mean| < margin (max of the two one-sided t-test p)."""
    n = len(values)
    sd = float(np.std(values, ddof=1))
    if sd == 0:
        return 0.0 if abs(float(np.mean(values))) < margin else 1.0
    se = sd / np.sqrt(n)
    mean = float(np.mean(values))
    p_low = 1 - stats.t.cdf((mean + margin) / se, n - 1)
    p_high = stats.t.cdf((mean - margin) / se, n - 1)
    return float(max(p_low, p_high))


def bayes_signed_rank(
    values: NDArray[np.float64], rope: float = ROPE, seed: int = 0
) -> tuple[float, float, float]:
    """Return P(Δ > rope), P(|Δ| <= rope), P(Δ < -rope) from baycomp."""
    from baycomp import SignedRankTest

    pos, mid, neg = SignedRankTest.probs(
        values, np.zeros_like(values), rope=rope, random_state=seed
    )
    return float(pos), float(mid), float(neg)


def lodo(
    values: pd.Series, test: Callable[[NDArray[np.float64]], float] = wilcoxon_p
) -> dict[str, float]:
    """Leave-one-dataset-out: range of the mean, worst p and sign stability."""
    means, ps = [], []
    for dataset in values.index:
        rest = values.drop(dataset).to_numpy(dtype=float)
        means.append(float(rest.mean()))
        ps.append(test(rest))
    full = float(values.mean())
    stable = all(np.sign(m) == np.sign(full) for m in means)
    return dict(
        lodo_min=min(means),
        lodo_max=max(means),
        lodo_p_max=max(ps),
        lodo_sign_stable=float(stable),
    )


def summarize(values: pd.Series) -> dict[str, float]:
    """Mean, median, win-rate, CI, Wilcoxon, TOST, Bayesian and LODO of Δ."""
    arr = values.to_numpy(dtype=float)
    low, high = bootstrap_ci(arr)
    pos, mid, neg = bayes_signed_rank(arr)
    return dict(
        n=float(len(arr)),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        win_rate=float(np.mean(arr > 0)),
        ci_low=low,
        ci_high=high,
        p_wilcoxon=wilcoxon_p(arr),
        p_tost=tost_p(arr),
        p_bayes_pos=pos,
        p_bayes_rope=mid,
        p_bayes_neg=neg,
        **lodo(values),
    )


def _tie_sums(sizes: NDArray[np.float64]) -> tuple[float, float, float]:
    a = float(np.sum(sizes * (sizes - 1) * (2 * sizes + 5)))
    b = float(np.sum(sizes * (sizes - 1) * (sizes - 2)))
    c = float(np.sum(sizes * (sizes - 1)))
    return a, b, c


def jt_stratum(groups: Sequence[NDArray[np.float64]]) -> tuple[float, float, float]:
    """Jonckheere-Terpstra for a decrease: statistic, null mean, tie-corrected var.

    ``groups`` are ordered by the dose. J counts pairs (earlier, later group)
    with earlier > later, plus half the ties (Hollander & Wolfe 1999, 6.2).
    """
    J = 0.0
    for i, first in enumerate(groups):
        for later in groups[i + 1 :]:
            diff = first[:, None] - later[None, :]
            J += float((diff > 0).sum() + 0.5 * (diff == 0).sum())
    n = np.array([len(g) for g in groups], dtype=float)
    N = float(n.sum())
    mean = (N**2 - float(np.sum(n**2))) / 4
    _, ties = np.unique(np.concatenate(groups), return_counts=True)
    t = ties.astype(float)
    na, nb, nc = _tie_sums(n)
    ta, tb, tc = _tie_sums(t)
    var = (N * (N - 1) * (2 * N + 5) - na - ta) / 72
    var += nb * tb / (36 * N * (N - 1) * (N - 2)) + nc * tc / (8 * N * (N - 1))
    return J, mean, var


def jt_stratified(
    strata: Sequence[Sequence[NDArray[np.float64]]],
) -> dict[str, float]:
    """Sum JT over strata with >= 2 groups; one-sided normal p for a decrease."""
    used = [s for s in strata if len(s) >= 2]
    parts = np.array([jt_stratum(s) for s in used]) if used else np.zeros((0, 3))
    J, mean, var = parts.sum(axis=0) if len(parts) else (0.0, 0.0, 0.0)
    z = (J - mean) / np.sqrt(var) if var > 0 else 0.0
    return dict(J=float(J), z=float(z), p=float(stats.norm.sf(z)), strata=len(used))


def page_test(matrix: NDArray[np.float64]) -> dict[str, float]:
    """Page's L for a decrease across ordered columns; blocks are rows."""
    if len(matrix) < 2:
        return dict(L=float("nan"), p=float("nan"), blocks=0.0)
    res = stats.page_trend_test(-matrix, method="asymptotic")
    return dict(L=float(res.statistic), p=float(res.pvalue), blocks=float(len(matrix)))


def slope(share: NDArray[np.float64], score: NDArray[np.float64]) -> float:
    """OLS slope of score on share (NaN if share is constant)."""
    if np.ptp(share) == 0:
        return float("nan")
    return float(np.polyfit(share, score, 1)[0])


def holm_column(table: pd.DataFrame, column: str = "p") -> pd.DataFrame:
    """Add ``p_holm`` over the whole table (one family)."""
    out = table.copy()
    out["p_holm"] = holm(out[column].tolist())
    return out


def avg_ranks(matrix: pd.DataFrame) -> pd.Series:
    """Friedman average ranks (1 = best, higher score is better) per column."""
    ranks = matrix.rank(axis=1, ascending=False)
    return ranks.mean().sort_values()


def nemenyi_cd(k: int, n: int) -> float:
    """Critical difference at alpha 0.05 for k methods on n datasets."""
    return float(NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6 * n)))


def friedman_p(matrix: pd.DataFrame) -> float:
    """Friedman chi-square p across columns (blocks are rows)."""
    return float(stats.friedmanchisquare(*matrix.to_numpy().T).pvalue)


def rank_reversal(
    means: pd.DataFrame, metric: str, threshold: float = 0.01
) -> dict[str, Any]:
    """Kendall tau per mult and the share of decisive pairwise flips.

    ``means`` has one row per (dataset, mult, protocol, learner) with ``metric``;
    protocols are ``stratified`` and ``random``.
    """
    taus = {}
    wide = means.pivot_table(
        index=["mult", "learner"], columns="protocol", values=metric
    ).reset_index()
    for mult, g in wide.groupby("mult"):
        taus[int(cast(Any, mult))] = float(stats.kendalltau(g.stratified, g.random)[0])
    flips = total = 0
    cells = means.pivot_table(
        index=["dataset", "mult", "learner"], columns="protocol", values=metric
    )
    for _, g in cells.groupby(level=[0, 1]):
        s, r = g.stratified.to_numpy(), g.random.to_numpy()
        ds = s[:, None] - s[None, :]
        dr = r[:, None] - r[None, :]
        upper = np.triu(np.ones_like(ds, dtype=bool), 1)
        decisive = upper & (np.abs(ds) > threshold) & (np.abs(dr) > threshold)
        total += int(decisive.sum())
        flips += int((decisive & (np.sign(ds) != np.sign(dr))).sum())
    return dict(tau=taus, flips=flips, decisive=total)
