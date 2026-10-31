"""Study 9 analysis (prereg/STUDY9_HOW_MANY_LABELS.md, Sec. 3-7).

Interval-censored saturation points (Turnbull NPMLE, log-normal AFT), the ICC
of their categories across FMs, label-equivalence ratios with censoring-aware
sign tests, exact coverage budgets (multivariate hypergeometric) against their
k-means estimates, and the leave-one-dataset-out stopping-signal analysis.
Nothing is imputed: censored quantities are intervals with grid endpoints.
"""

from collections.abc import Callable, Sequence
from math import inf
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats
from scipy.special import gammaln

from simaudit.analysis.report import holm

EPS = 0.01
N_BOOT = 2000
SEED = 20261021
Interval = tuple[float, float]
FloatArray = NDArray[np.float64]


# =================================================================== curves
def curves(frame: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """Mean over repetitions: one row per (dataset, learner, strategy), m columns."""
    keys = ["dataset", "learner", "strategy"]
    return frame.groupby([*keys, "m"])[metric].mean().unstack("m")


def _points(row: pd.Series) -> tuple[FloatArray, FloatArray]:
    """Feasible grid and curve values of one curve row (NaN columns dropped)."""
    ok = row.notna().to_numpy()
    return row.index.to_numpy(float)[ok], row.to_numpy(float)[ok]


# =================================================================== H1
def saturation_interval(ms: ArrayLike, f: ArrayLike, eps: float = EPS) -> Interval:
    """Interval (lo, hi] holding m*: after hi every measured slope is < eps.

    lo = 0 means left-censored (saturated at the first grid point); hi = inf
    means right-censored (the last measured segment still gains >= eps).
    """
    m, v = np.asarray(ms, float), np.asarray(f, float)
    slopes = np.diff(v) / np.diff(m)
    for k in range(len(m) - 1):
        if np.all(slopes[k:] < eps):
            return (float(m[k - 1]) if k else 0.0, float(m[k]))
    return (float(m[-2]) if len(m) > 1 else 0.0, inf)


def cells(grid: Sequence[float]) -> list[Interval]:
    """Return the Turnbull cells: (0, g1], then (g_i, g_i+1], then (g_last, inf)."""
    edges = [0.0, *map(float, grid), inf]
    return list(zip(edges[:-1], edges[1:]))


def _inside(interval: Interval, cell: Interval) -> bool:
    """Whether a cell lies in an interval; an exact value x picks the cell (., x]."""
    lo, hi = interval
    if lo == hi:
        return cell[1] == hi
    return lo <= cell[0] and cell[1] <= hi


def _membership(intervals: Sequence[Interval], grid: Sequence[float]) -> FloatArray:
    cs = cells(grid)
    return np.array([[_inside(iv, c) for c in cs] for iv in intervals], float)


def _em(A: FloatArray, tol: float = 1e-9, max_iter: int = 5000) -> FloatArray:
    p = np.full(A.shape[1], 1.0 / A.shape[1])
    for _ in range(max_iter):
        w = A * p
        new = (w / w.sum(axis=1, keepdims=True)).mean(axis=0)
        if np.abs(new - p).max() < tol:
            return np.asarray(new)
        p = new
    return np.asarray(p)


def turnbull(intervals: Sequence[Interval], grid: Sequence[float]) -> FloatArray:
    """NPMLE (Turnbull EM) of the cell masses for intervals with grid endpoints."""
    return _em(_membership(intervals, grid))


def cdf_at_grid(masses: FloatArray, grid: Sequence[float]) -> dict[float, float]:
    """F(m) = P(m* <= m) at each grid point."""
    cum = np.cumsum(masses)
    return {float(g): float(cum[i]) for i, g in enumerate(grid)}


def quantile_cell(masses: FloatArray, grid: Sequence[float], q: float) -> Interval:
    """Cell (a, b] in which the q-quantile of the NPMLE lies."""
    i = int(np.searchsorted(np.cumsum(masses), q - 1e-12))
    return cells(grid)[min(i, len(masses) - 1)]


def boot_ci(
    values: Sequence[Any],
    stat: Callable[[list[Any]], float],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile 95% CI of ``stat`` over dataset resamples."""
    rng = np.random.default_rng(seed)
    vals = list(values)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    out = np.array([stat([vals[i] for i in row]) for row in idx])
    out = out[np.isfinite(out)]
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def distribution_summary(
    intervals: Sequence[Interval],
    grid: Sequence[float] = (1, 2, 3, 5, 7, 10, 15, 20, 30),
    n_boot: int = 500,
    seed: int = SEED,
) -> dict[str, Any]:
    """H1-dist: F(m) at grid points, median/quartile cells and shares, with CIs."""
    A = _membership(intervals, grid)
    masses = _em(A)
    cdf = np.cumsum(masses)
    at = {float(g): i for i, g in enumerate(grid)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(A), size=(n_boot, len(A)))
    boot = np.array([np.cumsum(_em(A[row])) for row in idx])
    out: dict[str, Any] = dict(n=len(intervals), cdf=cdf_at_grid(masses, grid))
    for q, name in ((0.25, "q25"), (0.5, "median"), (0.75, "q75")):
        out[name] = quantile_cell(masses, grid, q)
    for m in (5.0, 10.0):
        b = boot[:, at[m]]
        out[f"P(m*<={m:g})"] = (
            float(cdf[at[m]]),
            float(np.quantile(b, 0.025)),
            float(np.quantile(b, 0.975)),
        )
    b = 1 - boot[:, at[20.0]]
    out["P(m*>20)"] = (
        float(1 - cdf[at[20.0]]),
        float(np.quantile(b, 0.025)),
        float(np.quantile(b, 0.975)),
    )
    return out


def category(interval: Interval, grid: Sequence[float]) -> int:
    """Ordinal category of an interval: index of its upper end (top = beyond)."""
    hi = interval[1]
    return len(grid) if hi == inf else int(list(map(float, grid)).index(hi))


def icc_a1(Y: FloatArray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement (Shrout & Fleiss)."""
    n, k = Y.shape
    grand = Y.mean()
    msr = k * ((Y.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    sse = ((Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom > 0 else float("nan")


def icc_decision(Y: FloatArray, threshold: float = 0.75) -> dict[str, Any]:
    """H1a: ICC(2,1) with a dataset-bootstrap CI and the pre-registered decision."""
    rows = list(Y)
    lo, hi = boot_ci(rows, lambda r: icc_a1(np.array(r)))
    verdict = (
        "supported"
        if lo >= threshold
        else "refuted" if hi < threshold else "unresolved"
    )
    return dict(n=len(rows), icc=icc_a1(Y), ci_low=lo, ci_high=hi, decision=verdict)


def _interval_nll(
    params: FloatArray, lo: FloatArray, hi: FloatArray, X: FloatArray
) -> float:
    beta, sigma = params[:-1], np.exp(params[-1])
    mu = X @ beta
    with np.errstate(divide="ignore"):
        zl = np.where(lo > 0, (np.log(np.where(lo > 0, lo, 1)) - mu) / sigma, -np.inf)
        zh = np.where(
            np.isfinite(hi),
            (np.log(np.where(np.isfinite(hi), hi, 1)) - mu) / sigma,
            np.inf,
        )
    prob = stats.norm.cdf(zh) - stats.norm.cdf(zl)
    return float(-np.log(np.clip(prob, 1e-300, None)).sum())


def _hessian(
    fn: Callable[[FloatArray], float], x: FloatArray, h: float = 1e-4
) -> FloatArray:
    k = len(x)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            e_i, e_j = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = (
                fn(x + e_i + e_j)
                - fn(x + e_i - e_j)
                - fn(x - e_i + e_j)
                + fn(x - e_i - e_j)
            ) / (4 * h * h)
    return H


def aft_fit(intervals: Sequence[Interval], covariates: pd.DataFrame) -> pd.DataFrame:
    """Interval-censored log-normal AFT: coefficients, Wald SE, one-sided p (> 0)."""
    lo = np.array([a for a, _ in intervals], float)
    hi = np.array([b for _, b in intervals], float)
    X = np.column_stack([np.ones(len(lo)), covariates.to_numpy(float)])
    start = np.zeros(X.shape[1] + 1)
    finite = np.where(np.isfinite(hi), hi, np.maximum(lo, 1) * 2)
    start[0] = np.log(np.sqrt(np.maximum(lo, 0.5) * finite)).mean()

    def nll(p: FloatArray) -> float:
        return _interval_nll(p, lo, hi, X)

    res = optimize.minimize(
        nll,
        start,
        method="Nelder-Mead",
        options=dict(maxiter=20_000, xatol=1e-8, fatol=1e-10),
    )
    res = optimize.minimize(nll, res.x, method="BFGS")
    cov = np.linalg.pinv(_hessian(nll, res.x))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    names = ["intercept", *covariates.columns, "log_sigma"]
    z = res.x / np.where(se > 0, se, np.nan)
    return pd.DataFrame(
        dict(
            coef=res.x,
            se=se,
            z=z,
            p_greater=stats.norm.sf(z),
            p_two=2 * stats.norm.sf(np.abs(z)),
        ),
        index=names,
    )


def selection_ceiling(
    ms: ArrayLike,
    f_sel: ArrayLike,
    f_rand: ArrayLike,
    eps: float = EPS,
) -> Interval:
    """m_sel*: from which grid point on |f_sel - f_rand| < eps (interval-censored)."""
    m = np.asarray(ms, float)
    close = np.abs(np.asarray(f_sel, float) - np.asarray(f_rand, float)) < eps
    for k in range(len(m)):
        if close[k:].all():
            return (float(m[k - 1]) if k else 0.0, float(m[k]))
    return (float(m[-1]), inf)


# =================================================================== H2
def equivalence_interval(
    ms: ArrayLike, rand: ArrayLike, score: Any, m: float
) -> Interval:
    """Interval for rho = m_eq / m on the monotone random envelope (log-m interp)."""
    grid = np.asarray(ms, float)
    env = np.maximum.accumulate(np.asarray(rand, float))
    if score <= env[0]:
        return (0.0, grid[0] / m)
    if score > env[-1]:
        return (grid[-1] / m, inf)
    i = int(np.argmax(env >= score))
    t = (score - env[i - 1]) / (env[i] - env[i - 1])
    m_eq = float(
        np.exp(np.log(grid[i - 1]) + t * (np.log(grid[i]) - np.log(grid[i - 1])))
    )
    return (m_eq / m, m_eq / m)


def sign_test_greater(successes: int, trials: int) -> float:
    """One-sided exact binomial p for P(success) > 1/2."""
    if trials == 0:
        return 1.0
    return float(stats.binomtest(successes, trials, 0.5, alternative="greater").pvalue)


def ordered_pairs(a: Sequence[Interval], b: Sequence[Interval]) -> tuple[int, int]:
    """(# pairs with a > b decided, # decidable pairs) for interval pairs."""
    greater = decided = 0
    for (alo, ahi), (blo, bhi) in zip(a, b):
        if alo >= bhi and not (alo == ahi == blo == bhi):
            greater += 1
            decided += 1
        elif ahi <= blo and not (alo == ahi == blo == bhi):
            decided += 1
    return greater, decided


def turnbull_median_ratio(intervals: Sequence[Interval]) -> float:
    """Median of rho from the NPMLE: the right end of the median cell (left if inf)."""
    pts = sorted({x for iv in intervals for x in iv if 0 < x < inf})
    if not pts:
        return float("nan")
    a, b = quantile_cell(turnbull(intervals, pts), pts, 0.5)
    return float(b if b < inf else a)


# =================================================================== H3
def _log_comb(n: Any, k: Any) -> Any:
    return (
        gammaln(np.asarray(n) + 1)
        - gammaln(np.asarray(k) + 1)
        - gammaln(np.asarray(n) - np.asarray(k) + 1)
    )


def _log_hyper_pmf(j: NDArray[np.int64], pop: int, size: int, draws: int) -> FloatArray:
    """Return log P(j hits) for ``draws`` draws from ``pop`` with ``size`` hits."""
    return np.asarray(
        _log_comb(size, j) + _log_comb(pop - size, draws - j) - _log_comb(pop, draws)
    )


def p_all_at_least_dp(counts: Sequence[int], budget: int, k: int = 1) -> float:
    """Return P(every class >= k labels), by sequential hypergeometric conditioning.

    dp[r] = P(classes so far all >= k and r draws still to place among the rest).
    """
    sizes = [int(c) for c in counts if c > 0]
    pop = sum(sizes)
    dp = np.zeros(budget + 1)
    dp[budget] = 1.0
    for size in sizes[:-1]:
        new = np.zeros(budget + 1)
        for r in (int(v) for v in np.flatnonzero(dp > 0)):
            js = np.arange(k, min(size, r) + 1, dtype=np.int64)
            js = js[r - js <= pop - size]
            if len(js):
                new[r - js] += dp[r] * np.exp(_log_hyper_pmf(js, pop, size, r))
        dp, pop = new, pop - size
    left = np.arange(budget + 1)
    return float(dp[(left >= k) & (left <= sizes[-1])].sum())


def p_all_at_least(counts: Sequence[int], budget: int, k: int = 1) -> float:
    """P(every class gets >= k of ``budget`` draws without replacement), exact.

    Inclusion-exclusion for k = 1 (C <= 10, so at most 1024 terms), the
    sequential hypergeometric dynamic program otherwise.
    """
    sizes = [int(c) for c in counts if c > 0]
    if k * len(sizes) > budget:
        return 0.0
    if budget >= sum(sizes):
        return 1.0 if min(sizes) >= k else 0.0
    if k == 1:
        return min(1.0, max(0.0, p_cover_inclusion_exclusion(sizes, budget)))
    return p_all_at_least_dp(sizes, budget, k)


def p_cover_inclusion_exclusion(counts: Sequence[int], budget: int) -> float:
    """P(every class drawn at least once), by inclusion-exclusion (test oracle)."""
    sizes = np.asarray(counts, int)
    n, total = len(sizes), int(sizes.sum())
    out = 0.0
    for mask in range(1 << n):
        missing = sum(int(sizes[i]) for i in range(n) if mask >> i & 1)
        if total - missing >= budget:
            sign = -1.0 if bin(mask).count("1") % 2 else 1.0
            out += sign * float(
                np.exp(_log_comb(total - missing, budget) - _log_comb(total, budget))
            )
    return out


def budget_for(counts: Sequence[int], k: int = 1, prob: float = 0.95) -> int:
    """Smallest B with P(every class >= k) >= prob (bisection; P is monotone in B)."""
    total = int(sum(c for c in counts if c > 0))
    lo, hi = k * sum(1 for c in counts if c > 0), total
    if p_all_at_least(counts, hi, k) < prob:
        return total + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if p_all_at_least(counts, mid, k) >= prob:
            hi = mid
        else:
            lo = mid + 1
    return lo


def closed_form_budget(shares: Sequence[float], delta: float = 0.05) -> int:
    """Smallest B with sum_k (1 - p_k)^B <= delta (with replacement)."""
    p = np.asarray([s for s in shares if s > 0], float)
    if len(p) == 1:
        return 1
    b = 1
    while ((1 - p) ** b).sum() > delta:
        b += 1
    return b


def inflate_rarest(counts: Sequence[int], factor: float) -> list[float]:
    """Shares with the rarest class's share multiplied by ``factor`` (renormalised)."""
    p = np.asarray(counts, float) / float(sum(counts))
    i = int(np.argmin(p))
    rest = 1 - p[i]
    q = p.copy()
    q[i] = min(p[i] * factor, 0.999)
    others = np.arange(len(p)) != i
    q[others] = p[others] / rest * (1 - q[i])
    return [float(v) for v in q]


def kmeans_pool_sizes(
    X: FloatArray, n_classes: int, pool: NDArray[np.int64], seed: int
) -> list[int]:
    """k-means (K = C) cluster sizes, counted over the labeling pool."""
    from sklearn.cluster import KMeans

    from simaudit.graph.knn import zscore

    lab = KMeans(n_classes, n_init=10, random_state=seed).fit_predict(zscore(X))
    return np.bincount(lab[pool], minlength=n_classes).astype(int).tolist()


def h3_row(
    counts: Sequence[int], est: Sequence[int], n_classes: int
) -> dict[str, float]:
    """Oracle and k-means budgets (in labels per class) for one dataset."""
    m95 = budget_for(counts, 1) / n_classes
    m95_hat = budget_for(est, 1) / n_classes
    total = sum(counts)
    b_hat = min(int(np.ceil(m95_hat * n_classes)), total)
    shares = [c / total for c in est]
    sens = {
        f"p_cov_at_pmin_x{f}": p_all_at_least(
            counts, min(closed_form_budget(inflate_rarest(counts, f)), total), 1
        )
        for f in (1.5, 2.0)
    }
    return dict(
        m95=m95,
        m95_hat=m95_hat,
        log_err=float(np.log(m95_hat / m95)),
        m95_learn=budget_for(counts, 3) / n_classes,
        m95_learn_hat=budget_for(est, 3) / n_classes,
        m95_closed_hat=closed_form_budget(shares) / n_classes,
        p_cov_at_hat=p_all_at_least(counts, b_hat, 1),
        **sens,
    )


def share_decision(
    hits: int, n: int, point: float = 0.75, lower: float = 0.65
) -> dict[str, Any]:
    """H3a: share within 1.5x with a Clopper-Pearson CI and the decision rule."""
    ci = stats.binomtest(hits, n).proportion_ci(confidence_level=0.95)
    share = hits / n
    verdict = (
        "supported"
        if share >= point and ci.low >= lower
        else "refuted" if ci.high < point else "unresolved"
    )
    return dict(
        n=n, share=share, ci_low=float(ci.low), ci_high=float(ci.high), decision=verdict
    )


def kendall_less(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """Kendall tau_b and its one-sided p for tau < 0."""
    res = stats.kendalltau(x, y, alternative="less")
    return float(res.statistic), float(res.pvalue)


# =================================================================== H4
def _argmax_on_stop(row: pd.Series) -> NDArray[np.int64]:
    pos = np.asarray(row.query_pos)
    P = np.asarray(row.proba, dtype=np.float32).reshape(len(pos), int(row.n_classes))
    stop = pos < int(row.n_stop)
    order = np.argsort(pos[stop])
    return np.asarray(P[stop].argmax(axis=1)[order], dtype=np.int64)


def stop_table(
    primary: pd.DataFrame,
    other: pd.DataFrame | None,
    strategy: str = "random",
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """Rows (dataset, rep, transition) with the next slope and the stop-set signals."""
    sub = primary[primary.strategy == strategy]
    alt = (
        None
        if other is None
        else other[other.strategy == strategy].set_index(["dataset", "rep", "m"])
    )
    rows = []
    for (ds, rep), g in sub.groupby(["dataset", "rep"]):
        g = g.sort_values("m").reset_index(drop=True)
        preds = [_argmax_on_stop(r) for _, r in g.iterrows()]
        for k in range(len(g) - 2):
            m0, m1, m2 = g.m[k], g.m[k + 1], g.m[k + 2]
            row = dict(
                dataset=ds,
                rep=rep,
                transition=f"{m0}-{m1}-{m2}",
                log_m=float(np.log(m1)),
                target=float((g[metric][k + 2] - g[metric][k + 1]) / (m2 - m1)),
                agree=(
                    float(np.mean(preds[k] == preds[k + 1]))
                    if len(preds[k])
                    else np.nan
                ),
            )
            key = (ds, rep, m1)
            if alt is not None and key in alt.index:
                other_row = alt.loc[key]  # type: ignore[index]
                other_pred = _argmax_on_stop(other_row)
                row["fm_disagree"] = float(np.mean(other_pred != preds[k + 1]))
            rows.append(row)
    return pd.DataFrame(rows)


def _design(frame: pd.DataFrame, signal: str | None, levels: list[str]) -> FloatArray:
    cols = [(frame.transition == t).to_numpy(float) for t in levels]
    if signal is not None:
        s = frame[signal].to_numpy(float)
        cols += [s, s * frame.log_m.to_numpy(float)]
    return np.column_stack(cols)


def lodo_residuals(frame: pd.DataFrame, signal: str | None) -> FloatArray:
    """Leave-one-dataset-out OLS residuals (transition fixed effects [+ signal])."""
    levels = sorted(frame.transition.unique())
    X = _design(frame, signal, levels)
    y = frame.target.to_numpy(float)
    res = np.empty(len(y))
    for ds in frame.dataset.unique():
        test = (frame.dataset == ds).to_numpy()
        beta, *_ = np.linalg.lstsq(X[~test], y[~test], rcond=None)
        res[test] = y[test] - X[test] @ beta
    return res


def mae_ratio(
    frame: pd.DataFrame, signal: str, n_boot: int = N_BOOT, seed: int = SEED
) -> dict[str, float]:
    """H4: LODO MAE ratio signal / mean gain, cluster-bootstrap CI and one-sided p."""
    frame = frame.dropna(subset=[signal]).reset_index(drop=True)
    e_sig = np.abs(lodo_residuals(frame, signal))
    e_base = np.abs(lodo_residuals(frame, None))
    per = (
        pd.DataFrame(dict(d=frame.dataset, s=e_sig, b=e_base))
        .groupby("d")[["s", "b"]]
        .sum()
    )
    s, b = per.s.to_numpy(), per.b.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boot = s[idx].sum(axis=1) / b[idx].sum(axis=1)
    return dict(
        n_datasets=float(len(s)),
        ratio=float(s.sum() / b.sum()),
        ci_low=float(np.quantile(boot, 0.025)),
        ci_high=float(np.quantile(boot, 0.975)),
        p=float((boot >= 1).mean()),
    )


# =================================================================== power
def min_detectable_tau(n: int, alpha: float = 0.05 / 9, power: float = 0.8) -> float:
    """Kendall tau detectable one-sided (normal approximation of the null)."""
    sd = np.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    return float((stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) * sd)


def sign_test_power(n: int, p: float, alpha: float = 0.05 / 9) -> float:
    """Power of the one-sided exact sign test at true success probability p."""
    k = int(stats.binom.ppf(1 - alpha, n, 0.5)) + 1
    return float(stats.binom.sf(k - 1, n, p))


# =================================================================== family
def holm_family(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values of the confirmatory family (Sec. 6)."""
    names = list(pvalues)
    return dict(zip(names, holm([pvalues[n] for n in names])))


# =================================================================== pipeline
PRIMARY, REPLICATION = "fm2", "fm1"
SELECTORS = ("kmedoids_nested", "lucos_nested")


def load_results(directory: Any) -> pd.DataFrame:
    """All finished Study 9 job files of one set (selections and partials skipped)."""
    paths = sorted(p for p in directory.glob("*.parquet") if "partial" not in p.name)
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def dataset_traits(
    keys_to_dids: dict[str, int], loader: Callable[[int], Any], base_seed: int
) -> pd.DataFrame:
    """n, d, C, IR and the H3 budgets per dataset (permuted order, pool of Sec. 2)."""
    from simaudit.runner.study9 import layout, row_permutation

    rows = []
    for key, did in keys_to_dids.items():
        data = loader(did)
        perm = row_permutation(len(data.y), did, base_seed)
        X, y = data.X[perm], data.y[perm]
        s, _ = layout(len(y))
        pool = np.arange(s, len(y), dtype=np.int64)
        n_classes = int(y.max()) + 1
        counts = np.bincount(y[pool], minlength=n_classes).tolist()
        est = kmeans_pool_sizes(X, n_classes, pool, base_seed)
        full = np.bincount(y)
        rows.append(
            dict(
                dataset=key,
                n=len(y),
                d=X.shape[1],
                C=n_classes,
                IR=float(full.max() / full.min()),
                **h3_row(counts, est, n_classes),
            )
        )
    return pd.DataFrame(rows).set_index("dataset")


def _xs(curve: pd.DataFrame, learner: str, strategy: str) -> pd.DataFrame:
    """Curves of one learner and strategy, indexed by dataset."""
    return cast(
        pd.DataFrame, curve.xs((learner, strategy), level=("learner", "strategy"))
    )


def _intervals(curve: pd.DataFrame, learner: str, strategy: str) -> pd.Series:
    sub = _xs(curve, learner, strategy)
    return pd.Series(
        {k: saturation_interval(*_points(r)) for k, r in sub.iterrows()}, dtype=object
    )


def h1(curve: pd.DataFrame, traits: pd.DataFrame) -> dict[str, Any]:
    """H1-dist, H1a (ICC) and H1b (AFT) plus the selection ceiling."""
    out: dict[str, Any] = {}
    learners = set(curve.index.get_level_values("learner"))
    strategies = set(curve.index.get_level_values("strategy"))
    for lr in sorted(learners):
        for st in sorted(strategies):
            ivs = _intervals(curve, lr, st)
            out[f"dist|{lr}|{st}"] = distribution_summary(ivs.tolist())
    ivs = _intervals(curve, PRIMARY, "random")
    cov = traits.loc[ivs.index, ["C", "IR", "n", "d"]].astype(float)
    cov = cov.apply(np.log).add_prefix("log_")
    out["aft"] = aft_fit(ivs.tolist(), cov) if len(ivs) > 6 else None
    if REPLICATION in learners:
        out["icc"] = _icc(curve, traits)
    out["ceiling"] = {
        st: distribution_summary(_ceiling(curve, st).tolist())
        for st in SELECTORS
        if st in strategies
    }
    return out


def _icc(curve: pd.DataFrame, traits: pd.DataFrame) -> dict[str, Any]:
    full = traits.index[traits.n >= 60 * traits.C]
    a = _intervals(curve, PRIMARY, "random")
    b = _intervals(curve, REPLICATION, "random")
    keys = [k for k in a.index if k in b.index and k in full]
    grid = (1, 2, 3, 5, 7, 10, 15, 20, 30)
    Y = np.array([[category(a[k], grid), category(b[k], grid)] for k in keys], float)
    return icc_decision(Y) if len(keys) > 2 else dict(n=len(keys))


def _ceiling(curve: pd.DataFrame, strategy: str) -> pd.Series:
    sel = _xs(curve, PRIMARY, strategy)
    rnd = _xs(curve, PRIMARY, "random")
    out = {}
    for key in sel.index.intersection(rnd.index):
        ms, fs = _points(sel.loc[key])
        r = rnd.loc[key].reindex(ms).to_numpy(float)
        out[key] = selection_ceiling(ms, fs, r)
    return pd.Series(out)


def _rho_intervals(sub: pd.DataFrame, rnd: pd.DataFrame) -> dict[float, list[Interval]]:
    """Equivalence intervals per budget m in {1, 2, 3, 5, 7, 10}."""
    rho: dict[float, list[Interval]] = {m: [] for m in (1, 2, 3, 5, 7, 10)}
    for key in sub.index.intersection(rnd.index):
        ms, fr = _points(rnd.loc[key])
        for m, store in rho.items():
            score = sub.loc[key].get(m)
            if m in ms and score is not None and np.isfinite(score):
                store.append(equivalence_interval(ms, fr, score, m))
    return rho


def h2(curve: pd.DataFrame, learner: str = PRIMARY) -> dict[str, Any]:
    """H2a sign tests and medians at 1C, H2b fade tests (1C vs 5C)."""
    rnd = _xs(curve, learner, "random")
    present = set(curve.index.get_level_values("strategy"))
    out: dict[str, Any] = {}
    for st in [s for s in (*SELECTORS, "stratified") if s in present]:
        rho = _rho_intervals(_xs(curve, learner, st), rnd)
        wins = sum(lo > 1 for lo, _ in rho[1])
        row: dict[str, Any] = dict(
            n=len(rho[1]),
            share_gt1=wins / max(len(rho[1]), 1),
            p_sign=sign_test_greater(wins, len(rho[1])),
            medians={m: turnbull_median_ratio(v) for m, v in rho.items() if v},
        )
        if st in SELECTORS:
            g, dcd = (
                ordered_pairs(rho[1], rho[5]) if len(rho[1]) == len(rho[5]) else (0, 0)
            )
            row.update(
                fade_greater=g, fade_decided=dcd, p_fade=sign_test_greater(g, dcd)
            )
        out[st] = row
    return out


def h3(traits: pd.DataFrame) -> dict[str, Any]:
    """H3a share within 1.5x, H3b IR failure mode, and descriptive budgets."""
    within = int((np.abs(traits.log_err) <= np.log(1.5)).sum())
    learn_err = np.log(traits.m95_learn_hat / traits.m95_learn)
    tau, p = kendall_less(traits.log_err, np.log(traits.IR))
    return dict(
        h3a=share_decision(within, len(traits)),
        h3c_learn_share=float((np.abs(learn_err) <= np.log(1.5)).mean()),
        h3b_tau=tau,
        h3b_p=p,
        unsafe_rate=float((traits.log_err < -np.log(1.5)).mean()),
        over_rate=float((traits.log_err > np.log(1.5)).mean()),
    )


def h4(raw: pd.DataFrame) -> dict[str, Any]:
    """H4: LODO MAE ratios of the stop-set signals against the mean gain."""
    primary = raw[raw.learner == PRIMARY]
    other = raw[raw.learner == REPLICATION] if REPLICATION in set(raw.learner) else None
    table = stop_table(primary, other)
    out: dict[str, Any] = dict(n_rows=len(table))
    for signal in ("agree", "fm_disagree"):
        if signal in table and table[signal].notna().any():
            out[signal] = mae_ratio(table, signal)
    return out


def analyze(raw: pd.DataFrame, traits: pd.DataFrame) -> dict[str, Any]:
    """Every pre-registered H1-H4 quantity and the Holm family (Sec. 6)."""
    curve = curves(raw)
    res = dict(h1=h1(curve, traits), h2=h2(curve), h3=h3(traits), h4=h4(raw))
    pvals: dict[str, float] = {}
    if res["h1"]["aft"] is not None:
        pvals["H1b"] = float(res["h1"]["aft"].loc["log_C", "p_greater"])
    for st, row in res["h2"].items():
        pvals[f"H2a|{st}"] = row["p_sign"]
        if "p_fade" in row:
            pvals[f"H2b|{st}"] = row["p_fade"]
    pvals["H3b"] = res["h3"]["h3b_p"]
    for signal in ("agree", "fm_disagree"):
        if signal in res["h4"]:
            pvals[f"H4|{signal}"] = res["h4"][signal]["p"]
    res["family"] = dict(p=pvals, holm=holm_family(pvals))
    return res


# =================================================================== report
def _fmt_iv(iv: Interval) -> str:
    return f"({iv[0]:g}, {iv[1]:g}]"


def _dist_lines(res: dict[str, Any]) -> list[str]:
    lines = [
        "| learner | strategy | n | median | IQR cells | P(m*<=5) | P(m*<=10)"
        " | P(m*>20) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, d in res["h1"].items():
        if not key.startswith("dist|"):
            continue
        _, lr, st = key.split("|")
        shares = [
            f"{d[k][0]:.2f} [{d[k][1]:.2f}, {d[k][2]:.2f}]"
            for k in ("P(m*<=5)", "P(m*<=10)", "P(m*>20)")
        ]
        lines.append(
            f"| {lr} | {st} | {d['n']} | {_fmt_iv(d['median'])} | "
            f"{_fmt_iv(d['q25'])}–{_fmt_iv(d['q75'])} | " + " | ".join(shares) + " |"
        )
    return lines


def report(res: dict[str, Any], title: str) -> str:
    """Markdown summary of ``analyze`` output."""
    lines = [f"# {title}", "", "## H1: saturation point (macro-F1, eps = 0.01)", ""]
    lines += _dist_lines(res)
    aft = res["h1"].get("aft")
    if aft is not None:
        lines += ["", "AFT (interval-censored log-normal), FM2, random:", ""]
        lines.append(aft.round(4).to_markdown())
    if "icc" in res["h1"]:
        lines += ["", f"H1a ICC: {res['h1']['icc']}"]
    for st, d in res["h1"]["ceiling"].items():
        lines.append(f"- selection ceiling {st}: median cell {_fmt_iv(d['median'])}")
    lines += ["", "## H2: label equivalence (FM2)", ""]
    for st, row in res["h2"].items():
        lines.append(f"- {st}: {row}")
    lines += ["", "## H3: coverage budget", "", f"{res['h3']}", ""]
    lines += ["## H4: stopping signals", "", f"{res['h4']}", ""]
    lines += ["## Holm family", "", "| test | p | p_holm |", "|---|---|---|"]
    fam = res["family"]
    lines += [f"| {k} | {fam['p'][k]:.4g} | {fam['holm'][k]:.4g} |" for k in fam["p"]]
    return "\n".join(lines) + "\n"
