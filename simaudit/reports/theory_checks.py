"""Numerical checks for reports/THEORY_PROTOCOL_BIAS.md (protocol bias, P1).

Five blocks, all written to reports/THEORY_CHECKS.md:

1. Closed forms of Proposition A and Lemma B1 (miss probability, expected
   coverage, P(full coverage) by inclusion-exclusion and by generating function,
   moments of the context shares) against Monte Carlo, on synthetic class
   distributions and on the DEV class counts.
2. Coverage budgets of Corollary 2 (exact, union, with replacement, Chernoff),
   with a Monte Carlo check of the exact budget.
3. Proposition B for the Bayes plug-in learner: exact binary Gaussian metrics,
   the second-order expansions, the location of the macro-F1 optimal prior, and
   a learned plug-in (estimated means) in binary and multiclass settings.
4. The Study 4 gaps (FM5, DEV): the exact binary coverage term, the first-order
   multiclass term, the balance term with a leave-one-dataset-out curvature, the
   imbalance correlations, and the regime boundary.
5. Label equivalence under a log learning curve (Corollary 1).

Only class counts are read from the data cache (DEV ids from data/pool_i2.yaml);
sealed_heldout/ is never touched. Thread pools are limited to one thread.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from itertools import combinations  # noqa: E402
from math import comb, log  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import gammaln, lambertw  # noqa: E402
from scipy.stats import hypergeom, norm, rankdata, spearmanr  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.settings import settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("THEORY_CHECKS.md")
STAGE_A = settings.results_dir / "study4_stage_a"
MULTS = (1, 3, 5, 10)
SEED = 20261021
MC_REPS = 200_000
DELTA = 0.05
SYNTHETIC = {
    "balanced C=2": [500, 500],
    "C=2, 1:4": [200, 800],
    "C=2, 1:19": [50, 950],
    "C=5, geometric": [400, 250, 160, 110, 80],
    "C=10, long tail": [300, 200, 150, 100, 80, 60, 40, 30, 25, 15],
}


# ----------------------------------------------------------------------------
# Closed forms (Proposition A, Lemma B1, Corollary 2)
# ----------------------------------------------------------------------------


def log_comb(a: float, b: float) -> float:
    """Log binomial coefficient via log-gamma (minus infinity when b > a)."""
    if b > a:
        return -np.inf
    return float(gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1))


def p_miss(counts: np.ndarray, budget: int) -> np.ndarray:
    """Exact hypergeometric probability that a uniform draw misses each class."""
    n = int(counts.sum())
    return np.array(
        [np.exp(log_comb(n - c, budget) - log_comb(n, budget)) for c in counts]
    )


def p_full_ie(counts: np.ndarray, budget: int) -> float:
    """P(every class drawn) by inclusion-exclusion over the missed classes."""
    n = int(counts.sum())
    total = 0.0
    for size in range(len(counts) + 1):
        for subset in combinations(range(len(counts)), size):
            rest = n - int(counts[list(subset)].sum())
            total += (-1) ** size * np.exp(log_comb(rest, budget) - log_comb(n, budget))
    return total


def p_all_at_least(counts: np.ndarray, budget: int, r: int) -> float:
    """Exact P(N_k >= r for all k) from the generating function (integer math)."""
    poly = [1]
    for c in (int(v) for v in counts):
        factor = [comb(c, j) if r <= j <= c else 0 for j in range(min(c, budget) + 1)]
        poly = poly_mul(poly, factor, budget)
    return poly[budget] / comb(int(counts.sum()), budget)


def poly_mul(p: list[int], f: list[int], degree: int) -> list[int]:
    """Product of two integer polynomials truncated at the given degree."""
    new = [0] * (degree + 1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(f[: degree - i + 1]):
            new[i + j] += a * b
    return new


def share_moments(counts: np.ndarray, budget: int) -> tuple[np.ndarray, float]:
    """Variance of each context share and E[D] for an unconditional uniform draw.

    D is the squared Euclidean distance of the shares from uniform.
    """
    n = int(counts.sum())
    pi = counts / n
    fpc = (n - budget) / (n - 1)
    var = pi * (1 - pi) * fpc / budget
    k = len(counts)
    e_d = float(((pi - 1 / k) ** 2).sum() + (1 - (pi**2).sum()) * fpc / budget)
    return var, e_d


def binary_cov_pmf(counts: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """Support 1..B-1 of N_1 and its pmf given full coverage (binary pool)."""
    n, n1 = int(counts.sum()), int(counts[0])
    j = np.arange(1, budget)
    pmf = hypergeom(n, n1, budget).pmf(j)
    return j, pmf / pmf.sum()


def e_d_cov_binary(counts: np.ndarray, budget: int) -> float:
    """Exact E[D | full coverage] for a binary pool."""
    j, pmf = binary_cov_pmf(counts, budget)
    return float((pmf * 2 * (j / budget - 0.5) ** 2).sum())


def first_budget(ok, start: int, stop: int = 5000) -> int:
    """Smallest budget in [start, stop) for which ok(budget) holds."""
    for b in range(start, stop):
        if ok(b):
            return b
    return -1


def budgets(counts: np.ndarray, delta: float, r: int) -> dict[str, float]:
    """Coverage budgets of Corollary 2 for one pool (labels per class: r)."""
    n, k = int(counts.sum()), len(counts)
    pi = counts / n
    lo = k * r
    big_l = log(k / delta)
    pmin = pi.min()

    def tail(b: int) -> float:
        return float(sum(hypergeom(n, int(c), b).cdf(r - 1) for c in counts))

    out = {
        "exact": first_budget(
            lambda b: p_all_at_least(counts, b, r) >= 1 - delta, lo, n
        ),
        "union_hyp": first_budget(lambda b: tail(b) <= delta, lo, n),
        "lower_max_k": first_budget(
            lambda b: max(hypergeom(n, int(c), b).cdf(r - 1) for c in counts) <= delta,
            lo,
            n,
        ),
    }
    if r == 1:
        out["with_repl"] = first_budget(
            lambda b: ((1 - pi) ** b).sum() <= delta, lo, 100_000
        )
        out["ln(C/d)/pmin"] = float(np.ceil(big_l / pmin))
    else:
        out["chernoff"] = float(
            np.ceil((r + big_l + np.sqrt(big_l**2 + 2 * r * big_l)) / pmin)
        )
    return out


# ----------------------------------------------------------------------------
# Block 1: closed forms against Monte Carlo
# ----------------------------------------------------------------------------


def mc_block(name: str, counts: np.ndarray, rng: np.random.Generator) -> list[dict]:
    """Closed forms vs Monte Carlo for one pool at every budget multiplier."""
    rows = []
    k = len(counts)
    n = int(counts.sum())
    for m in MULTS:
        b = m * k
        if b >= n:
            continue
        comp = rng.multivariate_hypergeometric(counts, b, size=MC_REPS)
        shares = comp / b
        cov = (comp > 0).all(axis=1)
        d = ((shares - 1 / k) ** 2).sum(axis=1)
        q = p_miss(counts, b)
        var, e_d = share_moments(counts, b)
        pi = counts / n
        bal = int(np.ceil(n / k))
        row = dict(
            pool=name,
            m=m,
            max_err_q=float(np.abs(q - (comp == 0).mean(axis=0)).max()),
            covered_pred=float(k - q.sum()),
            covered_mc=float((comp > 0).sum(axis=1).mean()),
            p_full_ie=p_full_ie(counts, b),
            p_full_gf=p_all_at_least(counts, b, 1),
            p_full_mc=float(cov.mean()),
            q_le_wr=bool((q <= (1 - pi) ** b + 1e-12).all()),
            karamata=bool(
                q.sum() >= k * p_miss(np.array([bal, n - bal]), b)[0] - 1e-12
            ),
            max_rel_err_var=float(np.abs(shares.var(axis=0) / var - 1).max()),
            e_d_pred=e_d,
            e_d_mc=float(d.mean()),
            e_d_cov_mc=float(d[cov].mean()) if cov.any() else np.nan,
            e_d_cov_exact=e_d_cov_binary(counts, b) if k == 2 else np.nan,
        )
        rows.append(row)
    return rows


def dev_counts() -> dict[int, np.ndarray]:
    """Class counts of the DEV datasets (labels only)."""
    pool = yaml.safe_load((ROOT / "data" / "pool_i2.yaml").read_text())
    out = {}
    for entry in pool["dev"]:
        y = load(int(entry["did"]), settings.data_cache_dir).y
        out[int(entry["did"])] = np.bincount(y).astype(np.int64)
    return out


# ----------------------------------------------------------------------------
# Block 3: Proposition B, Bayes plug-in learner with Gaussian classes
# ----------------------------------------------------------------------------


def gauss_rates(tau: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """TPR a and FPR b of 'predict 1 iff LLR > tau' for N(+-delta/2, 1) classes."""
    c = np.asarray(tau, dtype=float) / delta
    return norm.cdf(delta / 2 - c), norm.cdf(-delta / 2 - c)


def gauss_metrics(tau: np.ndarray, pi1: float, delta: float) -> dict[str, np.ndarray]:
    """Population BA, accuracy and macro-F1 of the threshold rule (class 1 = pos)."""
    a, b = gauss_rates(tau, delta)
    pi2 = 1 - pi1
    tp1, fp1, fn1, tp2 = pi1 * a, pi2 * b, pi1 * (1 - a), pi2 * (1 - b)
    f1a = 2 * tp1 / (2 * tp1 + fp1 + fn1)
    f1b = 2 * tp2 / (2 * tp2 + fp1 + fn1)
    return dict(ba=(a + 1 - b) / 2, acc=tp1 + tp2, mf1=(f1a + f1b) / 2)


def tau_opt_mf1(pi1: float, delta: float) -> float:
    """Threshold on the LLR that maximises population macro-F1."""
    res = minimize_scalar(
        lambda t: -gauss_metrics(np.array(t), pi1, delta)["mf1"],
        bounds=(-15, 15),
        method="bounded",
        options=dict(xatol=1e-9),
    )
    return float(res.x)


def second_derivative(f, x: float, h: float = 1e-3) -> float:
    """Central finite-difference second derivative."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


def bayes_binary_rows(pool: int = 1000) -> list[dict]:
    """Exact vs second-order protocol gaps for the binary Gaussian plug-in."""
    rows = []
    for pi1 in (0.5, 0.35, 0.25, 0.1):
        for delta in (1.0, 2.0):
            n1 = int(round(pi1 * pool))
            counts = np.array([n1, pool - n1])
            tau_a = log((1 - pi1) / pi1)
            tau_f = tau_opt_mf1(pi1, delta)
            f0 = norm.pdf(delta / 2) / delta
            g = (1 - pi1) * norm.pdf((-(delta**2) / 2 - tau_a) / delta) / delta
            hf = -second_derivative(
                lambda t: float(gauss_metrics(np.array(t), pi1, delta)["mf1"]), tau_f
            )
            strat = gauss_metrics(np.array(0.0), pi1, delta)
            for m in (2, 5, 10, 25, 100):
                b = 2 * m
                j, pmf = binary_cov_pmf(counts, b)
                tau_hat = np.log((b - j) / j)
                met = gauss_metrics(tau_hat, pi1, delta)
                gap = {k: float(strat[k] - (pmf * met[k]).sum()) for k in met}
                d_cov = float((pmf * 2 * (j / b - 0.5) ** 2).sum())
                var_tau = float((pmf * (tau_hat - tau_a) ** 2).sum())
                e_tf = float((pmf * (tau_hat - tau_f) ** 2).sum())
                rows.append(
                    dict(
                        pi1=pi1,
                        Delta=delta,
                        m=m,
                        tau_F=round(tau_f, 3),
                        tau_A=round(tau_a, 3),
                        gap_ba=gap["ba"],
                        ba_2nd=2 * f0 * d_cov,
                        gap_acc=gap["acc"],
                        acc_2nd=0.5 * g * (var_tau - tau_a**2),
                        gap_mf1=gap["mf1"],
                        mf1_2nd=0.5 * hf * (e_tf - tau_f**2),
                    )
                )
    return rows


def endpoint_rows() -> dict[str, float]:
    """Check Prop. B4's endpoint derivative signs on a grid of binary Gaussians.

    Cells where the rule is numerically degenerate at the endpoint (a or 1 - b
    below 1e-8, so the metric underflows) are counted separately.
    """
    out = dict(cells_pi=0, viol_pi=0, cells_u=0, viol_u=0, between=0, degenerate=0)
    eps = 1e-4
    for pi1 in np.linspace(0.02, 0.48, 47):
        for delta in np.linspace(0.2, 5.0, 49):
            tau_a = log((1 - pi1) / pi1)

            def mf1(t: float, p: float = pi1, d: float = delta) -> float:
                return float(gauss_metrics(np.array(t), p, d)["mf1"])

            a, b = gauss_rates(np.array(tau_a), delta)
            if pi1 <= 1 / 3 and min(a, 1 - b) < 1e-8:
                out["degenerate"] += 1
            elif pi1 <= 1 / 3:
                out["cells_pi"] += 1
                out["viol_pi"] += mf1(tau_a - eps) <= mf1(tau_a)
            a, b = gauss_rates(np.array(0.0), delta)
            if a - b >= pi1 / (1 - pi1):
                out["cells_u"] += 1
                out["viol_u"] += mf1(eps) <= mf1(0.0)
                out["between"] += 0 < tau_opt_mf1(pi1, delta) < tau_a
    return out


def class_means(k: int, delta: float, dim: int) -> np.ndarray:
    """Class means on scaled unit vectors (pairwise distance delta)."""
    mu = np.zeros((k, dim))
    mu[np.arange(k), np.arange(k)] = delta / np.sqrt(2)
    return mu


def plugin_predict(xt: np.ndarray, means: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Plug-in rule argmax_k log prior_k - ||x - mean_k||^2 / 2."""
    d2 = ((xt[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
    return np.argmax(np.log(prior)[None, :] - d2 / 2, axis=1)


def score(yt: np.ndarray, pred: np.ndarray, k: int) -> tuple[float, float, float]:
    """Macro-F1, balanced accuracy and accuracy."""
    labels = list(range(k))
    rec = np.array([(pred[yt == c] == c).mean() for c in labels])
    mf1 = f1_score(yt, pred, labels=labels, average="macro", zero_division=0)
    return float(mf1), float(rec.mean()), float((pred == yt).mean())


def context(comp: np.ndarray, mu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Estimated class means from a context of the given composition."""
    return np.array(
        [
            mu[c] + rng.standard_normal((n_c, mu.shape[1])).mean(axis=0)
            for c, n_c in enumerate(comp)
        ]
    )


def learned_rows(rng: np.random.Generator, reps: int = 300) -> list[dict]:
    """Stratified vs random contexts for a plug-in learner with estimated means."""
    configs = {
        "C=2 pi=(.25,.75) D=1.5": (np.array([0.25, 0.75]), 1.5),
        "C=2 pi=(.10,.90) D=2.5": (np.array([0.10, 0.90]), 2.5),
        "C=3 pi=(.1,.3,.6) D=2": (np.array([0.1, 0.3, 0.6]), 2.0),
        "C=3 balanced D=2": (np.full(3, 1 / 3), 2.0),
    }
    rows = []
    for name, (pi, delta) in configs.items():
        k, dim = len(pi), 4
        mu = class_means(k, delta, dim)
        yt = rng.choice(k, size=20_000, p=pi)
        xt = mu[yt] + rng.standard_normal((len(yt), dim))
        counts = np.round(pi * 2000).astype(np.int64)
        for m in (1, 2, 3, 5, 10, 25):
            b = m * k
            res = {"S": [], "R": [], "Rpc": []}
            for _ in range(reps):
                comp = rng.multivariate_hypergeometric(counts, b)
                while (comp == 0).any():
                    comp = rng.multivariate_hypergeometric(counts, b)
                strat = np.full(k, m)
                res["S"].append(
                    score(yt, plugin_predict(xt, context(strat, mu, rng), strat / b), k)
                )
                means_r = context(comp, mu, rng)
                res["R"].append(score(yt, plugin_predict(xt, means_r, comp / b), k))
                res["Rpc"].append(
                    score(yt, plugin_predict(xt, means_r, np.full(k, 1 / k)), k)
                )
            mean = {key: np.mean(v, axis=0) for key, v in res.items()}
            gap = mean["S"] - mean["R"]
            gap_pc = mean["S"] - mean["Rpc"]
            rows.append(
                dict(
                    config=name,
                    m=m,
                    gap_mf1=gap[0],
                    gap_ba=gap[1],
                    gap_acc=gap[2],
                    gap_mf1_priorcorr=gap_pc[0],
                    gap_acc_priorcorr=gap_pc[2],
                )
            )
    return rows


# ----------------------------------------------------------------------------
# Block 4: Study 4 gaps (FM5, DEV)
# ----------------------------------------------------------------------------


def per_class_f1(y: np.ndarray, rows: pd.DataFrame, k: int) -> np.ndarray:
    """Mean per-class F1 of the stratified draws on their complements."""
    out = []
    for r in rows.itertuples():
        rest = np.setdiff1d(np.arange(len(y)), np.asarray(r.labeled))
        out.append(
            f1_score(
                y[rest],
                np.asarray(r.pred)[rest],
                labels=list(range(k)),
                average=None,
                zero_division=0,
            )
        )
    return np.mean(out, axis=0)


def coverage_pred(
    y: np.ndarray, strat: pd.DataFrame, counts: np.ndarray, b: int
) -> tuple[float, float, str]:
    """Predicted coverage parts of the macro-F1 and accuracy gaps.

    Binary: exact (a missed class leaves a constant predictor). Multiclass:
    first order (missed class F1 = 0, recall = 0, others unchanged).
    """
    q = p_miss(counts, b)
    n, k = int(counts.sum()), len(counts)
    m_s, a_s = strat.macro_f1.mean(), strat.accuracy.mean()
    if k == 2:
        other = (counts[::-1] - b) / (n - b)
        return (
            float((q * (m_s - other / (1 + other))).sum()),
            float((q * (a_s - other)).sum()),
            "exact",
        )
    f1k = per_class_f1(y, strat, k)
    pi_c = counts / (n - b)
    return float((q * f1k).mean()), float((q * pi_c * f1k).sum()), "first"


def draw_rows(did: int, y: np.ndarray, df: pd.DataFrame) -> list[dict]:
    """One row per random draw: composition statistics and gaps to stratified."""
    k = int(y.max()) + 1
    pi = np.bincount(y, minlength=k) / len(y)
    out = []
    for mult, g in df.groupby("mult"):
        strat = g[g.strategy == "stratified"]
        for t in g[g.strategy == "random"].itertuples():
            sh = np.bincount(y[np.asarray(t.labeled)], minlength=k) / len(t.labeled)
            out.append(
                dict(
                    dataset=did,
                    mult=int(mult),
                    covered=bool((sh > 0).all()),
                    D=float(((sh - 1 / k) ** 2).sum()),
                    Dpi=float(((sh - pi) ** 2).sum()),
                    tv=float(0.5 * np.abs(sh - 1 / k).sum()),
                    dF=float(t.macro_f1 - strat.macro_f1.mean()),
                    dA=float(t.accuracy - strat.accuracy.mean()),
                )
            )
    return out


def constant_identity_error(
    y: np.ndarray, rand: pd.DataFrame, counts: np.ndarray, b: int
) -> float:
    """Max deviation of missed binary draws from the constant-predictor values."""
    if len(counts) != 2:
        return np.nan
    n = int(counts.sum())
    err = 0.0
    for t in rand[rand.coverage < 1].itertuples():
        seen = int(np.bincount(y[np.asarray(t.labeled)], minlength=2).argmax())
        p = (counts[seen] - b) / (n - b)
        err = max(err, abs(t.macro_f1 - p / (1 + p)), abs(t.accuracy - p))
    return err


def dataset_rows(did: int, counts: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Gap decomposition rows and per-draw rows for one DEV dataset (FM5)."""
    y = load(did, settings.data_cache_dir).y
    df = pd.read_parquet(STAGE_A / f"{did}.parquet")
    df = df[df.learner == "fm5"]
    rows = []
    for mult, g in df.groupby("mult"):
        b = int(mult) * len(counts)
        strat, rand = g[g.strategy == "stratified"], g[g.strategy == "random"]
        miss = (rand.coverage < 1).to_numpy()
        cf, ca, kind = coverage_pred(y, strat, counts, b)
        _, e_d = share_moments(counts, b)
        rows.append(
            dict(
                dataset=did,
                C=len(counts),
                mult=int(mult),
                kind=kind,
                p_full=p_full_ie(counts, b),
                e_d=e_d,
                f1_gap=strat.macro_f1.mean() - rand.macro_f1.mean(),
                acc_gap=strat.accuracy.mean() - rand.accuracy.mean(),
                f1_miss_obs=float(
                    ((strat.macro_f1.mean() - rand.macro_f1) * miss).mean()
                ),
                acc_miss_obs=float(
                    ((strat.accuracy.mean() - rand.accuracy) * miss).mean()
                ),
                f1_cov_pred=cf,
                acc_cov_pred=ca,
                strat_f1=strat.macro_f1.mean(),
                rand_f1=rand.macro_f1.mean(),
                identity_err=constant_identity_error(y, rand, counts, b),
            )
        )
    return rows, draw_rows(did, y, df)


def lodo_kappa(draws: pd.DataFrame, metric: str, var: str) -> dict[tuple, float]:
    """Leave-one-dataset-out slope of -gap on the imbalance (through the origin)."""
    cov = draws[draws.covered]
    out = {}
    for (did, mult), _ in cov.groupby(["dataset", "mult"]):
        rest = cov[(cov.dataset != did) & (cov.mult == mult)]
        x, z = rest[var].to_numpy(), -rest[metric].to_numpy()
        out[(did, mult)] = (
            float((x * z).sum() / (x * x).sum()) if (x > 0).any() else 0.0
        )
    return out


def centred(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Subtract dataset x multiplier means (within-dataset variation only)."""
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby(["dataset", "mult"])[c].transform("mean")
    return out


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation of x and y controlling for z (rank residuals)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    design = np.column_stack([np.ones_like(rz), rz])
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def other_fm_draws(counts: dict[int, np.ndarray]) -> pd.DataFrame:
    """Per-draw rows for FM1-FM3 (Study 4 Stage B), for the robustness table."""
    rows = []
    stage_b = settings.results_dir / "study4_stage_b"
    for did in counts:
        y = load(did, settings.data_cache_dir).y
        for fm in ("fm1", "fm2", "fm3"):
            path = stage_b / f"{did}_{fm}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            for r in draw_rows(did, y, df[df.learner == fm]):
                rows.append(dict(r, learner=fm))
    return pd.DataFrame(rows)


def imbalance_table(draws: pd.DataFrame) -> pd.DataFrame:
    """Within-dataset correlations of the gaps with D (uniform) and Dpi (prior)."""
    rows = []
    cov = centred(
        draws[draws.covered & (draws.mult > 1)], ["D", "Dpi", "tv", "dF", "dA"]
    )
    for mult, g in cov.groupby("mult"):
        rows.append(
            dict(
                mult=mult,
                draws=len(g),
                rho_tv_F1=spearmanr(g.tv, g.dF)[0],
                pr_F1_D_given_Dpi=partial_spearman(g.D, g.dF, g.Dpi),
                pr_F1_Dpi_given_D=partial_spearman(g.Dpi, g.dF, g.D),
                pr_acc_D_given_Dpi=partial_spearman(g.D, g.dA, g.Dpi),
                pr_acc_Dpi_given_D=partial_spearman(g.Dpi, g.dA, g.D),
            )
        )
    return pd.DataFrame(rows)


def intercept_table(draws: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Lemma 1 check: within covered draws, the gap at D = 0 should be zero.

    OLS of the per-draw gap (random - stratified mean) on D with an intercept;
    95% interval by bootstrap over datasets.
    """
    rows = []
    cov = draws[draws.covered & (draws.mult > 1)]
    for mult, g in cov.groupby("mult"):
        dids = g.dataset.unique()

        def fit(sub: pd.DataFrame) -> np.ndarray:
            design = np.column_stack([np.ones(len(sub)), sub.D])
            return np.linalg.lstsq(design, sub.dF.to_numpy(), rcond=None)[0]

        est = fit(g)
        boot = np.array(
            [
                fit(pd.concat([g[g.dataset == d] for d in rng.choice(dids, len(dids))]))
                for _ in range(1000)
            ]
        )
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        rows.append(
            dict(
                mult=mult,
                intercept=est[0],
                int_lo=lo[0],
                int_hi=hi[0],
                slope=est[1],
                slope_lo=lo[1],
                slope_hi=hi[1],
            )
        )
    return pd.DataFrame(rows)


def balance_prediction(frame: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    """Add the LODO balance term kappa * E[D | cov] * P(cov) and its observed twin."""
    kap_f = lodo_kappa(draws, "dF", "D")
    cov = draws[draws.covered].groupby(["dataset", "mult"])
    e_d_cov = cov.D.mean()
    out = frame.copy()
    keys = list(zip(out.dataset, out.mult))
    out["kappa_f1"] = [kap_f.get(key, 0.0) for key in keys]
    out["f1_bal_pred"] = [
        kap_f.get(key, 0.0) * e_d_cov.get(key, 0.0) * p
        for key, p in zip(keys, out.p_full)
    ]
    out["f1_bal_obs"] = out.f1_gap - out.f1_miss_obs
    out["acc_bal_obs"] = out.acc_gap - out.acc_miss_obs
    out["f1_total_pred"] = out.f1_cov_pred + out.f1_bal_pred
    return out


def regime_boundary(
    counts: dict[int, np.ndarray], frame: pd.DataFrame, kappa: float
) -> pd.DataFrame:
    """Smallest m at which the balance term exceeds the coverage term (model)."""
    rows = []
    for did, c in counts.items():
        k, n = len(c), int(c.sum())
        fbar = float(frame[frame.dataset == did].strat_f1.mean())
        crossing = np.nan
        for b in range(k, min(n // 2, 40 * k)):
            cov_term = fbar * p_miss(c, b).mean()
            _, e_d = share_moments(c, b)
            if kappa * e_d * p_full_ie(c, b) >= cov_term:
                crossing = b / k
                break
        pmin = c.min() / n
        cst = kappa * (1 - ((c / n) ** 2).sum()) / fbar
        arg = -pmin * cst
        lw = -np.real(lambertw(arg, -1)) / pmin / k if arg >= -1 / np.e else 1.0
        bias = kappa * float(((c / n - 1 / k) ** 2).sum())
        lg = max(log(fbar / bias), 0.0) / pmin / k if bias > 0 else np.inf
        rows.append(
            dict(
                dataset=did,
                C=k,
                pi_min=pmin,
                m_cross=crossing,
                m_lambertW=lw,
                m_bias=lg,
                m_closed=min(lw, lg),
            )
        )
    return pd.DataFrame(rows)


def equivalence_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Corollary 1: lambda = exp(gap / b) under the curve a + b ln B (FM5)."""
    rows = []
    for did, g in frame.groupby("dataset"):
        g = g.sort_values("mult")
        slope = np.polyfit(np.log(g.mult * g.C), g.rand_f1, 1)[0]
        if slope <= 0.01:
            continue
        for r in g.itertuples():
            rows.append(
                dict(
                    dataset=did,
                    mult=r.mult,
                    lam_obs=np.exp(r.f1_gap / slope),
                    lam_cov=np.exp(r.f1_cov_pred / slope),
                    lam_cov_bal=np.exp(r.f1_total_pred / slope),
                )
            )
    eq = pd.DataFrame(rows)
    cols = ["lam_obs", "lam_cov", "lam_cov_bal"]
    geo = eq.groupby("mult")[cols].agg(lambda v: float(np.exp(np.log(v).mean())))
    med = eq.groupby("mult")[cols].median().add_suffix("_median")
    return geo.join(med).reset_index()


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------


def md(df: pd.DataFrame, digits: int = 4) -> str:
    """Markdown table with rounding."""
    return df.round(digits).to_markdown(index=False)


def study4_sections(
    counts: dict[int, np.ndarray], rng: np.random.Generator
) -> list[str]:
    """Block 4 and 5 report sections."""
    parts = [dataset_rows(did, c) for did, c in counts.items()]
    frame = pd.DataFrame([r for p in parts for r in p[0]])
    draws = pd.DataFrame([r for p in parts for r in p[1]])
    frame = balance_prediction(frame, draws)
    cols = [
        "f1_gap",
        "f1_miss_obs",
        "f1_cov_pred",
        "f1_bal_obs",
        "f1_bal_pred",
        "f1_total_pred",
        "acc_gap",
        "acc_miss_obs",
        "acc_cov_pred",
        "acc_bal_obs",
    ]
    by_kind = frame.groupby(["kind", "mult"])[cols].mean().reset_index()
    by_mult = frame.groupby("mult")[cols].mean().reset_index()
    corr = []
    for mult, g in frame.groupby("mult"):
        corr.append(
            dict(
                mult=mult,
                r_f1_cov=np.corrcoef(g.f1_cov_pred, g.f1_miss_obs)[0, 1],
                r_acc_cov=np.corrcoef(g.acc_cov_pred, g.acc_miss_obs)[0, 1],
                r_f1_bal=np.corrcoef(g.f1_bal_pred, g.f1_bal_obs)[0, 1],
                r_f1_total=np.corrcoef(g.f1_total_pred, g.f1_gap)[0, 1],
                share_acc_bal_lt_f1_bal=float((g.acc_bal_obs < g.f1_bal_obs).mean()),
            )
        )
    kappa = float(frame[frame.mult == 3].kappa_f1.median())
    regime = regime_boundary(counts, frame, kappa)
    return [
        "## 4. Study 4 gaps (FM5, DEV, 20 datasets)",
        "",
        "Gaps are stratified minus random. `*_miss_obs` = observed part from draws"
        " that missed a class, P(miss) (M^S - E[M^R | miss]); `*_bal_obs` = the"
        " rest, P(cov) (M^S - E[M^R | cov]). `*_cov_pred`: exact binary formula"
        " (kind = exact) or first-order multiclass formula (kind = first)."
        " `f1_bal_pred` = LODO kappa x E[D | cov] x P(cov), kappa the slope of the"
        " per-draw macro-F1 loss on D (distance of the shares from uniform) in the"
        " other 19 datasets. `share_acc_bal_lt_f1_bal`: share of datasets whose"
        " accuracy balance part is below the macro-F1 one (Prop. B5).",
        "",
        "### By pool type",
        "",
        md(by_kind),
        "",
        "### All datasets",
        "",
        md(by_mult),
        "",
        "Binary constant-predictor identity (Prop. A7) on the missed draws: max"
        f" |observed - formula| = {np.nanmax(frame.identity_err):.2e}.",
        "",
        "### Correlation across datasets (predicted vs observed part)",
        "",
        md(pd.DataFrame(corr), 3),
        "",
        "### Imbalance within covered random draws (dataset-centred, Spearman)",
        "",
        "`rho_tv_F1` reproduces the earlier balance result; `pr_*` are partial"
        " Spearman correlations of the per-draw gap (random - stratified) with D"
        " given Dpi, and with Dpi given D.",
        "",
        md(imbalance_table(draws), 3),
        "",
        "Same, FM1-FM3 (Study 4 Stage B):",
        "",
        md(
            pd.concat(
                [
                    imbalance_table(g).assign(learner=fm)
                    for fm, g in other_fm_draws(counts).groupby("learner")
                ]
            ),
            3,
        ),
        "",
        "### Lemma 1 check: macro-F1 gap of covered draws regressed on D",
        "",
        md(intercept_table(draws, rng)),
        "",
        "### Regime boundary (Corollary 3)",
        "",
        f"kappa (median LODO slope at 3C) = {kappa:.3f}. m_cross: first m = B/C"
        " at which kappa E[D] P(cov) >= fbar mean_k q_k. Closed forms of"
        " Corollary 3: m_lambertW (variance part only; 1.0 when it dominates at"
        " every budget), m_bias = ln(fbar / (kappa ||pi - u||^2)) / (C pi_min),"
        " m_closed = their minimum.",
        "",
        md(regime, 3),
        "",
        f"Median m_cross = {regime.m_cross.median():.2f}; median m_closed ="
        f" {regime.m_closed.median():.2f}; Spearman(m_cross, m_closed) ="
        f" {spearmanr(regime.m_cross, regime.m_closed)[0]:.2f}.",
        "",
        "## 5. Label equivalence (Corollary 1), FM5, geometric mean and median",
        "",
        md(equivalence_table(frame), 3),
        "",
    ]


def main() -> None:
    """Run all checks and write reports/THEORY_CHECKS.md."""
    rng = np.random.default_rng(SEED)
    counts = dev_counts()
    pools = {k: np.array(v, dtype=np.int64) for k, v in SYNTHETIC.items()}
    pools.update({f"DEV {d}": c for d, c in counts.items()})
    mc = pd.DataFrame([r for name, c in pools.items() for r in mc_block(name, c, rng)])
    bud = pd.DataFrame(
        [
            dict(pool=name, r=r, **budgets(c, DELTA, r))
            for name, c in pools.items()
            for r in (1, 3)
        ]
    )
    mc_check = []
    for name, c in pools.items():
        b = int(bud[(bud.pool == name) & (bud.r == 1)].exact.iloc[0])
        comp = rng.multivariate_hypergeometric(c, b, size=MC_REPS)
        mc_check.append(dict(pool=name, B=b, p_full_mc=float((comp > 0).all(1).mean())))
    bayes = pd.DataFrame(bayes_binary_rows())
    text = [
        "# Numerical checks for THEORY_PROTOCOL_BIAS.md",
        "",
        "Generated by `reports/theory_checks.py` (seed"
        f" {SEED}, {MC_REPS} Monte Carlo draws per cell). Only class counts are read"
        " from the DEV datasets; sealed_heldout/ is not touched.",
        "",
        "## 1. Proposition A and Lemma B1: closed forms vs Monte Carlo",
        "",
        "`max_err_q`: max over classes of |q_k - MC|. `q_le_wr`: q_k <= (1-pi_k)^B"
        " for all k. `karamata`: sum q_k >= C q(ceil(n/C)). `max_rel_err_var`: max"
        " relative error of the share variances.",
        "",
        md(mc, 5),
        "",
        f"## 2. Coverage budgets (Corollary 2), delta = {DELTA}",
        "",
        "B in labels. r = labels per class required. `exact`: smallest B with"
        " P(all N_k >= r) >= 1 - delta (generating function); `union_hyp`: smallest"
        " B with sum_k P(N_k < r) <= delta; `lower_max_k`: smallest B with"
        " max_k P(N_k < r) <= delta (a lower bound on `exact`).",
        "",
        md(bud, 1),
        "",
        "Monte Carlo P(full coverage) at the exact r = 1 budget:",
        "",
        md(pd.DataFrame(mc_check), 4),
        "",
        "## 3. Proposition B: Bayes plug-in learner, binary Gaussians N(+-Delta/2, 1)",
        "",
        "Exact protocol gaps (stratified - random | covered; pool of 1000) against"
        " the second-order expressions of Prop. B2-B4. tau_F, tau_A: macro-F1 and"
        " accuracy optimal thresholds on the log-likelihood ratio (tau = 0 is the"
        " uniform prior, the stratified rule).",
        "",
        md(bayes),
        "",
        "Sign agreement of the second-order expressions with the exact gaps:"
        f" BA {np.mean(np.sign(bayes.gap_ba) == np.sign(bayes.ba_2nd)):.2f},"
        f" accuracy {np.mean(np.sign(bayes.gap_acc) == np.sign(bayes.acc_2nd)):.2f},"
        f" macro-F1 {np.mean(np.sign(bayes.gap_mf1) == np.sign(bayes.mf1_2nd)):.2f}"
        f" ({len(bayes)} cells; BA gap > 0 in {np.mean(bayes.gap_ba > 0):.2f}).",
        "",
        "### Prop. B4 endpoint conditions on a grid (47 priors x 49 separations)",
        "",
        md(pd.DataFrame([endpoint_rows()])),
        "",
        "### Learned plug-in (estimated means, known identity covariance), d = 4",
        "",
        "Gaps stratified - random (random contexts conditioned on coverage),"
        " population metrics on 20 000 test rows, 300 contexts per cell."
        " `*_priorcorr`: the random-context learner uses the uniform prior instead"
        " of the context shares.",
        "",
        md(pd.DataFrame(learned_rows(rng))),
        "",
    ]
    text += study4_sections(counts, rng)
    OUT.write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
