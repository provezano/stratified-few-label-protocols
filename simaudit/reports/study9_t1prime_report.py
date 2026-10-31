"""Study 9 on EXP' = T1' (Amendment 1): frozen analysis plus pre-specified extras.

Part A runs ``simaudit.analysis.study9.analyze`` and ``report`` unchanged (the
module's SHA-256 is checked against freeze note 1 first). That output is the
confirmatory result.

Part B computes quantities the prereg lists that ``analyze`` does not return,
using only the frozen module's functions with their documented parameters:
the 2000-resample H1-dist CIs, the H2a magnitude CIs, FM1 replication
(FM-robustness), the AFT for other strategies and FMs, the four-FM ICC, the
coverage floor, the selection ceiling CDF, H3 descriptives, accuracy, the
epsilon sensitivity and the T1' sensitivity subsets (Study 8 A2.5). Part B has
no confirmatory weight. The only new numerical code is the weighted Turnbull
EM for the post-stratified sensitivity, flagged as such.

Usage: ``uv run python reports/study9_t1prime_report.py`` writes
``reports/STUDY9_T1PRIME_OUTPUT.md`` (verbatim generated output).
"""

import hashlib
from collections.abc import Callable
from math import inf
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from simaudit.analysis import study9 as a9
from simaudit.data.openml_loader import load
from simaudit.runner.study9 import dataset_key, layout, row_permutation
from simaudit.settings import PROJECT_ROOT, settings

FROZEN_SHA = {
    "src/simaudit/analysis/study9.py": (
        "aaf0bae1defa10d80b1db235614fe228ebf0aa26e82499d0d29dd51a3f6c50ba"
    ),
    "src/simaudit/runner/study9.py": (
        "6fe40d008265fff82a64c8b98321774dc30736c075e89163acdcf176787dd4c0"
    ),
}
GRID = (1, 2, 3, 5, 7, 10, 15, 20, 30)
LEARNERS = ("fm2", "fm1", "fm5", "fm3")
STRATS = ("random", "stratified", "kmedoids_nested", "lucos_nested")
OUT = Path(__file__).with_name("STUDY9_T1PRIME_OUTPUT.md")


def check_freeze() -> None:
    """Abort if the analysis or runner module differs from freeze note 1."""
    for rel, expected in FROZEN_SHA.items():
        got = hashlib.sha256((PROJECT_ROOT / rel).read_bytes()).hexdigest()
        if got != expected:
            raise SystemExit(f"{rel}: SHA-256 {got} differs from the freeze note")


def pool_members() -> pd.DataFrame:
    """T1' members with their result key, source and stratum."""
    rows = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())
    frame = pd.DataFrame(rows["datasets"])
    frame["dataset"] = [dataset_key(int(d)) for d in frame.did]
    return frame.set_index("dataset")


# ------------------------------------------------------------------ helpers
def intervals(curve: pd.DataFrame, lr: str, st: str, eps: float = a9.EPS) -> pd.Series:
    """m* interval per dataset for one learner and strategy."""
    sub = a9._xs(curve, lr, st)
    return pd.Series(
        {k: a9.saturation_interval(*a9._points(r), eps=eps) for k, r in sub.iterrows()},
        dtype=object,
    )


def fmt_iv(iv: tuple[float, float]) -> str:
    return f"({iv[0]:g}, {iv[1]:g}]"


def dist_row(label: str, ivs: list[Any], n_boot: int = a9.N_BOOT) -> str:
    """One markdown row of the H1-dist summary."""
    d = a9.distribution_summary(ivs, n_boot=n_boot)
    shares = " | ".join(
        f"{d[k][0]:.2f} [{d[k][1]:.2f}, {d[k][2]:.2f}]"
        for k in ("P(m*<=5)", "P(m*<=10)", "P(m*>20)")
    )
    cdf = " ".join(f"{d['cdf'][float(g)]:.2f}" for g in GRID)
    return (
        f"| {label} | {d['n']} | {fmt_iv(d['median'])} | {fmt_iv(d['q25'])}–"
        f"{fmt_iv(d['q75'])} | {shares} | {cdf} |"
    )


DIST_HEAD = [
    "| set | n | median | IQR cells | P(m*<=5) | P(m*<=10) | P(m*>20) "
    "| F(m) at 1 2 3 5 7 10 15 20 30 |",
    "|---|---|---|---|---|---|---|---|",
]


def weighted_turnbull(ivs: list[Any], weights: np.ndarray) -> np.ndarray:
    """NEW CODE (flagged): Turnbull EM with dataset weights (post-stratification)."""
    A = a9._membership(ivs, GRID)
    w = weights / weights.sum()
    p = np.full(A.shape[1], 1.0 / A.shape[1])
    for _ in range(5000):
        post = A * p
        new = (w[:, None] * post / post.sum(axis=1, keepdims=True)).sum(axis=0)
        if np.abs(new - p).max() < 1e-9:
            break
        p = new
    return np.asarray(new)


def c_bin(c: int) -> str:
    return "C2" if c == 2 else "C3-4" if c <= 4 else "C5-10"


# ------------------------------------------------------------------ part B
def part_b(
    raw: pd.DataFrame, traits: pd.DataFrame, members: pd.DataFrame, res: dict[str, Any]
) -> list[str]:
    curve = a9.curves(raw)
    curve_acc = a9.curves(raw, "accuracy")
    L: list[str] = ["# Part B: pre-specified extras (frozen functions; no confirmatory weight)", ""]
    ivs = intervals(curve, "fm2", "random")

    # H1-dist at the prereg's 2000 resamples, and sensitivities.
    L += ["## B1. H1-dist (FM2, random, macro-F1) with 2000 resamples", ""]
    L += DIST_HEAD + [dist_row("EXP' all", ivs.tolist(), 2000)]
    subsets: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "Tier 2a (PMLB numeric)": lambda m: m.subtier == "2a",
        "Tier 3 (PMLB int-coded cat.)": lambda m: m.subtier == "3",
        "Tier 4 (UCI)": lambda m: m.subtier == "4",
        "numeric-only (2a + 4)": lambda m: m.numeric.astype(bool),
        "balanced core": lambda m: m.balanced.astype(bool),
    }
    for label, rule in subsets.items():
        keys = members.index[rule(members)]
        L.append(dist_row(label, ivs[keys].tolist(), 2000))
    for cb in ("C2", "C3-4", "C5-10"):
        keys = traits.index[traits.C.map(c_bin) == cb]
        if len(keys) > 1:
            L.append(dist_row(f"{cb}", ivs[keys].tolist(), 2000))
    for eps in (0.005, 0.02):
        L.append(dist_row(f"eps = {eps}", intervals(curve, "fm2", "random", eps).tolist(), 2000))
    acc = intervals(curve_acc, "fm2", "random")
    L.append(dist_row("accuracy", acc.tolist(), 2000))
    for lr in ("fm1", "fm5", "fm3"):
        L.append(dist_row(f"{lr}, random", intervals(curve, lr, "random").tolist(), 2000))
    for st in STRATS[1:]:
        L.append(dist_row(f"fm2, {st}", intervals(curve, "fm2", st).tolist(), 2000))

    strata = members.loc[ivs.index, "stratum"]
    w = 1.0 / strata.map(strata.value_counts()).to_numpy(float)
    masses = weighted_turnbull(ivs.tolist(), w)
    cdf = np.cumsum(masses)
    L += [
        "",
        "Post-stratified (equal weight per stratum; weighted EM is new code, flagged): "
        + ", ".join(f"F({g})={cdf[i]:.2f}" for i, g in enumerate(GRID))
        + f"; median cell {fmt_iv(a9.quantile_cell(masses, GRID, 0.5))}",
        "",
    ]

    # Censoring counts per C bin.
    L += ["## B2. Censoring of m* (FM2, random) per C bin", ""]
    L += ["| C bin | n | left-censored (m* <= 1) | right-censored | full grid (n >= 60C) |", "|---|---|---|---|---|"]
    for cb in ("C2", "C3-4", "C5-10"):
        keys = traits.index[traits.C.map(c_bin) == cb]
        sub = ivs[keys]
        L.append(
            f"| {cb} | {len(sub)} | {sum(lo == 0 for lo, _ in sub)} | "
            f"{sum(hi == inf for _, hi in sub)} | "
            f"{int((traits.loc[keys].n >= 60 * traits.loc[keys].C).sum())} |"
        )
    L.append("")

    # AFT for other learners and strategies.
    L += ["## B3. H1b AFT replications (beta_1 = log C)", ""]
    L += ["| learner, strategy | n | beta_1 | SE | p (beta_1 > 0) | beta_IR | beta_n | beta_d |", "|---|---|---|---|---|---|---|---|"]
    for lr, st in [("fm1", "random"), ("fm5", "random"), ("fm3", "random"),
                   ("fm2", "stratified"), ("fm2", "kmedoids_nested"), ("fm2", "lucos_nested")]:
        iv = intervals(curve, lr, st)
        cov = traits.loc[iv.index, ["C", "IR", "n", "d"]].astype(float).apply(np.log).add_prefix("log_")
        fit = a9.aft_fit(iv.tolist(), cov)
        L.append(
            f"| {lr}, {st} | {len(iv)} | {fit.loc['log_C','coef']:.3f} | {fit.loc['log_C','se']:.3f} | "
            f"{fit.loc['log_C','p_greater']:.4f} | {fit.loc['log_IR','coef']:.3f} | "
            f"{fit.loc['log_n','coef']:.3f} | {fit.loc['log_d','coef']:.3f} |"
        )
    L.append("")

    # ICC secondaries.
    full = traits.index[traits.n >= 60 * traits.C]
    cats = {lr: intervals(curve, lr, "random") for lr in LEARNERS}
    keys = [k for k in full if all(k in cats[lr].index for lr in LEARNERS)]
    Y4 = np.array([[a9.category(cats[lr][k], GRID) for lr in LEARNERS] for k in keys], float)
    icc4 = a9.icc_decision(Y4)
    L += ["## B4. H1a secondaries", "", f"- Four-FM ICC(2,1), full-grid datasets: {fmt_dict(icc4)}"]
    pairs = [("fm2", "fm5"), ("fm2", "fm3"), ("fm1", "fm5")]
    for a, b in pairs:
        Y = np.array([[a9.category(cats[a][k], GRID), a9.category(cats[b][k], GRID)] for k in keys], float)
        L.append(f"- ICC {a} vs {b}, full grid: {fmt_dict(a9.icc_decision(Y))}")
    lower = [
        [0 if cats[lr][k][0] == 0 else GRID.index(int(cats[lr][k][0])) + 1 for lr in ("fm2", "fm1")]
        for k in ivs.index
    ]
    L.append(
        "- FLAGGED operationalisation: ICC fm2 vs fm1 on all datasets with the category "
        f"of the lower interval end: {fmt_dict(a9.icc_decision(np.array(lower, float)))}"
    )
    L.append("")

    # Coverage floor and selection ceiling.
    m95 = traits.m95
    above = sum(lo >= m95[k] for k, (lo, hi) in ivs.items())
    below = sum(hi < m95[k] for k, (lo, hi) in ivs.items())
    L += [
        "## B5. Coverage floor and selection ceiling", "",
        f"- Floor: m* >= m95 decided yes for {above}, decided no for {below}, "
        f"undecided {len(ivs) - above - below} of {len(ivs)} (m95 median {m95.median():.2f}).",
    ]
    for st, d in res["h1"]["ceiling"].items():
        L.append(
            f"- Ceiling {st}: median cell {fmt_iv(d['median'])}, "
            + ", ".join(f"F({g:g})={v:.2f}" for g, v in d["cdf"].items())
            + f"; P(m_sel*>20) = {d['P(m*>20)'][0]:.2f}"
        )
    L.append("")

    # H2a magnitude with bootstrap CI, accuracy, FM1.
    rnd = a9._xs(curve, "fm2", "random")
    L += ["## B6. H2a magnitude: Turnbull median of rho_S(1), 2000-resample CI", ""]
    for st in ("kmedoids_nested", "lucos_nested", "stratified"):
        rho = a9._rho_intervals(a9._xs(curve, "fm2", st), rnd)[1]
        med = a9.turnbull_median_ratio(rho)
        lo, hi = a9.boot_ci(rho, a9.turnbull_median_ratio)
        if st == "stratified":
            keep = hi >= 2.0 and lo <= 3.2
            rule = "CI intersects [2.0, 3.2]"
        else:
            keep = hi >= 1.6 and lo <= 2.5 and lo > 1.2
            rule = "CI intersects [1.6, 2.5] and lower > 1.2"
        right = sum(h == inf for _, h in rho)
        left = sum(l == 0 for l, _ in rho)
        L.append(
            f"- {st}: median {med:.3f} [{lo:.3f}, {hi:.3f}]; {rule}: "
            f"{'retained' if keep else 'not retained'}; left-censored {left}, right-censored {right}, n {len(rho)}"
        )
    L += ["", "H2 in accuracy (FM2):", ""]
    for st, row in a9.h2(curve_acc).items():
        L.append(f"- {st}: {fmt_dict(row)}")
    L += ["", "## B7. FM-robustness: FM1 replication of the Holm family (unadjusted)", ""]
    h2_fm1 = a9.h2(curve, "fm1")
    iv1 = intervals(curve, "fm1", "random")
    cov1 = traits.loc[iv1.index, ["C", "IR", "n", "d"]].astype(float).apply(np.log).add_prefix("log_")
    aft1 = a9.aft_fit(iv1.tolist(), cov1)
    L.append(f"- H1b fm1: beta_1 {aft1.loc['log_C','coef']:.3f}, p {aft1.loc['log_C','p_greater']:.4g}")
    for st, row in h2_fm1.items():
        L.append(f"- H2 fm1 {st}: {fmt_dict(row)}")
    t1 = a9.stop_table(raw[raw.learner == "fm1"], raw[raw.learner == "fm2"])
    for signal in ("agree", "fm_disagree"):
        L.append(f"- H4 fm1 primary, {signal}: {fmt_dict(a9.mae_ratio(t1, signal))}")
    L.append("")

    # H3 descriptives.
    ir_bin = pd.cut(traits.IR, [0, 2, 5, inf], right=False, labels=["IR<2", "IR 2-5", "IR>5"])
    L += ["## B8. H3 descriptives", ""]
    L += ["| IR bin | n | unsafe (m95_hat < m95/1.5) | over (> 1.5x) | median p_cov at m95_hat |", "|---|---|---|---|---|"]
    for b, g in traits.groupby(ir_bin, observed=True):
        L.append(
            f"| {b} | {len(g)} | {(g.log_err < -np.log(1.5)).mean():.2f} | "
            f"{(g.log_err > np.log(1.5)).mean():.2f} | {g.p_cov_at_hat.median():.3f} |"
        )
    L += [
        "",
        f"- p_cov at m95_hat: median {traits.p_cov_at_hat.median():.3f}, share >= 0.95: "
        f"{(traits.p_cov_at_hat >= 0.95).mean():.2f}, min {traits.p_cov_at_hat.min():.3f}",
        f"- p_min sensitivity: achieved p_cov with rarest share x1.5: median "
        f"{traits['p_cov_at_pmin_x1.5'].median():.3f} (min {traits['p_cov_at_pmin_x1.5'].min():.3f}); "
        f"x2: median {traits['p_cov_at_pmin_x2.0'].median():.3f} (min {traits['p_cov_at_pmin_x2.0'].min():.3f})",
        f"- closed-form m95 (with replacement) vs exact k-means m95_hat: median ratio "
        f"{(traits.m95_closed_hat / traits.m95_hat).median():.3f}",
        "",
    ]
    L += validity_check(raw, members, traits)
    return L


def validity_check(raw: pd.DataFrame, members: pd.DataFrame, traits: pd.DataFrame) -> list[str]:
    """H3c validity: exact p_cov(mC) against the realized full-coverage rate (10 S-RR draws)."""
    sub = raw[(raw.learner == "fm2") & (raw.strategy == "random")]
    pred, real = [], []
    for key, did in members.did.items():
        data = load(int(did), settings.data_cache_dir)
        perm = row_permutation(len(data.y), int(did), settings.base_seed)
        y = data.y[perm]
        s, _ = layout(len(y))
        C = int(y.max()) + 1
        counts = np.bincount(y[s:], minlength=C).tolist()
        g = sub[sub.dataset == key]
        for m, gm in g.groupby("m"):
            pred.append(a9.p_all_at_least(counts, int(m) * C, 1))
            real.append(float((gm.coverage >= 1.0).mean()))
    pred_a, real_a = np.array(pred), np.array(real)
    return [
        "- Validity: exact p_cov(mC) vs realized full-coverage share of the 10 nested "
        f"random draws over {len(pred_a)} (dataset, m) cells: mean predicted {pred_a.mean():.3f}, "
        f"mean realized {real_a.mean():.3f}, mean |diff| {np.abs(pred_a - real_a).mean():.3f}, "
        f"corr {np.corrcoef(pred_a, real_a)[0, 1]:.3f}",
        "",
    ]


def fmt_dict(d: dict[str, Any]) -> str:
    def f(v: Any) -> str:
        if isinstance(v, dict):
            return "{" + ", ".join(f"{k}: {f(x)}" for k, x in v.items()) + "}"
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.4g}"
        return str(v)

    return f(d)


def main() -> None:
    check_freeze()
    members = pool_members()
    raw = a9.load_results(settings.results_dir / "study9" / "t1prime")
    if set(raw.dataset) != set(members.index):
        raise SystemExit("result keys differ from pool_t1prime.yaml")
    keys = {k: int(d) for k, d in members.did.items()}
    traits = a9.dataset_traits(
        keys, lambda did: load(did, settings.data_cache_dir), settings.base_seed
    )
    res = a9.analyze(raw[raw.dataset.isin(keys)], traits)
    title = f"Study 9 analysis, set t1prime = EXP' ({len(keys)} datasets, 4 FMs)"
    part_a = a9.report(res, title)
    named = traits.join(members[["key", "name", "subtier", "stratum"]])
    text = (
        "# Part A: frozen analysis output (confirmatory), verbatim\n\n"
        + part_a
        + "\n"
        + named.round(3).to_markdown()
        + "\n\n"
        + "\n".join(part_b(raw, traits, members, res))
        + "\n"
    )
    OUT.write_text(text)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
