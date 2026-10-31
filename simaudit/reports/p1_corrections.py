"""P1 protocol corrections (exploratory, DEV): analysis of Corrections 4 and 5.

Plan and criteria: reports/P1_PROTOCOL_CORRECTIONS_EXPLORATORY.md (committed
before any run). Reads Study 4 Stage A/B, Study 7 Parts (a)-(c) and the new FM2
runs of ``reports/p1_corrections_runs.py``; writes the generated tables to
reports/P1_PROTOCOL_CORRECTIONS_TABLES.md. No FM is called. DEV only.

Correction 4: realistic-protocol mean as a composition-weighted average
(post-stratification of random runs; design runs at chosen compositions).
Correction 5: posteriors re-weighted by an unlabeled-pool prior estimate (EM,
BBSE) over the context prior, against none / oracle / uniform targets.
"""

import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from simaudit.analysis.report import bootstrap_ci
from simaudit.data.openml_loader import load
from simaudit.settings import settings

OUT = Path(__file__).with_name("P1_PROTOCOL_CORRECTIONS_TABLES.md")
RES = settings.results_dir
NEW = RES / "p1_corrections"
C5_CACHE = NEW / "c5_runs.parquet"
C5_DOSE_CACHE = NEW / "c5_dose.parquet"
FMS = ("fm1", "fm2", "fm3", "fm5")
MULTS = (1, 3, 5, 10)
METRICS = ("macro_f1", "accuracy")
N_SUB = 500
EM_TOL, EM_ITER = 1e-8, 1000


def dev_ids() -> list[int]:
    from simaudit.cli import _dev_ids

    return _dev_ids()


def labels(did: int) -> np.ndarray:
    return load(did, settings.data_cache_dir).y


# ============================================================ metrics
def scores(y_true: np.ndarray, pred: np.ndarray, n_classes: int) -> dict[str, float]:
    """Macro-F1 over all classes (zero_division = 0) and accuracy."""
    cm = np.zeros((n_classes, n_classes))
    np.add.at(cm, (y_true, pred), 1)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return dict(macro_f1=float(f1.mean()), accuracy=float(tp.sum() / len(y_true)))


# ============================================================ Correction 5
def reweight(p: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """p * target / source, renormalised (columns are the seen classes)."""
    q = p * (target / source)
    return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-300)


def em_prior(p: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Saerens et al. (2002) EM estimate of the target prior."""
    prior = source.copy()
    for _ in range(EM_ITER):
        new = reweight(p, source, prior).mean(axis=0)
        if np.abs(new - prior).max() < EM_TOL:
            return new
        prior = new
    return prior


def bbse_prior(
    p_lab: np.ndarray, y_lab: np.ndarray, p_unl: np.ndarray, source: np.ndarray
) -> np.ndarray:
    """Soft BBSE (Lipton et al. 2018) with a resubstitution confusion matrix."""
    k = p_unl.shape[1]
    cond = np.stack([p_lab[y_lab == j].mean(axis=0) for j in range(k)], axis=1)
    target, *_ = np.linalg.lstsq(cond, p_unl.mean(axis=0), rcond=None)
    target = np.clip(target, 0, None)
    return target / target.sum() if target.sum() > 0 else source


def corrected_scores(
    proba: np.ndarray, y: np.ndarray, labeled: np.ndarray, context: np.ndarray
) -> dict[str, float]:
    """Scores of every correction for one run; ``context`` = counts the FM saw."""
    n_classes = proba.shape[1]
    rest = np.setdiff1d(np.arange(len(y)), labeled)
    seen = np.flatnonzero(context > 0)
    source = context[seen] / context.sum()
    p_unl = proba[rest][:, seen].astype(np.float64)
    p_unl /= np.maximum(p_unl.sum(axis=1, keepdims=True), 1e-300)
    out: dict[str, float] = {}
    names = ("none", "em", "bbse", "oracle", "uniform", "em_half")
    # One seen class: every correction is the constant prediction.
    targets: dict[str, np.ndarray] = {name: source for name in names}
    if len(seen) > 1:
        em = em_prior(p_unl, source)
        idx = {c: i for i, c in enumerate(seen)}
        lab_seen = labeled[np.isin(y[labeled], seen)]
        p_lab = proba[lab_seen][:, seen].astype(np.float64)
        p_lab /= np.maximum(p_lab.sum(axis=1, keepdims=True), 1e-300)
        y_lab = np.array([idx[c] for c in y[lab_seen]])
        oracle = np.bincount(y[rest], minlength=n_classes)[seen].astype(float)
        oracle = np.maximum(oracle, 0.5) / np.maximum(oracle, 0.5).sum()
        uniform = np.full(len(seen), 1 / len(seen))
        half = np.sqrt(uniform * em)
        targets.update(
            em=em,
            bbse=bbse_prior(p_lab, y_lab, p_unl, source),
            oracle=oracle,
            uniform=uniform,
            em_half=half / half.sum(),
        )
        out["em_l1"] = float(np.abs(em - oracle).sum())
        out["ctx_l1"] = float(np.abs(source - oracle).sum())
    for name, target in targets.items():
        pred = seen[reweight(p_unl, source, target).argmax(axis=1)]
        for metric, value in scores(y[rest], pred, n_classes).items():
            out[f"{name}_{metric}"] = value
    return out


def _context_counts(y: np.ndarray, labeled: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(y[labeled], minlength=n_classes)


def c5_part_a() -> pd.DataFrame:
    """Every Part (a) random/stratified run, all FMs, every correction (cached)."""
    if C5_CACHE.exists():
        return pd.read_parquet(C5_CACHE)
    rows = []
    for did in dev_ids():
        y = labels(did)
        n_classes = int(y.max()) + 1
        for fm in FMS:
            frame = pd.read_parquet(RES / "study7" / "a" / f"{did}_{fm}.parquet")
            frame = frame[frame.strategy.isin(["random", "stratified"])]
            for r in frame.itertuples():
                lab = np.asarray(r.labeled, dtype=np.int64)
                proba = np.asarray(r.proba, dtype=np.float64).reshape(len(y), n_classes)
                rec = corrected_scores(
                    proba, y, lab, _context_counts(y, lab, n_classes)
                )
                rows.append(
                    dict(
                        dataset=did,
                        fm=fm,
                        strategy=r.strategy,
                        mult=r.mult,
                        rep=r.rep,
                        stored_macro_f1=r.macro_f1,
                        stored_accuracy=r.accuracy,
                        **rec,
                    )
                )
        print(f"c5 part a: {did}", file=sys.stderr, flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(C5_CACHE, index=False)
    return out


def c5_part_c() -> pd.DataFrame:
    """Part (c) dose cells (fitted once per seed), every correction (cached)."""
    if C5_DOSE_CACHE.exists():
        return pd.read_parquet(C5_DOSE_CACHE)
    rows = []
    for did in dev_ids():
        y = labels(did)
        n_classes = int(y.max()) + 1
        for fm in FMS:
            frame = pd.read_parquet(RES / "study7" / "c" / f"{did}_{fm}.parquet")
            frame = frame[frame.feasible & (frame.arm != "anchor")]
            for r in frame.itertuples():
                lab = np.asarray(r.labeled, dtype=np.int64)
                proba = np.asarray(r.proba, dtype=np.float64).reshape(len(y), n_classes)
                rec = corrected_scores(proba, y, lab, np.asarray(r.counts))
                rows.append(
                    dict(
                        dataset=did,
                        fm=fm,
                        arm=r.arm,
                        share=r.share,
                        mult=r.mult,
                        rep=r.rep,
                        stored_pc_macro_f1=r.pc_macro_f1,
                        **rec,
                    )
                )
        print(f"c5 part c: {did}", file=sys.stderr, flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(C5_DOSE_CACHE, index=False)
    return out


def paired_summary(per_ds: pd.Series) -> str:
    """'mean [low, high] w=win-rate' over datasets."""
    values = per_ds.dropna().to_numpy()
    low, high = bootstrap_ci(values)
    return f"{values.mean():+.3f} [{low:+.3f}, {high:+.3f}] w={np.mean(values > 0):.2f}"


def c5_tables(runs: pd.DataFrame) -> list[str]:
    lines = ["## Correction 5: prior correction from the unlabeled pool", ""]
    for col in METRICS:
        bad = (runs[f"none_{col}"] - runs[f"stored_{col}"]).abs()
        lines.append(
            f"- Recomputed 'none' vs stored {col}: max |diff| = {bad.max():.2e}, "
            f"share of runs with |diff| > 1e-9: {np.mean(bad > 1e-9):.4f}"
        )
    cov = runs.dropna(subset=["em_l1"])
    prior_err = cov.groupby(["strategy", "mult"])[["em_l1", "ctx_l1"]].mean().round(3)
    lines += [
        "",
        "### Prior-estimation error (L1 to the true complement prior, seen classes)",
        "",
        prior_err.to_markdown(),
        "",
    ]
    corrections = ("em", "bbse", "oracle", "uniform", "em_half")
    for strategy in ("random", "stratified"):
        for col in METRICS:
            lines += [
                f"### Δ(corrected − none), {col}, {strategy} protocol "
                "(mean over datasets [paired bootstrap 95% CI], win-rate)",
                "",
            ]
            table = {}
            sub = runs[runs.strategy == strategy]
            for fm, mult in [(f, m) for f in FMS for m in MULTS]:
                cell = sub[(sub.fm == fm) & (sub.mult == mult)]
                by_ds = cell.groupby("dataset")
                row = {}
                for c in corrections:
                    delta = by_ds[f"{c}_{col}"].mean() - by_ds[f"none_{col}"].mean()
                    row[c] = paired_summary(delta)
                table[(fm, mult)] = row
            lines += [pd.DataFrame(table).T.to_markdown(), ""]
    lines += c5_gap_tables(runs, corrections)
    lines += c5_oracle_share(runs)
    return lines


def c5_gap_tables(runs: pd.DataFrame, corrections: tuple[str, ...]) -> list[str]:
    lines = []
    for col in METRICS:
        lines += [f"### Stratified − random gap after each correction, {col}", ""]
        table = {}
        means = runs.groupby(["fm", "mult", "strategy", "dataset"]).mean(
            numeric_only=True
        )
        for fm, mult in [(f, m) for f in FMS for m in MULTS]:
            strat = means.loc[(fm, mult, "stratified")]
            rand = means.loc[(fm, mult, "random")]
            table[(fm, mult)] = {
                c: paired_summary(strat[f"{c}_{col}"] - rand[f"{c}_{col}"])
                for c in ("none", *corrections)
            }
        lines += [pd.DataFrame(table).T.to_markdown(), ""]
    return lines


def c5_oracle_share(runs: pd.DataFrame) -> list[str]:
    lines = [
        "### Share of the oracle gain recovered by EM (ratio of mean gains, random)",
        "",
    ]
    sub = runs[runs.strategy == "random"]
    table = {}
    for fm, mult in [(f, m) for f in FMS for m in MULTS]:
        by_ds = (
            sub[(sub.fm == fm) & (sub.mult == mult)]
            .groupby("dataset")
            .mean(numeric_only=True)
        )
        row = {}
        for col in METRICS:
            em = (by_ds[f"em_{col}"] - by_ds[f"none_{col}"]).mean()
            orc = (by_ds[f"oracle_{col}"] - by_ds[f"none_{col}"]).mean()
            row[f"{col} em"] = f"{em:+.4f}"
            row[f"{col} oracle"] = f"{orc:+.4f}"
        table[(fm, mult)] = row
    return lines + [pd.DataFrame(table).T.to_markdown(), ""]


def c5_dose_tables(dose: pd.DataFrame) -> list[str]:
    lines = [
        "### Part (c) dose-response: macro-F1 drop from s = 0.5 to the largest "
        "feasible s (mean over datasets × mult; larger = steeper decline)",
        "",
    ]
    bad = (dose["uniform_macro_f1"] - dose["stored_pc_macro_f1"]).abs().max()
    lines.append(
        f"- Post-hoc uniform vs stored Study 7 pc_macro_f1: max |diff| = {bad:.2e}"
    )
    lines.append("")
    table = {}
    for (fm, arm), group in dose.groupby(["fm", "arm"]):
        row = {}
        for c in ("none", "uniform", "em", "oracle"):
            for col in METRICS:
                level = group.groupby(["dataset", "mult", "share"])[f"{c}_{col}"].mean()
                drops = []
                for (_, _), cell in level.groupby(level=[0, 1]):
                    shares = cell.index.get_level_values(2)
                    drops.append(
                        cell[shares == 0.5].iloc[0]
                        - cell[shares == shares.max()].iloc[0]
                    )
                row[f"{c} {col}"] = f"{np.mean(drops):+.3f}"
        table[(fm, arm)] = row
    return lines + [pd.DataFrame(table).T.to_markdown(), ""]


# ============================================================ Correction 4
def p_missed_exactly(counts: np.ndarray, budget: int) -> dict[frozenset[int], float]:
    """P(the set of missed classes is exactly M), by inclusion-exclusion."""
    n, k = int(counts.sum()), len(counts)
    total = comb(n, budget)
    p_avoid = {}
    for size in range(k + 1):
        for subset in combinations(range(k), size):
            p_avoid[frozenset(subset)] = (
                comb(n - int(counts[list(subset)].sum()), budget) / total
            )
    out = {}
    for missed in p_avoid:
        rest = [c for c in range(k) if c not in missed]
        value = 0.0
        for size in range(len(rest) + 1):
            for extra in combinations(rest, size):
                value += (-1) ** size * p_avoid[missed | frozenset(extra)]
        if value > 1e-15:
            out[missed] = value
    return out


def expected_d(counts: np.ndarray, budget: int) -> float:
    """E D under MVHyp (Prop. B1)."""
    n, k = counts.sum(), len(counts)
    pi = counts / n
    phi = (n - budget) / (n - 1)
    return float(((pi - 1 / k) ** 2).sum() + phi / budget * (1 - (pi**2).sum()))


def binary_missed_values(
    counts: np.ndarray, budget: int
) -> tuple[float, dict[str, float]]:
    """P(miss) and the probability-weighted constant-prediction scores (A7)."""
    n = int(counts.sum())
    p_miss, contrib = 0.0, {m: 0.0 for m in METRICS}
    for k in (0, 1):
        q = comb(n - int(counts[k]), budget) / comb(n, budget)
        p = (counts[1 - k] - budget) / (n - budget)
        p_miss += q
        contrib["macro_f1"] += q * p / (1 + p)
        contrib["accuracy"] += q * p
    return p_miss, contrib


def composition_features(y: np.ndarray, labeled: list[np.ndarray]) -> pd.DataFrame:
    n_classes = int(y.max()) + 1
    comps = np.stack(
        [np.bincount(y[np.asarray(lab)], minlength=n_classes) for lab in labeled]
    )
    budget = comps.sum(axis=1, keepdims=True)
    return pd.DataFrame(
        dict(
            n0=comps[:, 0],
            missed=(comps == 0).sum(axis=1),
            d=((comps / budget - 1 / n_classes) ** 2).sum(axis=1),
        )
    )


def random_pool(did: int, fm: str, y: np.ndarray) -> pd.DataFrame:
    """Every realistic random run of (dataset, FM): Study 4, Study 7 (a), new FM2."""
    s4 = (
        RES
        / ("study4_stage_a" if fm == "fm5" else "study4_stage_b")
        / (f"{did}.parquet" if fm == "fm5" else f"{did}_{fm}.parquet")
    )
    parts = []
    frame = pd.read_parquet(s4)
    parts.append(
        frame[(frame.strategy == "random") & (frame.learner == fm)].assign(src="s4")
    )
    frame = pd.read_parquet(RES / "study7" / "a" / f"{did}_{fm}.parquet")
    parts.append(frame[frame.strategy == "random"].assign(src="s7"))
    extra = NEW / "random" / f"{did}_{fm}.parquet"
    if extra.exists():
        parts.append(pd.read_parquet(extra).assign(src="new"))
    pool = pd.concat([p[["mult", "labeled", "src", *METRICS]] for p in parts])
    pool = pool.reset_index(drop=True)
    feats = composition_features(y, list(pool.labeled))
    return pd.concat([pool, feats], axis=1)


def stratified_pool(did: int, fm: str) -> pd.DataFrame:
    s4 = (
        RES
        / ("study4_stage_a" if fm == "fm5" else "study4_stage_b")
        / (f"{did}.parquet" if fm == "fm5" else f"{did}_{fm}.parquet")
    )
    frame = pd.read_parquet(s4)
    a = frame[(frame.strategy == "stratified") & (frame.learner == fm)]
    b = pd.read_parquet(RES / "study7" / "a" / f"{did}_{fm}.parquet")
    b = b[b.strategy == "stratified"]
    return pd.concat([a[["mult", *METRICS]], b[["mult", *METRICS]]])


class CellModel:
    """Exact composition law of one (dataset, mult)."""

    def __init__(self, y: np.ndarray, mult: int):
        self.counts = np.bincount(y).astype(np.int64)
        self.k = len(self.counts)
        self.budget = mult * self.k
        missed = p_missed_exactly(self.counts, self.budget)
        self.p_seen = np.zeros(self.k + 1)
        for m, p in missed.items():
            self.p_seen[self.k - len(m)] += p
        self.e_missed = float(sum(len(m) * p for m, p in missed.items()))
        self.e_d = expected_d(self.counts, self.budget)
        if self.k == 2:
            n, b = int(self.counts.sum()), self.budget
            self.pmf = hypergeom(n, int(self.counts[0]), b).pmf(np.arange(b + 1))
            self.p_miss, self.miss_contrib = binary_missed_values(self.counts, b)


def est_ps_seen(m: np.ndarray, seen: np.ndarray, model: CellModel) -> float:
    w, s = 0.0, 0.0
    for j in np.unique(seen):
        w += model.p_seen[j]
        s += model.p_seen[j] * m[seen == j].mean()
    return s / w


def est_ps_a7(m: np.ndarray, n0: np.ndarray, model: CellModel, metric: str) -> float:
    cov = (n0 > 0) & (n0 < model.budget)
    if not cov.any():
        return float(m.mean())
    return (1 - model.p_miss) * m[cov].mean() + model.miss_contrib[metric]


def g_interp(n0: np.ndarray, m: np.ndarray, model: CellModel, metric: str) -> float:
    """Composition-weighted mean with g interpolated over covered N_0 + exact misses."""
    cov = (n0 > 0) & (n0 < model.budget)
    if not cov.any():
        return float("nan")
    xs = np.unique(n0[cov])
    gs = np.array([m[cov][n0[cov] == x].mean() for x in xs])
    grid = np.arange(1, model.budget)
    return float(
        (model.pmf[grid] * np.interp(grid, xs, gs)).sum() + model.miss_contrib[metric]
    )


def greg_beta(
    pools: dict[int, pd.DataFrame], did: int, mult: int, metric: str
) -> np.ndarray:
    """Within-dataset OLS slope of M on (missed, D), pooled over the other datasets."""
    zs, ms = [], []
    for other, pool in pools.items():
        if other == did:
            continue
        cell = pool[pool.mult == mult]
        z = cell[["missed", "d"]].to_numpy(float)
        zs.append(z - z.mean(axis=0))
        ms.append(cell[metric].to_numpy() - cell[metric].mean())
    z, m = np.vstack(zs), np.concatenate(ms)
    beta, *_ = np.linalg.lstsq(z, m, rcond=None)
    return beta


def subsample_cell(
    cell: pd.DataFrame,
    model: CellModel,
    beta: np.ndarray,
    metric: str,
    k: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    """Errors (estimate − held-out truth) over N_SUB subsamples of size k."""
    m = cell[metric].to_numpy()
    seen = model.k - cell.missed.to_numpy()
    n0, z = cell.n0.to_numpy(), cell[["missed", "d"]].to_numpy(float)
    ez = np.array([model.e_missed, model.e_d])
    errs: dict[str, list[float]] = {"NAIVE": [], "PS-SEEN": [], "GREG": []}
    if model.k == 2:
        errs.update({"PS-A7": [], "PS-COMP": []})
    for _ in range(N_SUB):
        pick = rng.choice(len(m), k, replace=False)
        rest = np.setdiff1d(np.arange(len(m)), pick)
        truth = m[rest].mean()
        mp = m[pick]
        errs["NAIVE"].append(mp.mean() - truth)
        errs["PS-SEEN"].append(est_ps_seen(mp, seen[pick], model) - truth)
        errs["GREG"].append(mp.mean() - beta @ (z[pick].mean(axis=0) - ez) - truth)
        if model.k == 2:
            errs["PS-A7"].append(est_ps_a7(mp, n0[pick], model, metric) - truth)
            comp = g_interp(n0[pick], mp, model, metric)
            errs["PS-COMP"].append((comp if np.isfinite(comp) else mp.mean()) - truth)
    return errs


def c4_post_stratification(pools, models, strat) -> pd.DataFrame:
    rng = np.random.default_rng([settings.base_seed, 21, 4])
    rows = []
    for (did, fm), pool in pools.items():
        for mult in MULTS:
            cell = pool[pool.mult == mult].reset_index(drop=True)
            model = models[(did, mult)]
            for metric in METRICS:
                beta = greg_beta(
                    {d: p for (d, f), p in pools.items() if f == fm}, did, mult, metric
                )
                s2 = cell[metric].var(ddof=1)
                truth_all = cell[metric].mean()
                st = strat[(did, fm)]
                strat_gap = st[st.mult == mult][metric].mean() - truth_all
                for k in (5, 10, 20):
                    errs = subsample_cell(cell, model, beta, metric, k, rng)
                    for name, e in errs.items():
                        e = np.asarray(e)
                        rows.append(
                            dict(
                                dataset=did,
                                fm=fm,
                                mult=mult,
                                metric=metric,
                                k=k,
                                estimator=name,
                                bias=e.mean(),
                                mse=np.mean(e**2) - s2 / (len(cell) - k),
                                pool=len(cell),
                                strat_gap=strat_gap,
                                binary=model.k == 2,
                            )
                        )
        print(f"c4 ps: {did} {fm}", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


def rmse_table(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Bias [CI over datasets] and RMSE ratio to NAIVE (root of mean MSE)."""
    out = {}
    for key, group in frame.groupby(keys):
        naive = group[group.estimator == "NAIVE"].set_index("dataset")
        for name, est in group.groupby("estimator"):
            est = est.set_index("dataset")
            bias = est.groupby(level=0).bias.mean()
            low, high = bootstrap_ci(bias.to_numpy())
            mse = est.groupby(level=0).mse.mean().clip(lower=0)
            mse_n = naive.groupby(level=0).mse.mean().clip(lower=0)
            out[(*key, name)] = dict(
                bias=f"{bias.mean():+.4f} [{low:+.4f}, {high:+.4f}]",
                rmse=f"{np.sqrt(mse.mean()):.4f}",
                ratio=f"{np.sqrt(mse.mean() / mse_n.mean()):.2f}",
            )
    return pd.DataFrame(out).T


def c4_design_binary(pools, models, strat) -> pd.DataFrame:
    """DES-INTERP / STRAT+A7 on binary datasets from Part (c) + stratified runs."""
    rng = np.random.default_rng([settings.base_seed, 21, 5])
    rows = []
    for (did, fm), pool in pools.items():
        if models[(did, 1)].k != 2:
            continue
        y = labels(did)
        dose = pd.read_parquet(RES / "study7" / "c" / f"{did}_{fm}.parquet")
        dose = dose[dose.feasible].drop_duplicates("seed")
        st7 = pd.read_parquet(RES / "study7" / "a" / f"{did}_{fm}.parquet")
        st7 = st7[st7.strategy == "stratified"]
        for mult in MULTS:
            model = models[(did, mult)]
            design = pd.concat(
                [
                    dose[dose.mult == mult][["labeled", *METRICS]],
                    st7[st7.mult == mult][["labeled", *METRICS]],
                ]
            ).reset_index(drop=True)
            design["n0"] = composition_features(y, list(design.labeled)).n0
            cell = pool[pool.mult == mult]
            st = strat[(did, fm)]
            st = st[st.mult == mult]
            for metric in METRICS:
                truth, s2 = cell[metric].mean(), cell[metric].var(ddof=1)
                strat_a7 = (1 - model.p_miss) * st[metric].mean() + model.miss_contrib[
                    metric
                ]
                groups = [g[metric].to_numpy() for _, g in design.groupby("n0")]
                xs = np.array(sorted(design.n0.unique()))
                for r in (1, 2, 5):
                    if min(len(g) for g in groups) < r:
                        continue
                    n_runs = r * len(groups)
                    errs, naive = [], []
                    for _ in range(N_SUB):
                        gs = [rng.choice(g, r, replace=False).mean() for g in groups]
                        grid = np.arange(1, model.budget)
                        est = (
                            model.pmf[grid] * np.interp(grid, xs, gs)
                        ).sum() + model.miss_contrib[metric]
                        errs.append(est - truth)
                        if n_runs < len(cell):
                            pick = rng.choice(len(cell), n_runs, replace=False)
                            rest = np.setdiff1d(np.arange(len(cell)), pick)
                            naive.append(
                                cell[metric].to_numpy()[pick].mean()
                                - cell[metric].to_numpy()[rest].mean()
                            )
                    errs = np.asarray(errs)
                    row = dict(
                        dataset=did,
                        fm=fm,
                        mult=mult,
                        metric=metric,
                        r=r,
                        n_runs=n_runs,
                        des_bias=errs.mean(),
                        des_mse=np.mean(errs**2) - s2 / len(cell),
                        strat_a7_bias=strat_a7 - truth,
                        strat_bias=st[metric].mean() - truth,
                    )
                    if naive:
                        row["naive_mse"] = np.mean(np.square(naive)) - s2 / (
                            len(cell) - n_runs
                        )
                    rows.append(row)
    return pd.DataFrame(rows)


def c4_design_multiclass(pools) -> pd.DataFrame:
    """DES-STRAT on the new FM2 multiclass design runs."""
    rng = np.random.default_rng([settings.base_seed, 21, 6])
    rows = []
    for path in sorted((NEW / "design").glob("*_fm2.parquet")):
        design = pd.read_parquet(path)
        did = int(design.dataset.iloc[0])
        pool = pools[(did, "fm2")]
        for mult in MULTS:
            cell = pool[pool.mult == mult]
            des = design[design.mult == mult]
            for metric in METRICS:
                truth, s2 = cell[metric].mean(), cell[metric].var(ddof=1)
                groups = [g[metric].to_numpy() for _, g in des.groupby("stratum")]
                for r in (1, 2, 4):
                    n_runs = r * len(groups)
                    errs, naive = [], []
                    for _ in range(N_SUB):
                        est = np.mean(
                            [rng.choice(g, r, replace=False).mean() for g in groups]
                        )
                        errs.append(est - truth)
                        pick = rng.choice(len(cell), n_runs, replace=False)
                        rest = np.setdiff1d(np.arange(len(cell)), pick)
                        naive.append(
                            cell[metric].to_numpy()[pick].mean()
                            - cell[metric].to_numpy()[rest].mean()
                        )
                    errs = np.asarray(errs)
                    rows.append(
                        dict(
                            dataset=did,
                            mult=mult,
                            metric=metric,
                            r=r,
                            n_runs=n_runs,
                            des_bias=errs.mean(),
                            des_mse=np.mean(errs**2) - s2 / len(cell),
                            naive_mse=np.mean(np.square(naive))
                            - s2 / (len(cell) - n_runs),
                        )
                    )
    return pd.DataFrame(rows)


def design_summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = {}
    for key, g in frame.groupby(keys):
        bias = g.groupby("dataset").des_bias.mean().to_numpy()
        low, high = bootstrap_ci(bias)
        row = dict(
            n_runs=int(g.n_runs.median()),
            des_bias=f"{bias.mean():+.4f} [{low:+.4f}, {high:+.4f}]",
            des_rmse=f"{np.sqrt(g.des_mse.clip(lower=0).mean()):.4f}",
        )
        if "naive_mse" in g and g.naive_mse.notna().all():
            row["naive_rmse"] = f"{np.sqrt(g.naive_mse.clip(lower=0).mean()):.4f}"
            row["ratio"] = (
                f"{np.sqrt(g.des_mse.clip(lower=0).mean() / g.naive_mse.clip(lower=0).mean()):.2f}"
            )
        for extra in ("strat_bias", "strat_a7_bias"):
            if extra in g:
                vals = g.groupby("dataset")[extra].mean().to_numpy()
                lo, hi = bootstrap_ci(vals)
                row[extra] = f"{vals.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
        out[key] = row
    return pd.DataFrame(out).T


def c4_section() -> list[str]:
    pools, models, strat = {}, {}, {}
    for did in dev_ids():
        y = labels(did)
        for mult in MULTS:
            models[(did, mult)] = CellModel(y, mult)
        for fm in FMS:
            pools[(did, fm)] = random_pool(did, fm, y)
            strat[(did, fm)] = stratified_pool(did, fm)
    sizes = pd.Series({k: len(p) // len(MULTS) for k, p in pools.items()})
    lines = [
        "## Correction 4: composition-weighted estimator",
        "",
        f"- Random runs per (dataset, FM, mult): {sizes.value_counts().to_dict()}",
        "",
    ]
    ps = c4_post_stratification(pools, models, strat)
    ps.to_parquet(NEW / "c4_ps.parquet", index=False)
    for metric in METRICS:
        sub = ps[ps.metric == metric]
        lines += [
            f"### (a) Post-stratification, {metric}: bias [95% CI over datasets], "
            "RMSE, RMSE ratio to NAIVE (equal runs k)",
            "",
            rmse_table(sub, ["fm", "k"]).to_markdown(),
            "",
        ]
        lines += [
            f"#### by budget (k = 10), {metric}",
            "",
            rmse_table(sub[sub.k == 10], ["mult", "fm"]).to_markdown(),
            "",
        ]
        lines += [
            f"#### binary vs multiclass (k = 10), {metric}",
            "",
            rmse_table(sub[sub.k == 10], ["binary", "fm"]).to_markdown(),
            "",
        ]
    gap = ps[(ps.k == 10) & (ps.estimator == "NAIVE")].groupby(["metric", "fm", "mult"])
    gap = gap.strat_gap.mean().unstack().round(3)
    lines += [
        "### The bias being corrected: stratified mean − truth (mean over datasets)",
        "",
        gap.to_markdown(),
        "",
    ]
    db = c4_design_binary(pools, models, strat)
    db.to_parquet(NEW / "c4_design_binary.parquet", index=False)
    for metric in METRICS:
        lines += [
            f"### (b) Design, binary (14 datasets): DES-INTERP vs NAIVE at equal runs, "
            f"{metric}",
            "",
            design_summary(db[db.metric == metric], ["fm", "mult", "r"]).to_markdown(),
            "",
        ]
    if (NEW / "design").exists():
        dm = c4_design_multiclass(pools)
        dm.to_parquet(NEW / "c4_design_multiclass.parquet", index=False)
        for metric in METRICS:
            lines += [
                f"### (b) Design, multiclass (FM2, new runs): DES-STRAT vs NAIVE, "
                f"{metric}",
                "",
                design_summary(dm[dm.metric == metric], ["mult", "r"]).to_markdown(),
                "",
            ]
    return lines


def main() -> None:
    parts = sys.argv[1:] or ["c5", "c4"]
    lines = [
        "# P1 protocol corrections: generated tables (exploratory, DEV)",
        "",
        "Generated by `reports/p1_corrections.py`; plan and criteria in "
        "reports/P1_PROTOCOL_CORRECTIONS_EXPLORATORY.md.",
        "",
    ]
    if "c4" in parts:
        lines += c4_section()
    if "c5" in parts:
        lines += c5_tables(c5_part_a())
        lines += c5_dose_tables(c5_part_c())
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
