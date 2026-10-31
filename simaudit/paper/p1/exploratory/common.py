"""Shared helpers for the P1 exploratory checks E1-E5 (DEV only, read-only inputs).

Inputs are read from results/, reports/ and the data cache; nothing outside
paper/p1/exploratory/ is written. sealed_heldout/ is never touched.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

from simaudit.analysis.report import bootstrap_ci, holm  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
S7 = settings.results_dir / "study7"
FMS = ("fm1", "fm2", "fm3", "fm5")
FM_NAMES = dict(fm1="TabPFN v2", fm2="TabICLv2", fm3="Kumo-Tabular", fm5="TabPFN v3.5")
MULTS = (1, 3, 5, 10)


def dev_ids() -> list[int]:
    """The 20 DEV datasets of pilot I2 (data/pool_i2.yaml)."""
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_i2.yaml").read_text())
    return [int(d["did"]) for d in pool["dev"]]


def dev_data(did: int):
    """Read-only load of one DEV table with the project loader."""
    return load(did, settings.data_cache_dir)


def wilcoxon_p(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if not np.any(x != 0):
        return 1.0
    return float(stats.wilcoxon(x).pvalue)


def summarize(x: np.ndarray) -> dict[str, float]:
    """Mean, bootstrap 95% CI (10k resamples, as in the paper), median, win rate, Wilcoxon."""
    x = np.asarray(x, float)
    lo, hi = bootstrap_ci(x)
    return dict(
        n=len(x),
        mean=float(x.mean()),
        ci_lo=lo,
        ci_hi=hi,
        median=float(np.median(x)),
        win=float(np.mean(x > 0)),
        p=wilcoxon_p(x),
    )


def add_holm(df: pd.DataFrame, col: str = "p", out: str = "p_holm") -> pd.DataFrame:
    df = df.copy()
    df[out] = holm(df[col].tolist())
    return df


def spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, seed: int = 0):
    """Spearman rho with a percentile bootstrap CI over tables and the exact-ish p."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = []
    for row in idx:
        xs, ys = x[row], y[row]
        if np.ptp(xs) == 0 or np.ptp(ys) == 0:
            continue
        boots.append(stats.spearmanr(xs, ys)[0])
    b = np.array(boots)
    return float(rho), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)), float(p)


def md(df: pd.DataFrame, digits: int = 4) -> str:
    return df.round(digits).to_markdown(index=False)

# Run order (from simaudit/):
#   .venv/bin/python paper/p1/exploratory/e1_classical.py
#   .venv/bin/python paper/p1/exploratory/e2_auc.py
#   .venv/bin/python paper/p1/exploratory/e3_multiclass_sim.py
#   .venv/bin/python paper/p1/exploratory/e4_gap_moderators.py
#   .venv/bin/python paper/p1/exploratory/e5_study9_bootstrap.py
#   .venv/bin/python paper/p1/exploratory/e6_accuracy_heterogeneity.py
#   .venv/bin/python paper/p1/exploratory/e7_common_eval.py
#   .venv/bin/python paper/p1/exploratory/e8_schema_aware.py
#   .venv/bin/python paper/p1/exploratory/e9_meta_analysis.py
#   .venv/bin/python paper/p1/exploratory/e10_split_and_pools.py   (needs e6)
#   .venv/bin/python paper/p1/exploratory/e11_study9_sensitivity.py
#   .venv/bin/python paper/p1/exploratory/e12_small_checks.py
#   .venv/bin/python paper/p1/exploratory/e13_fm_vs_classical.py   (needs e1, e6)
#   .venv/bin/python paper/p1/exploratory/make_exploratory_numbers.py


# ----------------------------------------------------------------- E6-E12 helpers
def pool_traits(y: np.ndarray) -> dict[str, float]:
    """Label-based pool descriptors: n, C, pi_min, IR, L1 and TV distance to uniform."""
    counts = np.bincount(y)
    C = len(counts)
    pi = counts / counts.sum()
    l1 = float(np.abs(pi - 1 / C).sum())
    return dict(n=int(len(y)), C=C, pi_min=float(pi.min()), IR=float(counts.max() / counts.min()),
                l1=l1, tv=l1 / 2, prior_dist=float(((pi - 1 / C) ** 2).sum()))


def load_a(did: int, fm: str, strategies=("stratified", "random"), proba: bool = False) -> pd.DataFrame:
    """Stored Study 7 Part (a) rows of one table and FM (probabilities only if asked)."""
    cols = ["dataset", "learner", "strategy", "mult", "rep", "seed", "n_classes", "labeled",
            "coverage", "macro_f1", "accuracy", "balanced_accuracy"] + (["proba"] if proba else [])
    df = pd.read_parquet(S7 / "a" / f"{did}_{fm}.parquet", columns=cols)
    return df[df.strategy.isin(strategies)].reset_index(drop=True)


def fast_scores(y_true: np.ndarray, pred: np.ndarray, C: int, labels=None) -> tuple[float, float, float]:
    """(macro-F1 over ``labels`` (default all C), accuracy, balanced accuracy over present classes).

    Same conventions as analysis.study7.metric_set (zero_division = 0)."""
    cm_ = np.bincount(y_true * C + pred, minlength=C * C).reshape(C, C)
    tp = np.diag(cm_).astype(float)
    t = cm_.sum(1).astype(float)
    p = cm_.sum(0).astype(float)
    den = t + p
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    labs = np.arange(C) if labels is None else np.asarray(labels)
    present = t > 0
    return (float(f1[labs].mean()), float(tp.sum() / len(y_true)),
            float((tp[present] / t[present]).mean()))
