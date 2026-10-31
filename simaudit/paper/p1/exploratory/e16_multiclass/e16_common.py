"""E16 shared helpers (exploratory). Reads only e16_multiclass/out/{draws,tables}.parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SEED = 20261027
N_BOOT = 2000
FMS = ("fm1", "fm2", "fm3", "fm5")
FM_NAMES = dict(fm1="TabPFN v2", fm2="TabICL", fm3="Kumo", fm5="TabPFN v3.5")

# ---------------------------------------------------------------- ordinal / nominal rule
# Rule (documented in e16_report.md): a multiclass target is ORDINAL when its documented class
# labels are ordered levels of one underlying quantity (a grade, rating, functional level or
# progression); NOMINAL when the labels are unordered categories (species, digits, diseases,
# origins, waveforms, mine types, ...); UNKNOWN when the source documentation does not say what
# the class codes mean (they are then reported with the nominal group as a sensitivity only).
# ``rank`` maps the encoded class code (sorted target strings, as the loaders encode them) to
# its level; codes with the same level share a rank.
ORDINAL = {
    # strict: graded ratings
    "pmlb_570174309": dict(name="car_evaluation", tier="strict", rank=[0, 1, 2, 3]),  # unacc<acc<good<vgood
    "pmlb_1000000257": dict(name="User Knowledge Modeling", tier="strict",
                            rank=[3, 1, 2, 0, 0]),  # High, Low, Middle, 'Very Low', 'very_low'
    # broad: ordered states / progression
    "pmlb_1000000697": dict(name="Students' Dropout", tier="broad", rank=[0, 1, 2]),  # Dropout<Enrolled<Graduate
    "pmlb_842065541": dict(name="new_thyroid", tier="broad", rank=[1, 2, 0]),  # 1 normal, 2 hyper, 3 hypo
    "pmlb_434692359": dict(name="allbp", tier="broad", rank=[0, 2, 1]),  # decreased, increased, negative
}
UNKNOWN = {"pmlb_844895957": "cloud", "pmlb_403982248": "schizo", "dev_694": "diggle_table_a2"}


def label_kind(key: str, C: int) -> str:
    if C == 2:
        return "binary"
    if key in ORDINAL:
        return "ordinal_" + ORDINAL[key]["tier"]
    if key in UNKNOWN:
        return "unknown"
    return "nominal"


# ---------------------------------------------------------------- coverage
def log_comb(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(invalid="ignore"):
        out = gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)
    return np.where(b > a, -np.inf, out)


def q_hyp(counts, B: int) -> np.ndarray:
    c = np.asarray(counts, float)
    n = c.sum()
    return np.exp(log_comb(n - c, B) - log_comb(n, B))


# ---------------------------------------------------------------- loading
def load():
    D = pd.read_parquet(OUT / "draws.parquet")
    T = pd.read_parquet(OUT / "tables.parquet")
    T["kind"] = [label_kind(k, c) for k, c in zip(T.dataset, T.C)]
    T["mc"] = T.C > 2
    return D, T


def per_class(D: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    """Long frame: one row per (draw, class) with confusion-derived quantities."""
    tinfo = T.set_index("dataset")
    rec = []
    for i, r in enumerate(D.itertuples(index=False)):
        C = r.C
        cm = np.asarray(r.cm, float).reshape(C, C)
        N = np.asarray(r.N, int)
        seen = N > 0
        u = cm.sum(1)
        pp = cm.sum(0)
        tp = np.diag(cm)
        fp = pp - tp
        fn = u - tp
        fp_abs = cm[~seen].sum(0)  # predictions of class k on rows of unseen classes
        den = 2 * tp + fp + fn
        f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
        den2 = den - fp_abs
        f1_na = np.where(den2 > 0, 2 * tp / np.where(den2 > 0, den2, 1), 0.0)
        prec = np.where(pp > 0, tp / np.where(pp > 0, pp, 1), np.nan)
        pp2 = pp - fp_abs
        prec_na = np.where(pp2 > 0, tp / np.where(pp2 > 0, pp2, 1), np.nan)
        rec_ = np.where(u > 0, tp / np.where(u > 0, u, 1), np.nan)
        pi = np.asarray(tinfo.loc[r.dataset, "counts_pool"], float)
        pi = pi / pi.sum()
        rank = (-pi).argsort().argsort() + 1  # 1 = largest class
        soft = np.asarray(r.soft, float)
        U = u.sum()
        for k in range(C):
            rec.append((i, k, N[k], seen[k], u[k], tp[k], fp[k], fn[k], fp_abs[k], f1[k], f1_na[k], prec[k],
                        prec_na[k], rec_[k], pi[k], rank[k], soft[k], pp[k] / U, u[k] / U))
    cols = ["draw", "k", "N", "seen", "u", "tp", "fp", "fn", "fp_abs", "f1", "f1_na", "prec", "prec_na", "rec",
            "pi", "rank", "soft", "hard", "true_share"]
    L = pd.DataFrame(rec, columns=cols)
    meta = D[["set", "dataset", "fm", "strategy", "m", "rep", "C"]].reset_index(drop=True)
    meta["B"] = [int(np.sum(n)) for n in D.N]
    meta["n_missed"] = [int(np.sum(np.asarray(n) == 0)) for n in D.N]
    meta["covered"] = meta.n_missed == 0
    meta["macro_f1"] = D.macro_f1.to_numpy()
    return L.merge(meta, left_on="draw", right_index=True)


# ---------------------------------------------------------------- bootstrap over tables
def boot_mean(values: pd.Series, groups: pd.Series, seed: int = SEED, n: int = N_BOOT):
    """Mean of per-table means with a percentile cluster bootstrap over tables."""
    per = values.groupby(groups).mean().dropna()
    x = per.to_numpy()
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)), len(x)


def boot_frame(df: pd.DataFrame, cols, by="dataset", seed: int = SEED, n: int = N_BOOT) -> dict:
    """Means over tables of several per-table columns, bootstrapping tables jointly."""
    per = df.groupby(by)[list(cols)].mean()
    x = per.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    b = np.nanmean(x[idx], axis=1)
    out = dict(tables=len(x))
    for j, c in enumerate(cols):
        out[c] = float(np.nanmean(x[:, j]))
        out[c + "_lo"] = float(np.nanquantile(b[:, j], 0.025))
        out[c + "_hi"] = float(np.nanquantile(b[:, j], 0.975))
    return out
