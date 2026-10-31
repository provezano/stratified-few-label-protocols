"""Shared helpers for the TabEBM case study (paper/p1/exploratory/case_study/PREREG.md).

EXPLORATORY. Reads only the downloaded OpenML ARFF files under third_party/ and the
cached stock table; writes only under case_study/out/. sealed_heldout/ is never touched.
Scripts that import this module are run with ``python -I`` and add this directory to
sys.path themselves.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CS = HERE.parent
TP = CS / "third_party"
OUT = CS / "out"
BASE_SEED = 20261028
N_REPS = 10  # Amendment 2 (owner, option (a))
N_SYN = 500
TABPFN_CKPT = Path.home() / "Library/Caches/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"

# Paper Table 3 targets for the reconstruction check (N, D, C)
PAPER_SHAPE = {
    40966: (1080, 77, 8),
    14: (2000, 76, 10),
    841: (950, 9, 2),
    1472: (698, 9, 23),
    40971: (970, 19, 26),
    40499: (5500, 40, 11),
}
NAMES = {40966: "protein", 14: "fourier", 841: "stock", 1472: "energy", 40971: "collins", 40499: "texture"}
# Sec. 2: the nine pre-declared few-label cells
CELLS = [(40966, 20), (40966, 50), (14, 20), (14, 50), (841, 20), (1472, 50), (1472, 100), (40971, 100), (40499, 50)]
PROTOCOLS = ("S", "R")
ARMS = ("baseline", "smote", "tabebm")
CLASSICAL = ("lr", "knn", "mlp", "rf", "xgb")
TABPFN_MAX_C = 10  # Amendment 1 (c): TabPFN v2 downstream only on tables with C <= 10


def ss_ints(*keys: int, n: int = 1) -> list[int]:
    """Integer seeds from SeedSequence([BASE_SEED, *keys])."""
    st = np.random.SeedSequence([BASE_SEED, *[int(k) for k in keys]]).generate_state(n)
    return [int(s) for s in st]


# ------------------------------------------------------------------ data
def _arff(path: Path):
    from scipy.io import arff

    data, meta = arff.loadarff(str(path))
    return data, meta


def _num(col) -> np.ndarray:
    a = np.asarray(col)
    if a.dtype.kind in "SO":
        a = np.array([np.nan if v in (b"?", "?") else float(v) for v in a])
    return a.astype(float)


def load_raw(did: int) -> tuple[np.ndarray, np.ndarray]:
    """Features and string labels of one table, before the paper's class filter."""
    if did == 841:
        # P1's cached copy of DEV table 841 (stock v2), read without pickle
        p = Path(__file__).resolve().parents[5] / "data/cache/npz/841.npz"
        d = np.load(p, allow_pickle=False)
        return d["X"].astype(float), d["y"].astype(str)
    files = {40966: "MiceProtein", 14: "mfeat-fourier", 1472: "energy-efficiency", 40971: "collins", 40499: "texture"}
    data, meta = _arff(TP / f"openml_{did}" / f"{files[did]}.arff")
    names = list(meta.names())
    target = {40966: "class", 14: "class", 1472: "y1", 40971: "Corp.Genre", 40499: "Class"}[did]
    drop = {40966: {"MouseID", "Genotype", "Treatment", "Behavior"}, 40971: {"Text", "Genre", "Counter", "Corpus"}}.get(did, set())
    feats = [c for c in names if c != target and c not in drop]
    X = np.column_stack([_num(data[c]) for c in feats])
    y = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in data[target]])
    return X, y


def load_table(did: int) -> tuple[np.ndarray, np.ndarray]:
    """Paper rule: drop classes with fewer than 10 rows; labels coded 0..C-1."""
    X, y = load_raw(did)
    vals, cnt = np.unique(y, return_counts=True)
    keep = set(vals[cnt >= 10])
    m = np.array([v in keep for v in y])
    X, y = X[m], y[m]
    _, yc = np.unique(y, return_inverse=True)
    return X, yc.astype(int)


# ------------------------------------------------------------------ splits
def test_split(X, y, did: int, r: int):
    from sklearn.model_selection import train_test_split

    n_test = int(min(len(y) // 2, 500))
    (s,) = ss_ints(did, r, 0)
    idx = np.arange(len(y))
    pool, test = train_test_split(idx, test_size=n_test, stratify=y, random_state=s % (2**32))
    return np.sort(pool), np.sort(test)


def per_class_val(idx: np.ndarray, y: np.ndarray, seed: int):
    """Amendment 1 (a): floor(0.2 n_k) rows of each class go to validation."""
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for k in np.unique(y[idx]):
        rows = rng.permutation(idx[y[idx] == k])
        nv = int(np.floor(0.2 * len(rows)))
        va += list(rows[:nv])
        tr += list(rows[nv:])
    return np.sort(np.array(tr, int)), np.sort(np.array(va, int))


def draw_subset(pool: np.ndarray, y: np.ndarray, did: int, n_real: int, r: int, protocol: str):
    """S: proportional stratified subsample; R: uniform random subsample (no labels)."""
    from sklearn.model_selection import train_test_split

    p = 1 if protocol == "S" else 2
    s1, s2 = ss_ints(did, r, n_real, p, n=2)
    if protocol == "S":
        sub, _ = train_test_split(pool, train_size=n_real, stratify=y[pool], random_state=s1 % (2**32))
    else:
        sub = np.random.default_rng(s1).choice(pool, size=n_real, replace=False)
    tr, va = per_class_val(np.sort(sub), y, s2)
    return tr, va


def preprocess(X, pool):
    """Mean imputation and z-scoring with label-free statistics of the training pool."""
    mu = np.nanmean(X[pool], axis=0)
    Xi = np.where(np.isnan(X), mu, X)
    sd = Xi[pool].std(axis=0)
    sd[sd == 0] = 1.0
    return ((Xi - mu) / sd).astype(np.float32)


def syn_counts(y_tr: np.ndarray, total: int = N_SYN) -> dict[int, int]:
    """N_syn split over the seen classes in proportion to their counts (largest remainder)."""
    ks, n = np.unique(y_tr, return_counts=True)
    q = total * n / n.sum()
    base = np.floor(q).astype(int)
    rem = total - base.sum()
    order = np.argsort(-(q - base), kind="stable")
    base[order[:rem]] += 1
    return {int(k): int(b) for k, b in zip(ks, base)}


# ------------------------------------------------------------------ metrics
def metrics(y_true, y_pred, C: int) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return dict(
        ba=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=list(range(C)), average="macro", zero_division=0)),
        acc=float(accuracy_score(y_true, y_pred)),
    )
