"""Pre-registered data-integrity screens (Study 8 Amendment 2, A2.6).

Three problems found in the DEV audit are screened on every expansion dataset,
on the stored (unpermuted) row order and before any model runs:
rows sorted by class (flag only: runners permute rows), ID-like or counter
columns that recover the class (dropped), and duplicate rows with conflicting
labels (dataset excluded above a pre-set share). Thresholds are fixed here.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

KAPPA_SORTED = 0.5
MONOTONE_RHO = 0.99
UNIQUE_SHARE = 0.95
LEAK_BACC = 0.90
CONFLICT_SHARE = 0.05
# A2.3a: datasets sharing at least this share of rows are duplicates.
OVERLAP = 0.5
MIN_COLUMNS = 2

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def class_sort_kappa(y: IntArray) -> float:
    """Chance-corrected agreement of adjacent rows' classes (1 = sorted by class)."""
    if len(y) < 2:
        return 0.0
    adjacent = float(np.mean(y[1:] == y[:-1]))
    p = np.bincount(y) / len(y)
    chance = float(np.sum(p**2))
    return 0.0 if chance >= 1.0 else (adjacent - chance) / (1.0 - chance)


def loo_1nn_balanced_accuracy(x: FloatArray, y: IntArray) -> float:
    """Leave-one-out 1-NN balanced accuracy using one column (ties: lower value)."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    gaps = np.diff(xs)
    left = np.concatenate([[np.inf], gaps])
    right = np.concatenate([gaps, [np.inf]])
    pos = np.arange(len(xs))
    pred = ys[np.where(left <= right, pos - 1, pos + 1)]
    recalls = [np.mean(pred[ys == c] == c) for c in np.unique(ys)]
    return float(np.mean(recalls))


def _monotone(x: FloatArray) -> bool:
    if np.ptp(x) == 0:
        return False
    rho = spearmanr(x, np.arange(len(x))).statistic
    return bool(np.isfinite(rho) and abs(rho) >= MONOTONE_RHO)


def _unique_integers(x: FloatArray) -> bool:
    integer = bool(np.all(x == np.round(x)))
    return integer and len(np.unique(x)) >= UNIQUE_SHARE * len(x)


def id_like_columns(X: FloatArray) -> list[int]:
    """Columns monotone in the stored row index, or near-unique integers."""
    return [
        j for j in range(X.shape[1]) if _monotone(X[:, j]) or _unique_integers(X[:, j])
    ]


def leaky_columns(X: FloatArray, y: IntArray) -> list[int]:
    """ID-like columns whose LOO 1-NN balanced accuracy alone is >= LEAK_BACC."""
    return [
        j
        for j in id_like_columns(X)
        if loo_1nn_balanced_accuracy(X[:, j], y) >= LEAK_BACC
    ]


def conflicting_duplicates(X: FloatArray, y: IntArray) -> tuple[int, int, int]:
    """(extra duplicate rows, groups with >1 label, rows in those groups)."""
    _, inverse, sizes = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    dup = int(np.sum(sizes[sizes > 1] - 1))
    labels = {(int(g), int(c)) for g, c in zip(inverse, y)}
    per_group = np.bincount([g for g, _ in labels], minlength=len(sizes))
    conflict = per_group > 1
    return dup, int(conflict.sum()), int(sizes[conflict].sum())


@dataclass(frozen=True)
class ScreenResult:
    """Outcome of the three screens for one dataset."""

    kappa: float
    class_sorted: bool
    id_like: tuple[int, ...]
    drop_columns: tuple[int, ...]
    dup_rows: int
    conflict_groups: int
    conflict_rows: int
    conflict_share: float
    decision: str

    def as_row(self) -> dict[str, Any]:
        """Flat audit-CSV row (column lists joined by ';')."""
        row = asdict(self)
        for key in ("id_like", "drop_columns"):
            row[key] = ";".join(map(str, row[key]))
        return row


def screen(X: FloatArray, y: IntArray) -> ScreenResult:
    """Apply the screens in stored row order and return the pre-set decision."""
    kappa = class_sort_kappa(y)
    leaky = tuple(leaky_columns(X, y))
    dup, groups, rows = conflicting_duplicates(X, y)
    share = rows / len(y)
    if share > CONFLICT_SHARE:
        decision = "exclude: conflicting duplicates"
    elif X.shape[1] - len(leaky) < MIN_COLUMNS:
        decision = "exclude: fewer than 2 columns left"
    else:
        decision = "drop columns" if leaky else "keep"
    return ScreenResult(
        kappa=round(kappa, 4),
        class_sorted=kappa >= KAPPA_SORTED,
        id_like=tuple(id_like_columns(X)),
        drop_columns=leaky,
        dup_rows=dup,
        conflict_groups=groups,
        conflict_rows=rows,
        conflict_share=round(share, 4),
        decision=decision,
    )


def row_overlap(A: FloatArray, B: FloatArray) -> float:
    """Share of A's unique rows that also occur in B (0 if the widths differ)."""
    if A.shape[1] != B.shape[1]:
        return 0.0
    a = {r.tobytes() for r in np.ascontiguousarray(A)}
    b = {r.tobytes() for r in np.ascontiguousarray(B)}
    return len(a & b) / len(a)


SIG_DIGITS = 6


def _significant(X: FloatArray, digits: int = SIG_DIGITS) -> FloatArray:
    """X rounded to ``digits`` significant digits (zeros stay zero)."""
    with np.errstate(divide="ignore"):
        exp = np.floor(np.log10(np.abs(X)))
    scale = np.where(np.isfinite(exp), 10.0 ** (digits - 1 - exp), 1.0)
    return np.asarray(np.round(X * scale) / scale, dtype=np.float64)


def _dense_ranks(X: FloatArray) -> FloatArray:
    """Each column replaced by the dense rank of its values (coding-invariant)."""
    cols = [np.unique(X[:, j], return_inverse=True)[1] for j in range(X.shape[1])]
    return np.column_stack(cols).astype(np.float64)


def overlap_share(A: FloatArray, B: FloatArray) -> float:
    """Share of A's unique rows found in B, after rounding or after dense ranking.

    Rounding to SIG_DIGITS significant digits absorbs differences in how two
    sources print the same floats; dense ranks absorb integer recodings of the
    same discrete grid. The larger of the two shares is returned (A2.3b).
    """
    if A.shape[1] != B.shape[1]:
        return 0.0
    by_value = row_overlap(_significant(A), _significant(B))
    return max(by_value, row_overlap(_dense_ranks(A), _dense_ranks(B)))
