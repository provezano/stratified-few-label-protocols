"""LLGC (Zhou et al. 2004) and GFHF (Zhu et al. 2003) label propagation."""

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.sparse.csgraph import connected_components


def normalized_affinity(W: sparse.csr_matrix) -> NDArray[np.float64]:
    """Return S = D^-1/2 W D^-1/2 as a dense matrix (zero rows for isolated nodes)."""
    d = np.asarray(W.sum(axis=1)).ravel()
    inv = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
    return np.asarray(W.toarray() * inv[:, None] * inv[None, :])


def llgc(
    W: sparse.csr_matrix,
    Y0: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """Solve F = (1 - alpha)(I - alpha S)^-1 Y0 in closed form (Cholesky)."""
    S = normalized_affinity(W)
    A = np.eye(len(S)) - alpha * S
    factor = linalg.cho_factor(A)
    return np.asarray((1.0 - alpha) * linalg.cho_solve(factor, Y0))


def gfhf(
    W: sparse.csr_matrix,
    labeled: NDArray[np.int64],
    classes: NDArray[np.int64],
    n_classes: int,
    fallback: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], int]:
    """Harmonic solution solved per connected component.

    Components without a labeled point take the ``fallback`` rows (uniform if
    None). Returns the score matrix and how many points used the fallback.
    """
    n = W.shape[0]
    F = np.full((n, n_classes), 1.0 / n_classes)
    F[labeled] = np.eye(n_classes)[classes]
    is_labeled = np.zeros(n, dtype=bool)
    is_labeled[labeled] = True
    _, component = connected_components(W, directed=False)
    Wd = W.toarray()
    n_fallback = 0
    for c in np.unique(component):
        members = np.flatnonzero(component == c)
        lab = members[is_labeled[members]]
        unl = members[~is_labeled[members]]
        if unl.size == 0:
            continue
        if lab.size == 0:
            if fallback is not None:
                F[unl] = fallback[unl]
            n_fallback += unl.size
            continue
        L_uu = np.diag(Wd[unl].sum(axis=1)) - Wd[np.ix_(unl, unl)]
        F[unl] = _harmonic_solve(L_uu, Wd[np.ix_(unl, lab)] @ F[lab])
    return F, n_fallback


def _harmonic_solve(
    L_uu: NDArray[np.float64],
    B: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve L_uu F = B; least squares when floored weights make L_uu singular."""
    try:
        return np.asarray(linalg.solve(L_uu, B, assume_a="pos"))
    except linalg.LinAlgError:
        return np.asarray(linalg.lstsq(L_uu, B)[0])
