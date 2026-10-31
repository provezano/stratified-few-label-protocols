"""Mutual kNN graphs with RBF weights (de Sousa et al. 2013 conventions)."""

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial.distance import cdist

MIN_EDGE_WEIGHT = 1e-300


def zscore(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standardize columns on all points (label-free); constant columns map to 0."""
    std = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)


def _sorted_neighbours(
    X: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1, kind="stable")
    return D, order


def _bandwidth(D: NDArray[np.float64], order: NDArray[np.intp], k: int) -> float:
    """Mean k-th neighbour distance / 3; fallbacks for duplicated points.

    If every k-th neighbour is a duplicate, use the mean positive pairwise
    distance / 3; if all points coincide, use 1 (PILOT_I2.md, Amendment 3).
    """
    sigma = float(D[np.arange(len(D)), order[:, k - 1]].mean() / 3.0)
    if sigma > 0:
        return sigma
    positive = D[np.isfinite(D) & (D > 0)]
    return float(positive.mean() / 3.0) if positive.size else 1.0


def knn_bandwidth(X: NDArray[np.float64], k: int) -> float:
    """Return sigma = mean distance to the k-th nearest neighbour / 3."""
    return _bandwidth(*_sorted_neighbours(X), k)


def mutual_knn_graph(X: NDArray[np.float64], k: int) -> sparse.csr_matrix:
    """Build the symmetric mutual kNN graph with weights exp(-d^2 / 2 sigma^2).

    A vertex left without mutual neighbours is linked to its nearest neighbour.
    """
    n = len(X)
    D, order = _sorted_neighbours(X)
    k = min(k, n - 1)
    A = np.zeros((n, n), dtype=bool)
    A[np.arange(n)[:, None], order[:, :k]] = True
    M = A & A.T
    isolated = np.flatnonzero(~M.any(axis=1))
    M[isolated, order[isolated, 0]] = True
    M[order[isolated, 0], isolated] = True
    sigma = _bandwidth(D, order, k)
    rbf = np.exp(-(np.where(M, D, 0.0) ** 2) / (2.0 * sigma**2))
    # far-apart edges underflow to 0, which would silently drop them
    W = np.where(M, np.maximum(rbf, MIN_EDGE_WEIGHT), 0.0)
    return sparse.csr_matrix(W)
