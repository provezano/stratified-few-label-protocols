"""USL, RDSS and ZCore: label-free selectors re-implemented from the official code.

Study 4 Amendment 3 (prereg/STUDY4_LABEL_SELECTION.md, Sec. 8). Each selector
maps (X, budget, seed) to sorted, distinct row indices and sees only X.

- USL (Wang, Lian & Yu, ECCV 2022; TonyLianLong/UnsupervisedSelectiveLabeling
  @ceaab3c, MIT): kNN density, k-means with B centroids, and the regularised
  per-cluster selection of ``get_selection_with_reg`` (CIFAR code).
- RDSS (Shao et al., NeurIPS 2024; YanhuiAILab/RDSS @3d7bbd6, Apache-2.0):
  ``khsample`` with the Gaussian kernel, vectorised.
- ZCore (Griffin, Marks & Corso, WACV 2026; voxel51/zcore @f9ed952, no licence
  declared): coverage and redundancy scores over random 2-column subspaces.

No third-party code is copied here; the integration tests run the official
code from ``../third_party_src`` and compare the selections.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances

from simaudit.graph.knn import zscore
from simaudit.selection.strategies import Strategy, _check_budget

# USL: config.py defaults (used unchanged by cifar10_usl.yaml); K from ImageNet.
USL_K = 20
USL_KMEANS_ITERS = 200
USL_REG = dict(n_iter=10, w=0.5, momentum=0.9, horizon=4.0, alpha=0.5)
USL_FAR = 1e10
# RDSS: gaussiankernel default.
RDSS_GAMMA = 0.5
# ZCore: CLI defaults, identical to LUCoS's ZCoreUnderSampler.
ZCORE_N_SAMPLE = 1_000_000
ZCORE_SAMPLE_DIM = 2
ZCORE_REDUND_NN = 1000
ZCORE_REDUND_EXP = 4
ZCORE_BATCH = 1000


# ---------------------------------------------------------------------- USL


def usl_features(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Z-scored X / sqrt(d'): mean squared row norm 1, like USL's unit-norm features."""
    n_informative = max(1, int((X.std(axis=0) > 0).sum()))
    return np.asarray(zscore(X) / np.sqrt(n_informative))


def usl_knn_mean_sqdist(Z: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Mean squared distance to the k nearest rows, the row itself included."""
    D = cdist(Z, Z, "sqeuclidean")
    return np.asarray(np.sort(D, axis=1)[:, :k].mean(axis=1))


def _assign(
    Z: NDArray[np.float64], centroids: NDArray[np.float64], alive: NDArray[np.bool_]
) -> NDArray[np.int64]:
    D = ((Z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    D[:, ~alive] = np.inf
    return np.asarray(D.argmin(axis=1), dtype=np.int64)


def usl_kmeans(
    Z: NDArray[np.float64], k: int, seed: int, n_iter: int = USL_KMEANS_ITERS
) -> NDArray[np.int64]:
    """Lloyd's k-means as in USL's ``KMeans``: init at ``torch.randperm(n)[:k]``.

    Returns the labels of the last assignment step. A cluster that empties is
    dropped from later assignments (the official code gives it a NaN centroid).
    """
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    init = torch.randperm(len(Z), generator=generator)[:k].numpy()
    centroids = Z[init].astype(np.float64)
    alive = np.ones(k, dtype=bool)
    labels = _assign(Z, centroids, alive)
    for _ in range(n_iter):
        labels = _assign(Z, centroids, alive)
        counts = np.bincount(labels, minlength=k)
        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, Z)
        alive = counts > 0
        centroids[alive] = sums[alive] / counts[alive, None]
    return labels


def _usl_regulariser(
    Z: NDArray[np.float64],
    labels: NDArray[np.int64],
    picks: list[int],
    horizon: float | None,
    alpha: float,
) -> NDArray[np.float64]:
    """Sum over picks outside the row's cluster of 1 / (squared distance)^alpha."""
    sq = ((Z[:, None, :] - Z[picks][None, :, :]) ** 2).sum(axis=-1)
    sq = np.where(labels[:, None] == labels[picks][None, :], USL_FAR, sq)
    sq[sq == 0] = USL_FAR
    if horizon is not None:
        sq[sq >= horizon] = USL_FAR
    return np.asarray((1.0 / sq**alpha).sum(axis=1))


def _usl_rounds(
    Z: NDArray[np.float64],
    dist: NDArray[np.float64],
    labels: NDArray[np.int64],
    budget: int,
    n_iter: int,
    w: float,
    momentum: float,
    horizon: float | None,
    alpha: float,
) -> tuple[list[int], NDArray[np.float64]]:
    """The picks of the last round and the score they were chosen by."""
    with np.errstate(divide="ignore"):
        density = 1.0 / dist
    reg = np.zeros(len(Z))
    picks: list[int] = []
    score = density
    for _ in range(n_iter):
        score = density - w * reg
        picks = []
        for cluster in range(budget):
            if len(picks) == budget:
                break
            members = np.flatnonzero(labels == cluster)
            if len(members):
                picks.append(int(members[int(score[members].argmax())]))
        new = _usl_regulariser(Z, labels, picks, horizon, alpha)
        reg = reg * momentum + new * (1 - momentum)
    return picks, score


def usl_select_with_reg(
    Z: NDArray[np.float64],
    dist: NDArray[np.float64],
    labels: NDArray[np.int64],
    budget: int,
    n_iter: int = 10,
    w: float = 0.5,
    momentum: float = 0.9,
    horizon: float | None = 4.0,
    alpha: float = 0.5,
) -> NDArray[np.int64]:
    """Select as USL's ``get_selection_with_reg``: one row per non-empty cluster."""
    picks, _ = _usl_rounds(Z, dist, labels, budget, n_iter, w, momentum, horizon, alpha)
    return np.array(picks, dtype=np.int64)


def select_usl(X: NDArray[np.float64], budget: int, seed: int) -> NDArray[np.int64]:
    """USL with B centroids; missing picks (empty clusters) by the final score."""
    _check_budget(len(X), budget)
    Z = usl_features(X)
    dist = usl_knn_mean_sqdist(Z, min(USL_K, len(Z)))
    labels = usl_kmeans(Z, budget, seed)
    picks, score = _usl_rounds(Z, dist, labels, budget, **USL_REG)  # type: ignore
    if len(picks) < budget:
        rest = [i for i in np.argsort(-score, kind="stable") if i not in picks]
        picks += [int(i) for i in rest[: budget - len(picks)]]
    return np.sort(np.array(picks, dtype=np.int64))


# --------------------------------------------------------------------- RDSS


def rdss_alpha(budget: int) -> float:
    """Return RDSS's default trade-off alpha = 1 - 1/sqrt(budget)."""
    return float(1 - 1 / np.sqrt(max(1, budget)))


def rdss_khsample(
    Z: NDArray[np.float64], n: int, alpha: float, gamma: float = RDSS_GAMMA
) -> list[int]:
    """Select in order with RDSS's ``khsample`` (Gaussian kernel).

    mu_j is the mean kernel similarity of j to the other rows (sum / N); F_j is
    the running mean similarity of j to the picks, updated by RDSS's recursion;
    each step takes the first arg min of F - alpha * mu among unpicked rows.
    """
    N = len(Z)
    mu_kernel = np.exp(-gamma * pairwise_distances(Z, squared=True))
    mu = (mu_kernel - np.diag(np.diag(mu_kernel))).sum(axis=0) / N
    K = np.exp(-gamma * cdist(Z, Z, "sqeuclidean"))
    F = np.zeros(N)
    picked = np.zeros(N, dtype=bool)
    order: list[int] = []
    for i in range(n):
        objective = np.where(picked, np.inf, F - alpha * mu)
        idx = int(objective.argmin())
        order.append(idx)
        picked[idx] = True
        F = (1 - 1 / (i + 1)) * F + 1 / (i + 1) * K[idx]
    return order


def select_rdss(X: NDArray[np.float64], budget: int, seed: int) -> NDArray[np.int64]:
    """RDSS on z-scored X; deterministic, so the seed is unused."""
    _check_budget(len(X), budget)
    chosen = rdss_khsample(zscore(X), budget, rdss_alpha(budget))
    return np.sort(np.array(chosen, dtype=np.int64))


# -------------------------------------------------------------------- ZCore


def zcore_draws(
    E: NDArray[np.float64], n: int, sample_dim: int, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """N random column subsets and triangular(min, median, max) points on them."""
    dims = np.argsort(rng.random((n, E.shape[1])), axis=1)[:, :sample_dim]
    lo, mode, hi = E.min(axis=0), np.median(E, axis=0), E.max(axis=0)
    samples = rng.triangular(lo[dims], mode[dims], hi[dims])
    return dims.astype(np.int64), np.asarray(samples, dtype=np.float64)


def _redundancy(
    nn_dist: NDArray[np.float64], idx: NDArray[np.int64], redund_nn: int, exp: int
) -> NDArray[np.float64]:
    """Normalised 1/d^exp penalties over the redund_nn rows nearest each cover row.

    Rows whose nearest neighbour is at distance 0 are handled by the caller.
    """
    n = nn_dist.shape[1]
    safe = np.where(nn_dist == 0, 1.0, nn_dist)
    if n - 1 <= redund_nn:
        W = 1.0 / safe**exp
        W[np.arange(len(idx)), idx] = 0.0
    else:
        nearest = np.argsort(nn_dist, axis=1)[:, 1 : redund_nn + 1]
        W = np.zeros_like(nn_dist)
        values = 1.0 / np.take_along_axis(safe, nearest, axis=1) ** exp
        np.put_along_axis(W, nearest, values, axis=1)
    W /= W.sum(axis=1, keepdims=True)
    return np.asarray(W.sum(axis=0))


def zcore_accumulate(
    E: NDArray[np.float64],
    dims: NDArray[np.int64],
    samples: NDArray[np.float64],
    redund_nn: int = ZCORE_REDUND_NN,
    redund_exp: int = ZCORE_REDUND_EXP,
) -> NDArray[np.float64]:
    """Score as ZCore's ``sample_score`` on given draws (no init, no scaling).

    Each draw: the L1-nearest row on its columns gains 1; if that row's nearest
    other row (first after it in ``np.argsort``) is at distance 0, that one loses
    1, else its redund_nn nearest rows lose 1/d^exp normalised to sum 1.
    """
    n = len(E)
    P = np.transpose(E[:, dims], (1, 0, 2))
    idx = np.abs(P - samples[:, None, :]).sum(axis=-1).argmin(axis=1)
    scores = np.bincount(idx, minlength=n).astype(np.float64)
    centre = P[np.arange(len(idx)), idx]
    nn_dist = np.abs(P - centre[:, None, :]).sum(axis=-1)
    dup = (nn_dist == 0).sum(axis=1) >= 2
    if dup.any():
        second = np.argsort(nn_dist[dup], axis=1)[:, 1]
        scores -= np.bincount(second, minlength=n)
    if (~dup).any():
        scores -= _redundancy(nn_dist[~dup], idx[~dup], redund_nn, redund_exp)
    return scores


def zcore_scores(
    E: NDArray[np.float64], seed: int, n_sample: int | None = None
) -> NDArray[np.float32]:
    """Compute ZCore scores in [0, 1] (float32); constant columns are dropped first."""
    n_sample = ZCORE_N_SAMPLE if n_sample is None else n_sample
    keep = E.max(axis=0) > E.min(axis=0)
    if not keep.any():
        return np.ones(len(E), dtype=np.float32)
    E = E[:, keep]
    sample_dim = min(ZCORE_SAMPLE_DIM, int(keep.sum()))
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0, 1, len(E))
    for start in range(0, n_sample, ZCORE_BATCH):
        size = min(ZCORE_BATCH, n_sample - start)
        dims, samples = zcore_draws(E, size, sample_dim, rng)
        scores += zcore_accumulate(E, dims, samples)
    low = scores.min()
    return ((scores - low) / (scores.max() - low)).astype(np.float32)


def select_zcore(X: NDArray[np.float64], budget: int, seed: int) -> NDArray[np.int64]:
    """The budget rows with the highest ZCore score on z-scored X."""
    _check_budget(len(X), budget)
    scores = zcore_scores(zscore(X), seed)
    return np.sort(np.argsort(-scores)[:budget]).astype(np.int64)


BASELINES: dict[str, Strategy] = {
    "usl": select_usl,
    "rdss": select_rdss,
    "zcore": select_zcore,
}
