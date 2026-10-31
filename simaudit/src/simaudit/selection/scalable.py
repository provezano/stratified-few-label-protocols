"""Scalable versions of the dense cold-start competitors (EXPLORATORY, P2 paper).

Each function keeps the interface of its dense original, (X, budget, seed) ->
sorted distinct row indices, and never builds an n x n matrix:

- k-medoids and LUCoS with CLARA (Kaufman & Rousseeuw 1990, Ch. 3: 5 samples of
  40 + 2k rows, each later sample containing the best medoids so far) or with
  FasterPAM on one uniform sample of s = min(n, max(2000, 40k)) rows (the
  landmark size of the F1 family). The k-medoids search is FasterPAM
  (Schubert & Rousseeuw 2021): eager single swaps, O(k) swap evaluation, from a
  k-medoids++ seeding. The ``kmedoids`` package (GPL-3, Rust) was used only as
  an isolated reference to check this re-implementation, never imported here.
- ProbCover (Yehuda et al. 2022) with the purity of every candidate radius from
  each row's distance to its nearest other-pseudo-class row, a sparse radius
  graph and sparse greedy coverage: exact up to rounding, except that the 100
  radius candidates are quantiles of sampled pairs when n > EXACT_PAIRS_MAX.
- USL (Wang, Lian & Yu 2022) with an exact kNN search and chunked k-means and
  regulariser: identical to ``baselines.select_usl`` up to rounding.

LUCoS keeps the embedding of ``lucos.embed`` (TabClustPFN loaded with
``weights_only=True`` after the SHA-256 check) and changes only the k-medoids
step. Memory is O(n (d + k) + s^2) for the medoid methods and O(n k + |E|) for
ProbCover, with |E| the number of radius-graph edges (data-dependent).
"""

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from simaudit.graph.knn import zscore
from simaudit.selection import baselines as bl
from simaudit.selection.strategies import _check_budget

CLARA_SAMPLES = 5
LANDMARK_FLOOR = 2000
LANDMARK_PER_K = 40
FASTERPAM_MAX_ITER = 100
EXACT_PAIRS_MAX = 2000  # ProbCover: exact pairwise quantiles up to this n
PAIR_SAMPLE = 1_000_000
CHUNK_ELEMS = 1 << 20
IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


# ------------------------------------------------------------------ k-medoids


def clara_sample_size(k: int) -> int:
    """Return CLARA's sample size 40 + 2k (Kaufman & Rousseeuw 1990)."""
    return 40 + 2 * k


def landmark_size(n: int, k: int) -> int:
    """Sample size of FasterPAM-on-sample: min(n, max(2000, 40k))."""
    return min(n, max(LANDMARK_FLOOR, LANDMARK_PER_K * k))


def kmedoids_pp(Z: FloatArray, k: int, rng: np.random.Generator) -> IntArray:
    """k-medoids++ seeding (as ``strategies._kmedoids_pp``) with O(n) memory."""
    n = len(Z)
    medoids = [int(rng.integers(n))]
    dmin = cdist(Z, Z[medoids])[:, 0]
    while len(medoids) < k:
        weight = dmin**2
        if weight.sum() == 0:
            weight = np.isin(np.arange(n), medoids, invert=True).astype(float)
        nxt = int(rng.choice(n, p=weight / weight.sum()))
        medoids.append(nxt)
        np.minimum(dmin, cdist(Z, Z[[nxt]])[:, 0], out=dmin)
    return np.array(medoids, dtype=np.int64)


def _nearest_two(
    D: FloatArray, medoids: IntArray
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Nearest medoid (position), its distance and the second-nearest distance."""
    Dm = D[:, medoids]
    if len(medoids) == 1:
        near = np.zeros(len(D), dtype=np.int64)
        return near, Dm[:, 0].copy(), np.full(len(D), np.inf)
    order = np.argsort(Dm, axis=1, kind="stable")[:, :2]
    rows = np.arange(len(D))
    near = order[:, 0].astype(np.int64)
    return near, Dm[rows, order[:, 0]], Dm[rows, order[:, 1]]


def _removal_loss(
    near: IntArray, dnear: FloatArray, dsec: FloatArray, k: int
) -> FloatArray:
    return np.bincount(near, weights=dsec - dnear, minlength=k)


def _best_swap(
    dj: FloatArray,
    removal: FloatArray,
    near: IntArray,
    dnear: FloatArray,
    dsec: FloatArray,
) -> tuple[float, int]:
    """Return the FasterPAM change in total deviation of the best swap with j."""
    k = len(removal)
    if k == 1:  # every row moves to j (no second medoid to fall back on)
        return float((dj - dnear).sum()), 0
    closer = dj < dnear
    second = ~closer & (dj < dsec)
    ploss = removal + np.bincount(
        near[closer], weights=(dnear - dsec)[closer], minlength=k
    )
    ploss += np.bincount(near[second], weights=(dj - dsec)[second], minlength=k)
    b = int(np.argmin(ploss))
    return float(ploss[b] + (dj - dnear)[closer].sum()), b


def fasterpam(
    D: FloatArray, medoids: IntArray, max_iter: int = FASTERPAM_MAX_ITER
) -> tuple[IntArray, int]:
    """Eager FasterPAM on a dissimilarity matrix from the given medoids.

    Candidates are visited in index order; a swap is applied as soon as it
    lowers the total deviation; a pass ends early at the last swapped
    candidate. Returns the medoids and the number of swaps.
    """
    med = np.array(medoids, dtype=np.int64)
    n, k = len(D), len(med)
    near, dnear, dsec = _nearest_two(D, med)
    removal = _removal_loss(near, dnear, dsec, k)
    loss, last, swaps = float(dnear.sum()), n, 0
    for _ in range(max_iter):
        before, last_loss = swaps, loss
        for j in range(n):
            if j == last:
                break
            if med[near[j]] == j:
                continue
            change, b = _best_swap(D[j], removal, near, dnear, dsec)
            if change >= 0:
                continue
            med[b], last, swaps = j, j, swaps + 1
            near, dnear, dsec = _nearest_two(D, med)
            removal = _removal_loss(near, dnear, dsec, k)
            loss = float(dnear.sum())
        if swaps == before or loss >= last_loss:
            break
    return med, swaps


def assign_cost(Z: FloatArray, centres: FloatArray) -> float:
    """Total distance of every row to its nearest centre, in row chunks."""
    step = max(1, CHUNK_ELEMS // max(1, len(centres)))
    return float(
        sum(
            cdist(Z[a : a + step], centres).min(axis=1).sum()
            for a in range(0, len(Z), step)
        )
    )


def _pam_on(
    Z: FloatArray, rows: IntArray, k: int, rng: np.random.Generator
) -> IntArray:
    """Run FasterPAM from k-medoids++ on the given rows; returns row indices."""
    D = cdist(Z[rows], Z[rows])
    medoids, _ = fasterpam(D, kmedoids_pp(Z[rows], k, rng))
    return rows[medoids]


def clara(Z: FloatArray, k: int, seed: int, samples: int = CLARA_SAMPLES) -> IntArray:
    """CLARA with FasterPAM: the lowest full-data cost over ``samples`` samples."""
    n = len(Z)
    size = min(n, clara_sample_size(k))
    rng = np.random.default_rng([seed, 44])
    best: IntArray | None = None
    best_cost = np.inf
    for _ in range(samples):
        if best is None:
            rows = rng.choice(n, size, replace=False)
        else:
            rest = np.setdiff1d(np.arange(n), best)
            rows = np.concatenate([best, rng.choice(rest, size - k, replace=False)])
        rows = np.sort(rows).astype(np.int64)
        medoids = _pam_on(Z, rows, k, rng)
        cost = assign_cost(Z, Z[medoids])
        if cost < best_cost:
            best, best_cost = medoids, cost
    assert best is not None
    return np.sort(best)


def fasterpam_on_sample(Z: FloatArray, k: int, seed: int) -> IntArray:
    """Run FasterPAM on one uniform sample of ``landmark_size(n, k)`` rows."""
    n = len(Z)
    rng = np.random.default_rng([seed, 43])
    s = landmark_size(n, k)
    rows = np.arange(n) if s >= n else np.sort(rng.choice(n, s, replace=False))
    return np.sort(_pam_on(Z, rows.astype(np.int64), k, rng))


def select_kmedoids_clara(X: FloatArray, budget: int, seed: int) -> IntArray:
    """S-KM made scalable: CLARA + FasterPAM in z-scored X."""
    _check_budget(len(X), budget)
    return clara(zscore(X), budget, seed)


def select_kmedoids_fps(X: FloatArray, budget: int, seed: int) -> IntArray:
    """S-KM made scalable: FasterPAM on a sample of z-scored X."""
    _check_budget(len(X), budget)
    return fasterpam_on_sample(zscore(X), budget, seed)


def _lucos_embedding(X: FloatArray) -> FloatArray:
    from simaudit.selection import lucos

    return np.asarray(lucos.embed(X), dtype=np.float64)


def select_lucos_clara(X: FloatArray, budget: int, seed: int) -> IntArray:
    """Select with the LUCoS embedding (unchanged) + CLARA/FasterPAM."""
    _check_budget(len(X), budget)
    return clara(_lucos_embedding(X), budget, seed)


def select_lucos_fps(X: FloatArray, budget: int, seed: int) -> IntArray:
    """Select with the LUCoS embedding (unchanged) + FasterPAM on a sample."""
    _check_budget(len(X), budget)
    return fasterpam_on_sample(_lucos_embedding(X), budget, seed)


# ------------------------------------------------------------------ ProbCover


def _pair_quantiles(Z: FloatArray, qs: FloatArray, seed: int) -> FloatArray:
    n = len(Z)
    if n <= EXACT_PAIRS_MAX:
        return np.asarray(np.quantile(pdist(Z), qs))
    rng = np.random.default_rng([seed, 45])
    i = rng.integers(0, n, PAIR_SAMPLE)
    j = rng.integers(0, n - 1, PAIR_SAMPLE)
    j = j + (j >= i)  # a uniform pair with i != j
    return np.asarray(np.quantile(np.linalg.norm(Z[i] - Z[j], axis=1), qs))


def nearest_other_class(Z: FloatArray, pseudo: IntArray) -> FloatArray:
    """Distance of each row to its nearest row with a different pseudo-label."""
    out = np.full(len(Z), np.inf)
    for c in np.unique(pseudo):
        inside = pseudo == c
        if inside.all():
            continue
        tree = NearestNeighbors(n_neighbors=1).fit(Z[~inside])
        out[inside] = tree.kneighbors(Z[inside])[0][:, 0]
    return out


def probcover_delta_scalable(
    X: FloatArray, n_classes: int, seed: int, purity: float = 0.95
) -> float:
    """``strategies.probcover_delta`` without the n x n matrix."""
    Z = zscore(X)
    pseudo = KMeans(n_clusters=n_classes, n_init=10, random_state=seed).fit_predict(Z)
    candidates = _pair_quantiles(Z, np.linspace(0.005, 0.5, 100), seed)
    other = nearest_other_class(Z, pseudo)
    best = float(candidates[0])
    for delta in candidates:
        if 1.0 - (other <= delta).mean() >= purity:
            best = float(delta)
    return best


def select_probcover_scalable(
    X: FloatArray, budget: int, seed: int, n_classes: int
) -> IntArray:
    """Select by ProbCover greedy coverage on a sparse delta-ball graph."""
    _check_budget(len(X), budget)
    Z = zscore(X)
    delta = probcover_delta_scalable(X, n_classes, seed)
    graph = NearestNeighbors(radius=delta).fit(Z).radius_neighbors_graph(Z)
    covers = sparse.csr_matrix(graph, dtype=np.float64)
    uncovered = np.ones(len(Z))
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    while len(chosen) < budget:
        gain = covers @ uncovered
        gain[chosen] = -1
        if gain.max() > 0:
            nxt = int(gain.argmax())
        else:
            rest = np.setdiff1d(np.arange(len(Z)), chosen)
            nxt = int(rng.choice(rest))
        chosen.append(nxt)
        uncovered[covers.indices[covers.indptr[nxt] : covers.indptr[nxt + 1]]] = 0
    return np.sort(np.array(chosen, dtype=np.int64))


# ------------------------------------------------------------------------ USL


def usl_knn_mean_sqdist_scalable(Z: FloatArray, k: int) -> FloatArray:
    """Mean squared distance to the k nearest rows, the row itself included."""
    dist, _ = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(Z)
    return np.asarray((dist**2).mean(axis=1))


def _rows_per_chunk(width: int, d: int) -> int:
    return max(1, CHUNK_ELEMS // max(1, width * d))


def _assign_chunked(
    Z: FloatArray, centroids: FloatArray, alive: NDArray[np.bool_]
) -> IntArray:
    """``baselines._assign`` in row chunks (same arithmetic, bit-identical)."""
    out = np.empty(len(Z), dtype=np.int64)
    step = _rows_per_chunk(len(centroids), Z.shape[1])
    for a in range(0, len(Z), step):
        out[a : a + step] = bl._assign(Z[a : a + step], centroids, alive)
    return out


def usl_kmeans_chunked(
    Z: FloatArray, k: int, seed: int, n_iter: int = bl.USL_KMEANS_ITERS
) -> IntArray:
    """``baselines.usl_kmeans`` with the chunked assignment."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    init = torch.randperm(len(Z), generator=generator)[:k].numpy()
    centroids = Z[init].astype(np.float64)
    alive = np.ones(k, dtype=bool)
    labels = _assign_chunked(Z, centroids, alive)
    for _ in range(n_iter):
        labels = _assign_chunked(Z, centroids, alive)
        counts = np.bincount(labels, minlength=k)
        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, Z)
        alive = counts > 0
        centroids[alive] = sums[alive] / counts[alive, None]
    return labels


def _regulariser_chunked(
    Z: FloatArray,
    labels: IntArray,
    picks: list[int],
    horizon: float | None,
    alpha: float,
) -> FloatArray:
    """``baselines._usl_regulariser`` in row chunks (bit-identical)."""
    out = np.empty(len(Z))
    step = _rows_per_chunk(len(picks), Z.shape[1])
    for a in range(0, len(Z), step):
        out[a : a + step] = _regulariser_block(
            Z, labels, picks, horizon, alpha, a, step
        )
    return out


def _regulariser_block(
    Z: FloatArray,
    labels: IntArray,
    picks: list[int],
    horizon: float | None,
    alpha: float,
    a: int,
    step: int,
) -> FloatArray:
    """``baselines._usl_regulariser`` for rows a..a+step against all picks."""
    rows = Z[a : a + step]
    sq = ((rows[:, None, :] - Z[picks][None, :, :]) ** 2).sum(axis=-1)
    sq = np.where(labels[a : a + step, None] == labels[picks][None, :], bl.USL_FAR, sq)
    sq[sq == 0] = bl.USL_FAR
    if horizon is not None:
        sq[sq >= horizon] = bl.USL_FAR
    return np.asarray((1.0 / sq**alpha).sum(axis=1))


def _usl_rounds_chunked(
    Z: FloatArray, dist: FloatArray, labels: IntArray, budget: int
) -> tuple[list[int], FloatArray]:
    """``baselines._usl_rounds`` at USL's defaults, with the chunked regulariser."""
    cfg = bl.USL_REG
    with np.errstate(divide="ignore"):
        density = 1.0 / dist
    reg = np.zeros(len(Z))
    picks: list[int] = []
    score = density
    members = [np.flatnonzero(labels == c) for c in range(budget)]
    for _ in range(int(cfg["n_iter"])):
        score = density - cfg["w"] * reg
        picks = [int(m[int(score[m].argmax())]) for m in members if len(m)]
        new = _regulariser_chunked(Z, labels, picks, cfg["horizon"], cfg["alpha"])
        reg = reg * cfg["momentum"] + new * (1 - cfg["momentum"])
    return picks, score


def select_usl_scalable(X: FloatArray, budget: int, seed: int) -> IntArray:
    """USL with an exact kNN index and chunked k-means: O(n k) memory."""
    _check_budget(len(X), budget)
    Z = bl.usl_features(X)
    dist = usl_knn_mean_sqdist_scalable(Z, min(bl.USL_K, len(Z)))
    labels = usl_kmeans_chunked(Z, budget, seed)
    picks, score = _usl_rounds_chunked(Z, dist, labels, budget)
    if len(picks) < budget:
        taken = set(picks)
        rest = [int(i) for i in np.argsort(-score, kind="stable") if i not in taken]
        picks += rest[: budget - len(picks)]
    return np.sort(np.array(picks, dtype=np.int64))
