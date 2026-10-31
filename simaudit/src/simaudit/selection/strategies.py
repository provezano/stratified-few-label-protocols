"""Cold-start label selection: choose which rows to label before any label is known.

Every strategy maps (X, budget, seed) to sorted, distinct row indices, using only
the features (z-scored, as for the graphs). The protocol is a total budget B, so
a strategy may miss classes; ``predict_with_budget`` then gives unseen classes
zero probability, and their F1 is 0 in the macro average.
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

from simaudit.graph.knn import zscore

Strategy = Callable[[NDArray[np.float64], int, int], NDArray[np.int64]]
Predictor = Callable[..., NDArray[np.float64]]


def _check_budget(n: int, budget: int) -> None:
    if not 1 <= budget < n:
        raise ValueError(f"budget {budget} must be in [1, {n - 1}] for {n} rows")


def select_random(X: NDArray[np.float64], budget: int, seed: int) -> NDArray[np.int64]:
    """Uniform sample without replacement (the realistic random baseline)."""
    _check_budget(len(X), budget)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(X), size=budget, replace=False)).astype(np.int64)


def select_kcenter(X: NDArray[np.float64], budget: int, seed: int) -> NDArray[np.int64]:
    """Greedy k-center (Sener & Savarese 2018) from one random seed row."""
    _check_budget(len(X), budget)
    Z = zscore(X)
    chosen = [int(np.random.default_rng(seed).integers(len(Z)))]
    dist = np.linalg.norm(Z - Z[chosen[0]], axis=1)
    while len(chosen) < budget:
        nxt = int(dist.argmax())
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(Z - Z[nxt], axis=1))
    return np.sort(np.array(chosen, dtype=np.int64))


def select_kmeans_central(
    X: NDArray[np.float64], budget: int, seed: int
) -> NDArray[np.int64]:
    """Pick the row nearest each k-means (k = budget) centroid; ties: lowest index."""
    _check_budget(len(X), budget)
    Z = zscore(X)
    km = KMeans(n_clusters=budget, n_init=10, random_state=seed).fit(Z)
    chosen: list[int] = []
    for center in km.cluster_centers_:
        order = np.argsort(np.linalg.norm(Z - center, axis=1), kind="stable")
        chosen.append(int(next(i for i in order if i not in chosen)))
    return np.sort(np.array(chosen, dtype=np.int64))


def _kmedoids_pp(
    D: NDArray[np.float64], k: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """k-medoids++ seeding: each new medoid with probability ~ squared distance."""
    medoids = [int(rng.integers(len(D)))]
    while len(medoids) < k:
        weight = D[:, medoids].min(axis=1) ** 2
        if weight.sum() == 0:
            weight = np.isin(np.arange(len(D)), medoids, invert=True).astype(float)
        medoids.append(int(rng.choice(len(D), p=weight / weight.sum())))
    return np.array(medoids, dtype=np.int64)


def _update_medoids(
    D: NDArray[np.float64], current: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Move each medoid to the member minimizing the within-cluster distance."""
    assign = D[:, current].argmin(axis=1)
    updated = current.copy()
    for j in range(len(current)):
        members = np.flatnonzero(assign == j)
        if len(members):
            updated[j] = members[int(D[np.ix_(members, members)].sum(axis=1).argmin())]
    return updated


def _kmedoids_inits(
    Z: NDArray[np.float64], k: int, seed: int, n_init: int
) -> NDArray[np.int64] | None:
    """Lowest-cost of n_init k-medoids++ runs with k distinct medoids, or None."""
    D = cdist(Z, Z)
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in range(n_init):
        current = _kmedoids_pp(D, k, rng)
        for _ in range(100):
            updated = _update_medoids(D, current)
            if np.array_equal(updated, current):
                break
            current = updated
        cost = D[:, current].min(axis=1).sum()
        if cost < best_cost and len(set(current.tolist())) == k:
            best, best_cost = current, cost
    return None if best is None else np.sort(best)


def _fill_farthest(
    Z: NDArray[np.float64], chosen: NDArray[np.int64], k: int, seed: int
) -> NDArray[np.int64]:
    """Add rows until k: the farthest from the chosen set, ties by seeded draw."""
    rng = np.random.default_rng([seed, 4])
    picks = [int(i) for i in chosen]
    dist = cdist(Z, Z[picks]).min(axis=1)
    while len(picks) < k:
        dist[picks] = -np.inf
        candidates = np.flatnonzero(dist == dist.max())
        nxt = int(rng.choice(candidates))
        picks.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(Z - Z[nxt], axis=1))
    return np.sort(np.array(picks, dtype=np.int64))


def _kmedoids_distinct(
    Z: NDArray[np.float64], k: int, seed: int, n_init: int
) -> NDArray[np.int64]:
    """k-medoids on the distinct rows (K = min(k, n_distinct)), then fill to k."""
    _, first = np.unique(Z, axis=0, return_index=True)
    first = np.sort(first)
    medoids = _kmedoids_inits(Z[first], min(k, len(first)), seed, n_init)
    if medoids is None:  # pragma: no cover - distinct rows always succeed
        raise RuntimeError("k-medoids failed on distinct rows")
    return _fill_farthest(Z, first[medoids], k, seed)


def _kmedoids(
    Z: NDArray[np.float64], k: int, seed: int, n_init: int = 10
) -> NDArray[np.int64]:
    """Alternating k-medoids with k-medoids++ seeding; the lowest-cost of n_init.

    Amendment 4: if no initialisation ends with k distinct medoids (only when
    rows repeat), k-medoids runs on the distinct rows and the remaining seats
    are filled by farthest point; the result is otherwise unchanged.
    """
    best = _kmedoids_inits(Z, k, seed, n_init)
    return _kmedoids_distinct(Z, k, seed, n_init) if best is None else best


def select_kmedoids(
    X: NDArray[np.float64], budget: int, seed: int
) -> NDArray[np.int64]:
    """Run k-medoids with K = budget in z-scored X (LUCoS's raw-space baseline)."""
    _check_budget(len(X), budget)
    return _kmedoids(zscore(X), budget, seed)


def select_kmedoids_raw(
    X: NDArray[np.float64], budget: int, seed: int
) -> NDArray[np.int64]:
    """S-KMR (Amendment 4): LUCoS's raw-space k-medoids, unscaled X, one init.

    Mirrors ``select_lucos`` on the raw features: one k-medoids++ initialisation;
    if it ends with repeated medoids, the 10-init routine (with the Amendment 4
    distinct-row fallback) is used instead.
    """
    _check_budget(len(X), budget)
    chosen = _kmedoids_inits(np.asarray(X, dtype=np.float64), budget, seed, 1)
    return _kmedoids(X, budget, seed) if chosen is None else chosen


def _typicality(Z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse mean distance to the min(20, size - 1) nearest neighbours in Z."""
    if len(Z) == 1:
        return np.ones(1)
    D = np.sort(cdist(Z, Z), axis=1)[:, 1 : min(20, len(Z) - 1) + 1]
    return np.asarray(1.0 / (D.mean(axis=1) + 1e-12))


def select_typiclust(
    X: NDArray[np.float64], budget: int, seed: int
) -> NDArray[np.int64]:
    """Select with TypiClust (Hacohen et al. 2022) and an empty initial labeled set.

    k-means with K = budget; repeatedly take the cluster with the fewest picks
    (then the largest), ignoring clusters of 5 rows or fewer unless none is
    larger, and pick its most typical unpicked row.
    """
    _check_budget(len(X), budget)
    Z = zscore(X)
    assign = KMeans(n_clusters=budget, n_init=10, random_state=seed).fit_predict(Z)
    sizes = np.bincount(assign, minlength=budget)
    valid = [c for c in range(budget) if sizes[c] > 5] or list(range(budget))
    typical = {c: _typicality(Z[assign == c]) for c in range(budget)}
    picks = {c: 0 for c in range(budget)}
    chosen: list[int] = []
    while len(chosen) < budget:
        open_ = [c for c in valid if picks[c] < sizes[c]] or [
            c for c in range(budget) if picks[c] < sizes[c]
        ]
        cluster = min(open_, key=lambda c: (picks[c], -sizes[c], c))
        members = np.flatnonzero(assign == cluster)
        order = np.argsort(-typical[cluster], kind="stable")
        chosen.append(int(next(members[i] for i in order if members[i] not in chosen)))
        picks[cluster] += 1
    return np.sort(np.array(chosen, dtype=np.int64))


def probcover_delta(
    X: NDArray[np.float64], n_classes: int, seed: int, purity: float = 0.95
) -> float:
    """Largest ball radius whose purity w.r.t. k-means(n_classes) is >= purity."""
    Z = zscore(X)
    pseudo = KMeans(n_clusters=n_classes, n_init=10, random_state=seed).fit_predict(Z)
    D = cdist(Z, Z)
    candidates = np.quantile(
        D[np.triu_indices(len(Z), 1)], np.linspace(0.005, 0.5, 100)
    )
    best = float(candidates[0])
    for delta in candidates:
        mixed = ((D <= delta) & (pseudo[:, None] != pseudo[None, :])).any(axis=1)
        if 1.0 - mixed.mean() >= purity:
            best = float(delta)
    return best


def select_probcover(
    X: NDArray[np.float64], budget: int, seed: int, n_classes: int
) -> NDArray[np.int64]:
    """Select with ProbCover (Yehuda et al. 2022): greedy delta-ball coverage.

    Uses C through the purity heuristic for delta, hence it is class-aware.
    Once everything is covered, the remaining picks are uniform at random.
    """
    _check_budget(len(X), budget)
    Z = zscore(X)
    covers = cdist(Z, Z) <= probcover_delta(X, n_classes, seed)
    uncovered = np.ones(len(Z), dtype=bool)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    while len(chosen) < budget:
        gain = (covers & uncovered[None, :]).sum(axis=1)
        gain[chosen] = -1
        if gain.max() > 0:
            nxt = int(gain.argmax())
        else:
            rest = np.setdiff1d(np.arange(len(Z)), chosen)
            nxt = int(rng.choice(rest))
        chosen.append(nxt)
        uncovered &= ~covers[nxt]
    return np.sort(np.array(chosen, dtype=np.int64))


def select_sigma_opt(
    X: NDArray[np.float64], budget: int, seed: int, k: int = 10, reg: float = 1e-2
) -> NDArray[np.int64]:
    """Greedy Sigma-optimality (Ma et al. 2013) on a symmetric kNN graph of X.

    The GRF covariance is (L + reg*I)^-1 on an unweighted union kNN graph; each
    step labels the row that most reduces 1'Sigma_uu 1 and conditions on it.
    The seed is unused: the criterion is deterministic.
    """
    _check_budget(len(X), budget)
    Z = zscore(X)
    nbrs = np.argsort(cdist(Z, Z), axis=1, kind="stable")[:, 1 : min(k, len(Z) - 1) + 1]
    W = np.zeros((len(Z), len(Z)))
    W[np.repeat(np.arange(len(Z)), nbrs.shape[1]), nbrs.ravel()] = 1.0
    W = np.maximum(W, W.T)
    sigma = np.linalg.inv(np.diag(W.sum(axis=1)) - W + reg * np.eye(len(Z)))
    chosen: list[int] = []
    while len(chosen) < budget:
        col = sigma.sum(axis=0)
        score = col**2 / np.maximum(np.diag(sigma), 1e-300)
        score[chosen] = -np.inf
        v = int(score.argmax())
        chosen.append(v)
        sigma = sigma - np.outer(sigma[:, v], sigma[v]) / sigma[v, v]
    return np.sort(np.array(chosen, dtype=np.int64))


def select_outlier_filtered(
    X: NDArray[np.float64], budget: int, seed: int, drop: float = 0.10
) -> NDArray[np.int64]:
    """Uniform sample after removing the IsolationForest's most anomalous rows."""
    _check_budget(len(X), budget)
    score = IsolationForest(random_state=seed).fit(X).score_samples(X)
    keep = np.argsort(score, kind="stable")[int(np.floor(drop * len(X))) :]
    if len(keep) < budget:
        keep = np.arange(len(X))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(keep, size=budget, replace=False)).astype(np.int64)


def select_stratified_medoids(
    X: NDArray[np.float64], y: NDArray[np.int64], m: int, seed: int
) -> NDArray[np.int64]:
    """Diagnostic only (uses y): m k-medoids per class, in z-scored X."""
    Z = zscore(X)
    chosen: list[int] = []
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        if len(rows) <= m:
            chosen.extend(rows.tolist())
        else:
            chosen.extend(rows[_kmedoids(Z[rows], m, seed)].tolist())
    return np.sort(np.array(chosen, dtype=np.int64))


STRATEGIES: dict[str, Strategy] = {
    "random": select_random,
    "kcenter": select_kcenter,
    "kmeans_central": select_kmeans_central,
    "kmedoids": select_kmedoids,
    "kmedoids_raw": select_kmedoids_raw,
    "typiclust": select_typiclust,
    "sigma_opt": select_sigma_opt,
    "outlier_filtered": select_outlier_filtered,
}
CLASS_AWARE: dict[str, Callable[..., NDArray[np.int64]]] = {
    "probcover": select_probcover,
}


def class_coverage(y: NDArray[np.int64], labeled: NDArray[np.int64]) -> float:
    """Fraction of the classes present in y that have at least one labeled row."""
    return len(np.unique(y[labeled])) / len(np.unique(y))


def expand_proba(
    proba_seen: NDArray[np.float64], seen: NDArray[np.int64], n_classes: int
) -> NDArray[np.float64]:
    """Place columns of seen classes into an n_classes matrix; unseen get 0."""
    out = np.zeros((len(proba_seen), n_classes))
    out[:, seen] = proba_seen
    return out


def predict_with_budget(
    predictor: Predictor,
    X: NDArray[np.float64],
    labeled: NDArray[np.int64],
    y_lab: NDArray[np.int64],
    n_classes: int,
    seed: int,
) -> NDArray[np.float64]:
    """FM prediction when the labeled rows may cover only some classes."""
    seen, y_fit = np.unique(y_lab, return_inverse=True)
    if len(seen) == 1:
        return expand_proba(np.ones((len(X), 1)), seen, n_classes)
    proba = predictor(X[labeled], y_fit.astype(np.int64), X, seed)
    return expand_proba(proba, seen, n_classes)
