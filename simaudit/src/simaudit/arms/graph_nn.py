"""GCN, APPNP and Correct & Smooth on the mutual kNN graph (dense, CPU, float64).

Specs: research_notes/Modern tabular SSL baselines/gnn_cs_graph_learning.md,
Sec. 7. Every model trains for a fixed number of epochs without early stopping
(no labeled validation exists); GNN outputs are averaged over ``restarts``
initialisations. C&S labeled-row predictions are cross-fitted with
k = min(m, 5) stratified folds; at m = 1 the in-sample predictions are used
(Smooth-dominated C&S), as pre-registered.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from torch import nn

from simaudit.graph.knn import zscore

DTYPE = torch.float64
Base = Callable[..., NDArray[np.float64]]


@dataclass(frozen=True)
class GCNConfig:
    """Two-layer GCN (Kipf & Welling 2017, Eq. 9)."""

    hidden: int = 64
    dropout: float = 0.5
    l2: float = 5e-4
    epochs: int = 300
    lr: float = 0.01
    restarts: int = 5


@dataclass(frozen=True)
class APPNPConfig:
    """APPNP (Klicpera et al. 2019, Eq. 4)."""

    alpha: float = 0.1
    l2: float = 5e-3
    dropout: float = 0.5
    K: int = 10
    hidden: int = 64
    epochs: int = 300
    lr: float = 0.01
    restarts: int = 5


@dataclass(frozen=True)
class CSConfig:
    """Correct & Smooth (Huang et al. 2021)."""

    scaling: str = "autoscale"
    alpha1: float = 0.8
    alpha2: float = 0.8
    iterations: int = 50


def grid(name: str) -> list[Any]:
    """The 16 pre-registered configurations, default first."""
    if name == "gcn":
        return [
            GCNConfig(hidden=h, dropout=d, l2=w, epochs=e)
            for h, d, w, e in itertools.product(
                (64, 16), (0.5, 0.2), (5e-4, 5e-3), (300, 100)
            )
        ]
    if name == "appnp":
        return [
            APPNPConfig(alpha=a, l2=w, dropout=d)
            for a, w, d in itertools.product(
                (0.1, 0.05, 0.2, 0.3), (5e-3, 5e-4), (0.5, 0.3)
            )
        ]
    if name == "cs":
        return [
            CSConfig(scaling=s, alpha1=a1, alpha2=a2)
            for s, a1, a2 in itertools.product(
                ("autoscale", "fdiff"), (0.8, 0.4), (0.8, 0.95, 0.6, 0.4)
            )
        ]
    raise ValueError(f"no grid for {name}")


# --- graph operators -----------------------------------------------------------


def _dense(W: sparse.spmatrix | NDArray[np.float64]) -> NDArray[np.float64]:
    if sparse.issparse(W):
        return np.asarray(W.toarray(), dtype=np.float64)  # type: ignore[union-attr]
    return np.asarray(W, dtype=np.float64)


def sym_norm(W: sparse.spmatrix | NDArray[np.float64]) -> NDArray[np.float64]:
    """D^-1/2 W D^-1/2 with zero rows for isolated vertices."""
    A = _dense(W)
    d = A.sum(axis=1)
    inv = np.where(d > 0, 1 / np.sqrt(np.where(d > 0, d, 1)), 0.0)
    return inv[:, None] * A * inv[None, :]


def gcn_adjacency(W: sparse.spmatrix | NDArray[np.float64]) -> NDArray[np.float64]:
    """Renormalised adjacency D~^-1/2 (A + I) D~^-1/2."""
    A = _dense(W) + np.eye(W.shape[0])
    inv = 1 / np.sqrt(A.sum(axis=1))
    return inv[:, None] * A * inv[None, :]


# --- GNNs ------------------------------------------------------------------------


def _edge_dropout(A: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Dropout on the stored entries of a sparse adjacency (resampled per call)."""
    if not training or p == 0:
        return A
    keep = torch.rand(A.values().shape, dtype=DTYPE) >= p
    values = A.values() * keep / (1 - p)
    return torch.sparse_coo_tensor(A.indices(), values, A.shape).coalesce()


def _sparse(A: NDArray[np.float64]) -> torch.Tensor:
    rows, cols = np.nonzero(A)
    index = torch.as_tensor(np.vstack([rows, cols]))
    return torch.sparse_coo_tensor(
        index, torch.as_tensor(A[rows, cols]), A.shape
    ).coalesce()


class _GCN(nn.Module):
    def __init__(self, d: int, cfg: GCNConfig, n_classes: int) -> None:
        super().__init__()
        self.lin0 = nn.Linear(d, cfg.hidden)
        self.lin1 = nn.Linear(cfg.hidden, n_classes)
        self.drop = nn.Dropout(cfg.dropout)
        for lin in (self.lin0, self.lin1):
            nn.init.xavier_uniform_(lin.weight)

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        H = torch.relu(A @ self.lin0(self.drop(X)))
        return A @ self.lin1(self.drop(H))  # type: ignore[no-any-return]

    def penalty(self) -> torch.Tensor:
        return (self.lin0.weight**2).sum() / 2


class _APPNP(nn.Module):
    def __init__(self, d: int, cfg: APPNPConfig, n_classes: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(d, cfg.hidden)
        self.lin2 = nn.Linear(cfg.hidden, n_classes)
        self.drop = nn.Dropout(cfg.dropout)
        self.cfg = cfg
        for lin in (self.lin1, self.lin2):
            nn.init.xavier_uniform_(lin.weight)

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        H: torch.Tensor = self.lin2(self.drop(torch.relu(self.lin1(self.drop(X)))))
        Z = H
        for _ in range(self.cfg.K):
            Z = (
                (1 - self.cfg.alpha)
                * _edge_dropout(A, self.cfg.dropout, self.training)
                @ Z
            )
            Z = Z + self.cfg.alpha * H
        return Z

    def penalty(self) -> torch.Tensor:
        value: torch.Tensor = (self.lin1.weight**2).sum() / 2
        return value


def _train_gnn(
    model: Any,
    A: torch.Tensor,
    X: torch.Tensor,
    labeled: torch.Tensor,
    y: torch.Tensor,
    cfg: Any,
) -> NDArray[np.float64]:
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    for _ in range(cfg.epochs):
        loss = nn.functional.cross_entropy(model(A, X)[labeled], y)
        loss = loss + cfg.l2 * model.penalty()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return np.asarray(torch.softmax(model(A, X), dim=1).numpy(), dtype=np.float64)


def fit_predict(
    name: str,
    X: NDArray[np.float64],
    W: sparse.spmatrix,
    labeled: NDArray[np.int64],
    y_lab: NDArray[np.int64],
    seed: int,
    config: Any,
) -> NDArray[np.float64]:
    """GCN or APPNP probabilities for every node, averaged over restarts."""
    n_classes = int(y_lab.max()) + 1
    Xt = torch.as_tensor(zscore(X), dtype=DTYPE)
    A = _sparse(gcn_adjacency(W))  # APPNP uses the same renormalised A
    lab, y = torch.as_tensor(labeled), torch.as_tensor(y_lab)
    runs = []
    for restart in range(config.restarts):
        torch.manual_seed(seed * 1000 + restart)
        cls = _GCN if name == "gcn" else _APPNP
        model = cls(X.shape[1], config, n_classes).to(DTYPE)
        runs.append(_train_gnn(model, A, Xt, lab, y, config))
    return np.asarray(np.mean(runs, axis=0), dtype=np.float64)


# --- Correct & Smooth --------------------------------------------------------------


def correct(
    S: NDArray[np.float64],
    Z: NDArray[np.float64],
    labeled: NDArray[np.int64],
    Y_lab: NDArray[np.float64],
    alpha: float,
    scaling: str,
    iterations: int = 50,
) -> NDArray[np.float64]:
    """Spread labeled residuals Y_L - Z_L and add them (scaled) to unlabeled rows."""
    n = len(Z)
    unl = np.setdiff1d(np.arange(n), labeled)
    E0 = np.zeros_like(Z)
    E0[labeled] = Y_lab - Z[labeled]
    if scaling == "autoscale":
        E = E0.copy()
        for _ in range(iterations):
            E = (1 - alpha) * E0 + alpha * S @ E
        sigma = np.abs(E0[labeled]).sum(axis=1).mean()
        norm = np.abs(E[unl]).sum(axis=1, keepdims=True)
        step = sigma * E[unl] / np.where(norm > 0, norm, 1.0)
    else:  # FDiff: harmonic diffusion with clamped labeled residuals, scale s = 1
        P = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-300)
        E = E0.copy()
        for _ in range(iterations):
            E = P @ E
            E[labeled] = E0[labeled]
        step = E[unl]
    out = Z.copy()
    out[unl] = Z[unl] + step
    return out


def smooth(
    S: NDArray[np.float64],
    Z: NDArray[np.float64],
    labeled: NDArray[np.int64],
    Y_lab: NDArray[np.float64],
    alpha: float,
    iterations: int = 50,
) -> NDArray[np.float64]:
    """Label-spreading of G (G_L = Y_L, G_U = Z_U) over the graph."""
    G0 = Z.copy()
    G0[labeled] = Y_lab
    G = G0.copy()
    for _ in range(iterations):
        G = (1 - alpha) * G0 + alpha * S @ G
    return G


def _logreg(
    X_fit: NDArray[np.float64],
    y_fit: NDArray[np.int64],
    X_query: NDArray[np.float64],
    seed: int,
    n_classes: int,
) -> NDArray[np.float64]:
    model = LogisticRegression(C=1.0, max_iter=1000).fit(X_fit, y_fit)
    proba = np.zeros((len(X_query), n_classes))
    proba[:, model.classes_] = model.predict_proba(X_query)
    return proba


def base_predictions(
    X: NDArray[np.float64],
    labeled: NDArray[np.int64],
    y_lab: NDArray[np.int64],
    seed: int,
    base: Base | str,
) -> NDArray[np.float64]:
    """Base probabilities for every row; labeled rows are cross-fitted (Sec. 7.3).

    ``base(X_fit, y_fit, X_query, seed) -> proba`` sees z-scored features;
    ``base="logreg"`` is multinomial logistic regression (C = 1).
    """
    n_classes = int(y_lab.max()) + 1
    Xz = zscore(X)
    predict: Base
    if base == "logreg":

        def predict(Xf: Any, yf: Any, Xq: Any, s: int) -> NDArray[np.float64]:
            return _logreg(Xf, yf, Xq, s, n_classes)

    else:
        predict = base  # type: ignore[assignment]
    Z = np.zeros((len(X), n_classes))
    Z[:] = predict(Xz[labeled], y_lab, Xz, seed)
    Z[labeled] = _cross_fit(predict, Xz, labeled, y_lab, seed, n_classes, Z[labeled])
    return Z


def cs_apply(
    W: sparse.spmatrix,
    Z: NDArray[np.float64],
    labeled: NDArray[np.int64],
    y_lab: NDArray[np.int64],
    config: CSConfig,
) -> NDArray[np.float64]:
    """Correct then Smooth a base prediction matrix ``Z``."""
    S = sym_norm(W)
    Y_lab = np.eye(Z.shape[1])[y_lab]
    Zr = correct(S, Z, labeled, Y_lab, config.alpha1, config.scaling, config.iterations)
    return smooth(S, Zr, labeled, Y_lab, config.alpha2, config.iterations)


def cs_fit_predict(
    X: NDArray[np.float64],
    W: sparse.spmatrix,
    labeled: NDArray[np.int64],
    y_lab: NDArray[np.int64],
    seed: int,
    config: CSConfig,
    base: Base | str,
) -> NDArray[np.float64]:
    """C&S end to end: cross-fitted base predictions, then Correct and Smooth."""
    Z = base_predictions(X, labeled, y_lab, seed, base)
    return cs_apply(W, Z, labeled, y_lab, config)


def _cross_fit(
    predict: Base,
    Xz: NDArray[np.float64],
    labeled: NDArray[np.int64],
    y_lab: NDArray[np.int64],
    seed: int,
    n_classes: int,
    in_sample: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Out-of-fold predictions for labeled rows; in-sample when m = 1."""
    m = int(np.bincount(y_lab).min())
    if m < 2:
        return in_sample
    k = min(m, 5)
    rng = np.random.default_rng(seed)
    fold = np.empty(len(labeled), dtype=int)
    for c in range(n_classes):
        idx = np.flatnonzero(y_lab == c)
        fold[rng.permutation(idx)] = np.arange(idx.size) % k
    out = np.empty((len(labeled), n_classes))
    for j in range(k):
        held = fold == j
        out[held] = predict(Xz[labeled[~held]], y_lab[~held], Xz[labeled[held]], seed)
    return out
