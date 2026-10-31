"""One pilot cell: every arm and diagnostic on one (dataset, m, seed) split.

Only ``y[labeled]`` is ever read, so unlabeled labels cannot leak into any arm
or diagnostic (enforced by tests/features/leakage.feature).
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import accuracy_score, f1_score

from simaudit.arms import fm
from simaudit.graph.knn import mutual_knn_graph, zscore
from simaudit.graph.propagation import gfhf, llgc

ARMS = ("FM1", "FM2", "FM3", "G1", "G2", "ST", "H1", "H2", "H3", "H4")


@dataclass(frozen=True)
class CellConfig:
    """Hyperparameters fixed a priori (PILOT_I2.md, Sec. 4)."""

    k: int = 10
    alpha_llgc: float = 0.99
    alpha_prior: float = 0.9
    loop_iterations: int = 5


@dataclass
class CellResult:
    """Predictions on the unlabeled points, diagnostics and fallback counts."""

    predictions: dict[str, NDArray[np.int64]]
    diagnostics: dict[str, float]
    fallbacks: dict[str, int] = field(default_factory=dict)


def _argmax_with_fallback(
    F: NDArray[np.float64],
    fallback: NDArray[np.int64],
) -> tuple[NDArray[np.int64], int]:
    """Argmax per row; rows with no propagated mass take the fallback label."""
    empty = F.max(axis=1) <= 0
    return np.where(empty, fallback, F.argmax(axis=1)), int(empty.sum())


def _diagnostics(
    W: sparse.csr_matrix,
    z: NDArray[np.int64],
    proba_unl: NDArray[np.float64],
    agree: float,
    n_classes: int,
) -> dict[str, float]:
    i, j = sparse.triu(W, k=1).nonzero()
    n_components = connected_components(W, directed=False)[0]
    return {
        "D1": float(np.mean(z[i] == z[j])) if i.size else 1.0,
        "D2": float(proba_unl.max(axis=1).mean()),
        "D3": agree,
        "D4": n_components / n_classes,
    }


def run_cell(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    labeled: NDArray[np.int64],
    unlabeled: NDArray[np.int64],
    seed: int,
    W_feat: sparse.csr_matrix,
    config: CellConfig,
) -> CellResult:
    """Run all arms of PILOT_I2.md on one split."""
    y_lab = y[labeled]
    n_classes = int(y_lab.max()) + 1
    Y0 = np.zeros((len(X), n_classes))
    Y0[labeled, y_lab] = 1.0

    tabpfn = fm.make_tabpfn(seed)
    p_fm1 = fm.fit_predict_proba(tabpfn, X[labeled], y_lab, X[unlabeled])
    emb = fm.tabpfn_embeddings(tabpfn, X)
    p_fm2 = fm.fit_predict_proba(fm.make_tabicl(seed), X[labeled], y_lab, X[unlabeled])
    p_fm3 = fm.kumo_predict_proba(X[labeled], y_lab, X[unlabeled], seed)
    p_st = fm.loop_fm(X[labeled], y_lab, X[unlabeled], seed, config.loop_iterations)
    fm1_pred = p_fm1.argmax(axis=1)

    prior = Y0.copy()
    prior[unlabeled] = p_fm1
    prior_kumo = Y0.copy()
    prior_kumo[unlabeled] = p_fm3
    W_emb = mutual_knn_graph(zscore(emb), config.k)

    predictions = {
        "FM1": fm1_pred,
        "FM2": p_fm2.argmax(axis=1),
        "FM3": p_fm3.argmax(axis=1),
        "ST": p_st.argmax(axis=1),
    }
    fallbacks: dict[str, int] = {}
    propagated = {
        "G1": llgc(W_feat, Y0, config.alpha_llgc),
        "H1": llgc(W_emb, Y0, config.alpha_llgc),
        "H2": llgc(W_feat, prior, config.alpha_prior),
        "H3": llgc(W_emb, prior, config.alpha_prior),
        "H4": llgc(W_feat, prior_kumo, config.alpha_prior),
    }
    for arm, F in propagated.items():
        predictions[arm], fallbacks[arm] = _argmax_with_fallback(F[unlabeled], fm1_pred)
    F_g2, fallbacks["G2"] = gfhf(W_feat, labeled, y_lab, n_classes, fallback=prior)
    predictions["G2"] = F_g2[unlabeled].argmax(axis=1)

    z = np.empty(len(X), dtype=np.int64)
    z[labeled], z[unlabeled] = y_lab, fm1_pred
    agree = float(np.mean(fm1_pred == predictions["H1"]))
    diagnostics = _diagnostics(W_feat, z, p_fm1, agree, n_classes)
    return CellResult(predictions, diagnostics, fallbacks)


def score_cell(
    result: CellResult,
    y_unlabeled: NDArray[np.int64],
    n_classes: int,
) -> dict[str, dict[str, float]]:
    """Return macro-F1 and accuracy of every arm on the unlabeled points."""
    labels = list(range(n_classes))
    return {
        arm: {
            "macro_f1": float(
                f1_score(
                    y_unlabeled, pred, labels=labels, average="macro", zero_division=0
                ),
            ),
            "accuracy": float(accuracy_score(y_unlabeled, pred)),
        }
        for arm, pred in result.predictions.items()
    }
