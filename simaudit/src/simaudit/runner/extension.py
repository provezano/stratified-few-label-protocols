"""Amendment 2 arms on the pilot's own splits: TabFM (FM4/H5), TabPFN v3.5 (FM5/H6)."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from simaudit.arms import fm
from simaudit.data.openml_loader import load
from simaudit.data.splits import draw_split
from simaudit.graph.knn import mutual_knn_graph, zscore
from simaudit.graph.propagation import llgc
from simaudit.runner.cell import (
    CellConfig,
    CellResult,
    _argmax_with_fallback,
    score_cell,
)
from simaudit.runner.pilot import N_REPS, cell_seed


def _tabpfn_v35_predict_proba(
    X_fit: NDArray[np.float64],
    y_fit: NDArray[np.int64],
    X_query: NDArray[np.float64],
    seed: int,
) -> NDArray[np.float64]:
    return fm.fit_predict_proba(fm.make_tabpfn_v35(seed), X_fit, y_fit, X_query)


EXT_MODELS = {"tabfm": ("FM4", "H5"), "v35": ("FM5", "H6")}


def _predictor(model: str) -> Callable[..., NDArray[np.float64]]:
    # looked up at call time so tests can substitute fakes
    return fm.tabfm_predict_proba if model == "tabfm" else _tabpfn_v35_predict_proba


def run_extension_cell(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    labeled: NDArray[np.int64],
    unlabeled: NDArray[np.int64],
    seed: int,
    W_feat: sparse.csr_matrix,
    config: CellConfig,
    model: str,
) -> CellResult:
    """Run one extension FM and its prior-smoothing hybrid on one split."""
    fm_arm, hybrid_arm = EXT_MODELS[model]
    y_lab = y[labeled]
    n_classes = int(y_lab.max()) + 1
    proba = _predictor(model)(X[labeled], y_lab, X[unlabeled], seed)
    fm_pred = proba.argmax(axis=1)
    prior = np.zeros((len(X), n_classes))
    prior[labeled, y_lab] = 1.0
    prior[unlabeled] = proba
    F = llgc(W_feat, prior, config.alpha_prior)[unlabeled]
    hybrid_pred, n_fallback = _argmax_with_fallback(F, fm_pred)
    return CellResult(
        {fm_arm: fm_pred, hybrid_arm: hybrid_pred}, {}, {hybrid_arm: n_fallback}
    )


def run_extension_dataset_m(
    did: int,
    m: int,
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    config: CellConfig,
    model: str,
) -> Path:
    """Score the extension arms for one (dataset, m) on the pilot's splits."""
    out = out_dir / f"{did}_m{m}.parquet"
    if out.exists():
        return out
    data = load(did, cache_dir)
    W_feat = mutual_knn_graph(zscore(data.X), config.k)
    n_classes = int(data.y.max()) + 1
    rows = []
    for rep in range(N_REPS):
        seed = cell_seed(base_seed, did, m, rep)
        lab, unl = draw_split(data.y, m, seed)
        result = run_extension_cell(
            data.X, data.y, lab, unl, seed, W_feat, config, model
        )
        for arm, metrics in score_cell(result, data.y[unl], n_classes).items():
            rows.append(
                dict(
                    dataset=did,
                    m=m,
                    rep=rep,
                    seed=seed,
                    arm=arm,
                    **metrics,
                    fallback=result.fallbacks.get(arm, 0),
                ),
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(out)
    return out
