"""Tabular foundation-model arms: TabPFN v2, TabICL and a looped TabPFN."""

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import sdm
import torch
from numpy.typing import NDArray
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


def make_tabpfn(seed: int) -> Any:
    """Return TabPFN v2 (Apache-2.0 weights) on CPU, for bit-reproducible output."""
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        device="cpu",
        random_state=seed,
    )


def make_tabicl(seed: int) -> Any:
    """Return TabICL (BSD-3) with default settings on CPU."""
    return TabICLClassifier(device="cpu", random_state=seed)


@lru_cache(maxsize=1)
def _kumo_model() -> Any:
    return sdm.models.KumoTabular(task="classification", size="large", device="cpu")


@lru_cache(maxsize=1)
def _tabfm_model() -> Any:
    # TabFM Non-Commercial License v1.0, accepted by the study owner on 2026-10-20
    # for research use only (prereg/PILOT_I2.md, Amendment 2). Never ship.
    return sdm.models.TabFM(task="classification", device="cpu", accept_license=True)


def _sdm_predict_proba(
    model: Any,
    X_fit: NDArray[np.float64],
    y_fit: NDArray[np.int64],
    X_query: NDArray[np.float64],
    seed: int,
    n_estimators: int,
) -> NDArray[np.float64]:
    """In-context prediction with an ``sdm`` model; columns ordered by class index."""
    columns = [f"f{i}" for i in range(X_fit.shape[1])]
    frame = pd.DataFrame(np.vstack([X_fit, X_query]), columns=columns)
    frame["target"] = np.r_[y_fit, np.full(len(X_query), y_fit[0])].astype(str)
    table = sdm.TableTensor.from_pandas(
        df=frame,
        stypes=sdm.infer_stypes(frame, overrides={"target": "categorical"}),
        device="cpu",
    )
    n = len(X_fit)
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = model(
            x_context=table[:n].drop_columns("target"),
            y_context=table[:n, "target"],
            x_query=table[n:].drop_columns("target"),
            num_estimators=n_estimators,
        ).to_pandas()
    n_classes = int(y_fit.max()) + 1
    return np.column_stack([out[str(c)].to_numpy(float) for c in range(n_classes)])


def kumo_predict_proba(
    X_fit: NDArray[np.float64],
    y_fit: NDArray[np.int64],
    X_query: NDArray[np.float64],
    seed: int,
    n_estimators: int = 8,
) -> NDArray[np.float64]:
    """Return Kumo-Tabular (NVIDIA, OpenMDW-1.1) probabilities by class index."""
    return _sdm_predict_proba(_kumo_model(), X_fit, y_fit, X_query, seed, n_estimators)


def tabfm_predict_proba(
    X_fit: NDArray[np.float64],
    y_fit: NDArray[np.int64],
    X_query: NDArray[np.float64],
    seed: int,
    n_estimators: int = 8,
) -> NDArray[np.float64]:
    """Return TabFM (Google weights, research-only licence) probabilities by class."""
    return _sdm_predict_proba(_tabfm_model(), X_fit, y_fit, X_query, seed, n_estimators)


def make_tabpfn_v35(seed: int) -> Any:
    """Return TabPFN v3.5 on CPU (gated non-commercial weights; needs TABPFN_TOKEN)."""
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V3_5,
        device="cpu",
        random_state=seed,
    )


def fit_predict_proba(
    model: Any,
    X_fit: NDArray[np.float64],
    y_fit: NDArray[np.int64],
    X_query: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fit on (X_fit, y_fit) and return class probabilities for X_query."""
    model.fit(X_fit, y_fit)
    return np.asarray(model.predict_proba(X_query), dtype=np.float64)


def tabpfn_embeddings(model: Any, X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Embed X as test tokens of a fitted TabPFN, averaged over estimators."""
    emb = np.asarray(model.get_embeddings(X, data_source="test"), dtype=np.float64)
    return emb.mean(axis=0) if emb.ndim == 3 else emb


def _risk(proba: NDArray[np.float64]) -> float:
    return float(np.mean(1.0 - proba.max(axis=1)))


def loop_fm(
    X_lab: NDArray[np.float64],
    y_lab: NDArray[np.int64],
    X_unl: NDArray[np.float64],
    seed: int,
    iterations: int,
) -> NDArray[np.float64]:
    """Run LoopTabFM (Li et al. 2025, Alg. 1), adapted to multiclass.

    Each loop refits TabPFN v2 on the labeled points plus the unlabeled points
    with hard pseudo-labels. The returned probabilities are those of the loop
    with the lowest risk, mean(1 - max p).
    """
    proba = fit_predict_proba(make_tabpfn(seed), X_lab, y_lab, X_unl)
    best, best_risk = proba, _risk(proba)
    X_fit = np.vstack([X_lab, X_unl])
    for _ in range(iterations):
        pseudo = proba.argmax(axis=1)
        y_fit = np.concatenate([y_lab, pseudo])
        proba = fit_predict_proba(make_tabpfn(seed), X_fit, y_fit, X_unl)
        if _risk(proba) < best_risk:
            best, best_risk = proba, _risk(proba)
    return best
