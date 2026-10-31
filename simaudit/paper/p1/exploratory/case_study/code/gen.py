"""Generators (TabEBM, SMOTE) and downstream classical predictors for the case study.

Safety: torch.load is forced to weights_only=True, and outbound sockets are blocked,
so neither TabPFN 2.1.2 nor TabEBM can download or unpickle arbitrary objects.
"""

from __future__ import annotations

import socket

import cs_common as cc  # noqa: F401  (sets thread and offline env first)
import numpy as np
import torch

torch.set_num_threads(1)
_orig_load = torch.load


def _safe_load(*a, **k):
    k["weights_only"] = True
    return _orig_load(*a, **k)


torch.load = _safe_load


def _no_net(*a, **k):
    raise RuntimeError("network access blocked in the case study")


socket.socket.connect = _no_net  # type: ignore[assignment]
socket.create_connection = _no_net  # type: ignore[assignment]


def tabebm_generate(X_tr: np.ndarray, y_tr: np.ndarray, counts: dict[int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One fresh TabEBM per class (its per-instance fit cache is keyed by class only).

    The class-k EBM uses only the rows of class k (as in TabEBM._get_or_create_ebm_dataset),
    so calling it per class with label 0 is equivalent and allows class-specific sample counts.
    """
    from tabebm.TabEBM import TabEBM

    Xs, ys = [], []
    for k, s_k in sorted(counts.items()):
        if s_k <= 0:
            continue
        Xk = X_tr[y_tr == k]
        eb = TabEBM()
        eb.model.model_path = cc.TABPFN_CKPT
        out = eb.generate(Xk, np.zeros(len(Xk), dtype=int), num_samples=int(s_k), seed=int((seed + 7919 * k) % (2**31)))
        g = np.asarray(out["class_0"], dtype=np.float32).reshape(int(s_k), X_tr.shape[1])
        Xs.append(g)
        ys.append(np.full(int(s_k), k, dtype=int))
    if not Xs:
        return np.zeros((0, X_tr.shape[1]), np.float32), np.zeros(0, int)
    return np.vstack(Xs), np.concatenate(ys)


def smote_generate(X_tr, y_tr, counts, seed):
    """SMOTE per class, k_neighbors = min(5, n_k - 1); classes with n_k < 2 get nothing."""
    from imblearn.over_sampling import SMOTE

    Xs, ys, n_fallback = [], [], 0
    for k, s_k in sorted(counts.items()):
        n_k = int((y_tr == k).sum())
        if s_k <= 0:
            continue
        if n_k < 2 or len(np.unique(y_tr)) < 2:
            n_fallback += 1
            continue
        sm = SMOTE(sampling_strategy={k: n_k + int(s_k)}, k_neighbors=min(5, n_k - 1), random_state=int(seed % (2**31)))
        Xo, yo = sm.fit_resample(X_tr, y_tr)
        Xs.append(Xo[len(X_tr):].astype(np.float32))
        ys.append(yo[len(y_tr):].astype(int))
    if not Xs:
        return np.zeros((0, X_tr.shape[1]), np.float32), np.zeros(0, int), n_fallback
    return np.vstack(Xs), np.concatenate(ys), n_fallback


def make_classical(name: str, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier

    s = int(seed % (2**31))
    return {
        "lr": lambda: LogisticRegression(),
        "knn": lambda: KNeighborsClassifier(),
        "mlp": lambda: MLPClassifier(random_state=s),
        "rf": lambda: RandomForestClassifier(random_state=s, n_jobs=1),
        "xgb": lambda: XGBClassifier(random_state=s, n_jobs=1, nthread=1),
    }[name]()


def fit_predict(name: str, X_fit, y_fit, X_test, seed: int) -> np.ndarray:
    """Encode the seen classes 0..K-1; one seen class -> constant prediction (Sec. 3)."""
    import warnings

    seen = np.unique(y_fit)
    if len(seen) == 1:
        return np.full(len(X_test), seen[0])
    enc = np.searchsorted(seen, y_fit)
    m = make_classical(name, seed)
    if name == "knn" and len(y_fit) < 5:
        m.set_params(n_neighbors=len(y_fit))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X_fit, enc)
        p = m.predict(X_test)
    return seen[np.asarray(p, int)]
