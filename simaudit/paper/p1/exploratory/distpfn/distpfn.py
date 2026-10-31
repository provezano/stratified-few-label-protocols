"""DistPFN and DistPFN-T (Lee, ICML 2026, PMLR 306), reimplemented from the paper.

Post-hoc test-time rescaling of in-context-learner posteriors (Sec. 4.2-4.3, Alg. 1,
App. E.2). Default variant: the test-set average posterior ``pbar`` replaces the
per-instance factor. Columns are the classes seen in the context; ``prior`` is the
context class share (strictly positive).
"""

from __future__ import annotations

import numpy as np


def _norm(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(q.sum(axis=-1, keepdims=True), 1e-300)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def temperature(pbar: np.ndarray, prior: np.ndarray) -> float:
    """tau = CE(pbar, prior) = -sum_c pbar(c) log prior(c)."""
    return float(-(pbar * np.log(prior)).sum())


def distpfn(p: np.ndarray, prior: np.ndarray, pbar: np.ndarray | None = None) -> np.ndarray:
    """Norm(p * pbar / prior); pbar defaults to the mean of p over the rows (test set)."""
    pbar = p.mean(axis=0) if pbar is None else pbar
    return _norm(p * (pbar / prior))


def distpfn_t(p: np.ndarray, prior: np.ndarray, pbar: np.ndarray | None = None) -> np.ndarray:
    """Norm(p * softmax(pbar / tau) / prior), tau = CE(pbar, prior)."""
    pbar = p.mean(axis=0) if pbar is None else pbar
    p_t = _softmax(pbar / temperature(pbar, prior))
    return _norm(p * (p_t / prior))


def distpfn_inst(p: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Per-instance form (Eq. 4): Norm(p^2 / prior)."""
    return _norm(p * p / prior)


def distpfn_t_inst(p: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Per-instance DistPFN-T (the form of the paper's Table 4 example)."""
    return np.stack([distpfn_t(row[None], prior, pbar=row)[0] for row in p])
