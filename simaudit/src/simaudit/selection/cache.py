"""Disk cache for selections that are costly to recompute (ZCore).

A selection depends only on (X, budget, seed), so every learner of a Study 4
run reuses the same rows. Files are ``{name}_{hash of X}_{budget}_{seed}.npy``.
"""

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from simaudit.settings import settings

Strategy = Callable[[NDArray[np.float64], int, int], NDArray[np.int64]]


def cache_key(X: NDArray[np.float64]) -> str:
    """SHA-256 over the shape and the float64 bytes of X."""
    data = np.ascontiguousarray(X, dtype=np.float64)
    digest = hashlib.sha256(repr(data.shape).encode())
    digest.update(data.tobytes())
    return digest.hexdigest()


def default_dir() -> Path:
    """Where cached selections live, beside the dataset cache (outside git)."""
    return settings.data_cache_dir / "selections"


def disk_cached(
    name: str, strategy: Strategy, cache_dir: Callable[[], Path] | None = None
) -> Strategy:
    """Wrap a strategy so each (X, budget, seed) is computed once and reloaded."""

    def run(X: NDArray[np.float64], budget: int, seed: int) -> NDArray[np.int64]:
        directory = cache_dir() if cache_dir else default_dir()
        path = directory / f"{name}_{cache_key(X)[:32]}_{budget}_{seed}.npy"
        if path.exists():
            return np.asarray(np.load(path), dtype=np.int64)
        chosen = strategy(X, budget, seed)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp, chosen)
        tmp.replace(path)
        return chosen

    run.__name__ = name
    return run
