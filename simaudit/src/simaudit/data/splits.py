"""Labeled/unlabeled splits with exactly m labeled points per class."""

import numpy as np
from numpy.typing import NDArray


def draw_split(
    y: NDArray[np.int64],
    m: int,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Draw m labeled points per class; return sorted (labeled, unlabeled) indices.

    Raises:
        ValueError: if some class has m points or fewer, which would leave it
            with no unlabeled point to evaluate on.
    """
    rng = np.random.default_rng(seed)
    labeled = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        if members.size <= m:
            raise ValueError(
                f"class {c} has {members.size} points: no point left unlabeled "
                f"with m={m}",
            )
        labeled.append(rng.choice(members, size=m, replace=False))
    lab = np.sort(np.concatenate(labeled))
    unl = np.setdiff1d(np.arange(y.size), lab)
    return lab, unl
