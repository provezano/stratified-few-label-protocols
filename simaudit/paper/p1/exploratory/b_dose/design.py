"""Context designs of the exploratory dose-response arms F, S and N (DEV only).

EXPLORATORY, not pre-registered. Plan: paper/p1/exploratory/b_dose_response_plan.md.

Notation (as in Study 7 Part (c), prereg/STUDY7_PROTOCOL_MECHANISM.md Sec. 3):
C classes, budget B = mult * C, u = the upweighted class (natural majority or
natural minority), the other C - 1 classes are the "context-minority" classes.

- k(s): per-class count of the other classes in the ORIGINAL fixed-budget design
  at share s, floor((B - n_u(s)) / (C - 1)), at least 1.
- k0 = k(0.5).
- n_u(s, k) = round-half-up(s / (1 - s) * (C - 1) * k), at least 1; the context
  then has share n_u / (n_u + (C - 1) k) ~= s.

Arms (per dataset x mult x rep x upweighted class):
- F: (s, k0) for s in LEVELS, rows drawn independently per level (s = 0.5 is the
  base context); the other classes keep k0 rows, B grows with s.
- N: the same counts as F, but nested: one random permutation per class and per
  (dataset, mult, rep); a context takes the first k rows of each permutation, so
  the context at a higher share contains every row of the lower-share contexts.
- S: (0.5, k(s')) for s' in LEVELS: share fixed at 0.5, the other classes' count
  follows the count that the original design gives them at share s'. Rows drawn
  independently per count (k = k0 is the base context).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

SEED_NAMESPACE = 41  # unused elsewhere (4/5/6/8/10/12/32 are taken)
LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)
ARMS = ("F", "N", "S")


def rhu(x: float) -> int:
    """Round half up, as in Study 7."""
    return int(np.floor(x + 0.5))


def original_other_count(n_classes: int, mult: int, share: float) -> int:
    """Per-class count of the non-upweighted classes in Part (c) at share s."""
    budget = mult * n_classes
    n_up = min(max(rhu(share * budget), 1), budget - (n_classes - 1))
    return max(1, (budget - n_up) // (n_classes - 1))


def upweighted_count(n_classes: int, share: float, k: int) -> int:
    """Rows of u so that u has share ~s when every other class has k rows."""
    return max(1, rhu(share / (1 - share) * (n_classes - 1) * k))


def natural_classes(y: NDArray[np.int64]) -> tuple[int, int]:
    """Natural majority (ties: lowest index) and minority (ties: highest index)."""
    counts = np.bincount(y)
    majority = int(np.argmax(counts))
    minority = int(len(counts) - 1 - np.argmin(counts[::-1]))
    return majority, minority


def _seed(*parts: int) -> int:
    state = np.random.SeedSequence([int(p) for p in parts])
    return int(state.generate_state(1)[0] % (2**31 - 1))


@dataclass(frozen=True)
class Cell:
    """One (arm, upweighted side, level) design within a (dataset, mult, rep)."""

    arm: str  # F, N or S
    side: str  # majority or minority
    level: float  # s for F/N; s' (the matched original share) for S
    mult: int
    rep: int
    upweighted: int
    counts: tuple[int, ...]
    feasible: bool
    labeled: tuple[int, ...]  # empty if infeasible

    @property
    def share_real(self) -> float:
        total = sum(self.counts)
        return self.counts[self.upweighted] / total if total else float("nan")


def counts_for(n_classes: int, upweighted: int, n_up: int, k: int) -> NDArray[np.int64]:
    counts = np.full(n_classes, k, dtype=np.int64)
    counts[upweighted] = n_up
    return counts


def rep_cells(
    y: NDArray[np.int64], did: int, mult: int, rep: int, base_seed: int
) -> list[Cell]:
    """Every F/N/S cell of one (dataset, mult, rep)."""
    n_classes = int(y.max()) + 1
    available = np.bincount(y, minlength=n_classes).astype(np.int64)
    majority, minority = natural_classes(y)
    perm_rng = np.random.default_rng([base_seed, did, mult, rep, SEED_NAMESPACE, 0])
    perms = [perm_rng.permutation(np.flatnonzero(y == c)) for c in range(n_classes)]
    k0 = original_other_count(n_classes, mult, 0.5)

    def prefix(counts: NDArray[np.int64]) -> tuple[int, ...]:
        rows = np.concatenate([perms[c][: counts[c]] for c in range(n_classes)])
        return tuple(int(i) for i in np.sort(rows))

    def independent(counts: NDArray[np.int64], tag: int, level: float) -> tuple[int, ...]:
        seed = _seed(
            base_seed, did, mult, rep, round(100 * level), *counts, SEED_NAMESPACE, tag
        )
        rng = np.random.default_rng(seed)
        rows = np.concatenate(
            [
                rng.choice(np.flatnonzero(y == c), size=int(counts[c]), replace=False)
                for c in range(n_classes)
            ]
        )
        return tuple(int(i) for i in np.sort(rows))

    cells = []
    for (side, up), arm, level in product(
        (("majority", majority), ("minority", minority)), ARMS, LEVELS
    ):
        if arm == "S":
            k = original_other_count(n_classes, mult, level)
            counts = counts_for(n_classes, up, upweighted_count(n_classes, 0.5, k), k)
            base = k == k0
        else:
            counts = counts_for(n_classes, up, upweighted_count(n_classes, level, k0), k0)
            base = level == 0.5
        feasible = bool(np.all(counts <= available - 1))
        labeled: tuple[int, ...] = ()
        if feasible:
            if base or arm == "N":
                labeled = prefix(counts)
            else:
                labeled = independent(counts, 1 if arm == "F" else 2, level if arm == "F" else 0.5)
        cells.append(
            Cell(arm, side, level, mult, rep, up, tuple(int(c) for c in counts), feasible, labeled)
        )
    return cells


def fm_seed(base_seed: int, did: int, mult: int, rep: int) -> int:
    """One FM seed per (dataset, mult, rep): common random numbers across cells."""
    return _seed(base_seed, did, mult, rep, SEED_NAMESPACE, 7)
