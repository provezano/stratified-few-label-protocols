"""Study 10 analysis: the non-inferiority tests and the Sec. 8 decision table.

prereg/STUDY10_SELECTOR_CONFIRMATION.md Secs. 7-8. A contrast Δ_d (F1 minus a
comparator, one value per dataset) is NI-established when its one-sided
Wilcoxon test of Δ + δ > 0 is Holm-significant within its family **and** the
one-sided 95 % bootstrap lower bound of its mean exceeds −δ.
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from simaudit.analysis.report import holm

MARGIN = 0.01
ALPHA = 0.05
N_BOOT = 10_000
H1 = ("H1a", "H1b", "H1c", "H1d")  # LUCoS+FPS (F1, acc), k-medoids (F1, acc)
PAIRS = (("H1a", "H1b"), ("H1c", "H1d"))
REFUTE_PROB = 0.95
CONFIRM_PROB = 0.05


def ni_test(
    values: NDArray[np.float64], margin: float = MARGIN, seed: int = 0
) -> dict[str, float]:
    """One-sided NI test: Wilcoxon (zsplit) of Δ + δ > 0 and bootstrap lower bound."""
    v = np.asarray(values, dtype=np.float64)
    shifted = v + margin
    if np.any(shifted != 0):
        p = float(
            stats.wilcoxon(shifted, alternative="greater", zero_method="zsplit").pvalue
        )
    else:
        p = 1.0
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(N_BOOT, v.size)).mean(axis=1)
    return dict(
        p=p,
        lower=float(np.quantile(means, ALPHA)),
        upper=float(np.quantile(means, 1 - ALPHA)),
        mean=float(v.mean()),
        n=float(v.size),
    )


def ni_family(
    deltas: dict[str, NDArray[np.float64]], margin: float = MARGIN
) -> pd.DataFrame:
    """NI tests of one Holm family; ``established`` = Holm p < α and lower > −δ."""
    table = pd.DataFrame({k: ni_test(v, margin) for k, v in deltas.items()}).T
    table["p_holm"] = holm(table["p"].tolist())
    table["established"] = (table["p_holm"] < ALPHA) & (table["lower"] > -margin)
    return table


def _passes(row: dict[str, Any], margin: float) -> bool:
    return bool(row["established"]) and min(row["mean_ho"], row["mean_t1"]) > -margin


def decide(rows: dict[str, dict[str, Any]], margin: float = MARGIN) -> str:
    """Sec. 8 outcome from the four H1 rows (refuted is checked first)."""
    h1 = [rows[k] for k in H1]
    if any(r["p_worse"] >= REFUTE_PROB or r["upper"] < -margin for r in h1):
        return "refuted"
    if all(
        _passes(r, margin) and r["lodo_min"] > -margin and r["p_worse"] < CONFIRM_PROB
        for r in h1
    ):
        return "confirmed"
    if any(all(_passes(rows[k], margin) for k in pair) for pair in PAIRS):
        return "partially confirmed"
    return "not shown"
