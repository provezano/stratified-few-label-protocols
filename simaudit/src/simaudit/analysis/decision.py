"""Pre-registered pilot decision (PILOT_I2.md, Sec. 5 and 6)."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DIAGNOSTICS = ("D1", "D2", "D3", "D4")
ROPE_KILL_MEAN = 0.005
BIG_GAIN = 0.02
KILL_SHARE = 0.20
GO_SHARE = 0.25
GO_AUC = 0.70


class Outcome(Enum):
    """Pilot outcome."""

    KILL = "KILL"
    GO = "GO"
    GREY = "GREY"


@dataclass(frozen=True)
class LodoResult:
    """Leave-one-dataset-out evaluation of the diagnostic rule."""

    auc: float
    regret_rule: float
    regret_always_fm: float
    regret_always_hybrid: float
    choices: pd.Series

    @property
    def beats_baselines(self) -> bool:
        """Whether the rule's regret is below both fixed policies."""
        return self.regret_rule < min(self.regret_always_fm, self.regret_always_hybrid)


def paired_deltas(
    scores: pd.DataFrame,
    fms: tuple[str, ...],
    hybrids: tuple[str, ...],
) -> tuple[pd.DataFrame, str]:
    """Return Δ = best hybrid − best FM per (dataset, m), and the best hybrid.

    The best FM is chosen per dataset by mean score over m; the best hybrid is
    the single hybrid with the highest mean Δ over all cells.
    """
    wide = scores.pivot_table(index=["dataset", "m"], columns="arm", values="macro_f1")
    best_fm = wide[list(fms)].groupby(level="dataset").mean().idxmax(axis=1)
    fm_score = pd.Series(
        [wide.loc[idx, best_fm[idx[0]]] for idx in wide.index],
        index=wide.index,
    )
    gains = wide[list(hybrids)].sub(fm_score, axis=0)
    hybrid = str(gains.mean().idxmax())
    table = pd.DataFrame(
        {
            "delta": gains[hybrid],
            "best_fm": best_fm.reindex(
                wide.index.get_level_values("dataset"),
            ).to_numpy(),
        },
        index=wide.index,
    ).reset_index()
    return table, hybrid


def classify_outcome(
    deltas: pd.DataFrame,
    rule_beats_baselines: bool,
    auc: float,
) -> Outcome:
    """Apply the KILL / GO / GREY thresholds of PILOT_I2.md, Sec. 6."""
    share_big = float((deltas.groupby("dataset").delta.max() > BIG_GAIN).mean())
    if deltas.delta.mean() <= ROPE_KILL_MEAN and share_big < KILL_SHARE:
        return Outcome.KILL
    if share_big >= GO_SHARE and rule_beats_baselines and auc >= GO_AUC:
        return Outcome.GO
    return Outcome.GREY


def _features(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([frame[list(DIAGNOSTICS)], np.log(frame.m)])


def lodo_rule(cells: pd.DataFrame) -> LodoResult:
    """Fit the logistic rule leaving one dataset out at a time.

    ``cells`` has one row per (dataset, m) with the diagnostics, ``f1_fm`` and
    ``f1_hybrid``. The rule uses the hybrid iff the predicted P(gain > 0) > 0.5.
    """
    target = (cells.f1_hybrid > cells.f1_fm).to_numpy()
    prob = np.empty(len(cells))
    for dataset in cells.dataset.unique():
        test = (cells.dataset == dataset).to_numpy()
        train_y = target[~test]
        if np.unique(train_y).size < 2:
            prob[test] = train_y.mean()
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression())
        model.fit(_features(cells[~test]), train_y)
        prob[test] = model.predict_proba(_features(cells[test]))[:, 1]
    use_hybrid = prob > 0.5
    best = np.maximum(cells.f1_fm, cells.f1_hybrid)
    chosen = np.where(use_hybrid, cells.f1_hybrid, cells.f1_fm)
    auc = roc_auc_score(target, prob) if np.unique(target).size == 2 else float("nan")
    return LodoResult(
        auc=float(auc),
        regret_rule=float(np.mean(best - chosen)),
        regret_always_fm=float(np.mean(best - cells.f1_fm)),
        regret_always_hybrid=float(np.mean(best - cells.f1_hybrid)),
        choices=pd.Series(use_hybrid, index=cells.index),
    )
