"""Study 4 analysis (prereg/STUDY4_LABEL_SELECTION.md, Sec. 4 and Amendment 1)."""

from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

from simaudit.analysis.report import bootstrap_ci, holm

KEYS = ["dataset", "mult", "learner"]
BASELINE = "random"


def dataset_means(raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (dataset, mult, learner, strategy): mean over repetitions."""
    return pd.DataFrame(raw.groupby([*KEYS, "strategy"], as_index=False)[metric].mean())


def paired_deltas(raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-dataset Δ = strategy − realistic random, for every non-random strategy."""
    means = dataset_means(raw, metric)
    base = means[means.strategy == BASELINE].drop(columns="strategy")
    other = means[means.strategy != BASELINE]
    merged = other.merge(base, on=KEYS, suffixes=("", "_random"))
    merged["delta"] = merged[metric] - merged[f"{metric}_random"]
    return merged[[*KEYS, "strategy", "delta"]]


def tests_table(raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean Δ, bootstrap CI, Wilcoxon p and Holm within (mult, learner)."""
    rows = []
    for (mult, learner, strategy), g in paired_deltas(raw, metric).groupby(
        ["mult", "learner", "strategy"]
    ):
        values = g.delta.to_numpy()
        p = wilcoxon(values).pvalue if np.any(values != 0) else 1.0
        low, high = bootstrap_ci(values)
        rows.append(
            dict(
                mult=mult,
                learner=learner,
                strategy=strategy,
                mean_delta=values.mean(),
                ci_low=low,
                ci_high=high,
                wins=int((values > 0).sum()),
                n=len(values),
                p=float(p),
            )
        )
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    for _, idx in table.groupby(["mult", "learner"]).groups.items():
        table.loc[idx, "p_holm"] = holm(table.loc[idx, "p"].tolist())
    return table


def risk_table(raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per strategy: mean across-draw std, 10% quantile and worst regret vs random."""
    spread = raw.groupby([*KEYS, "strategy"])[metric].agg(["std", "mean"])
    q10 = raw.groupby([*KEYS, "strategy"])[metric].quantile(0.1).rename("q10")
    frame = spread.join(q10).reset_index()
    base = frame[frame.strategy == BASELINE][[*KEYS, "mean"]]
    frame = frame.merge(base, on=KEYS, suffixes=("", "_random"))
    frame["regret"] = frame["mean_random"] - frame["mean"]
    return frame.groupby(["mult", "learner", "strategy"], as_index=False).agg(
        std=("std", "mean"), q10=("q10", "mean"), max_regret=("regret", "max")
    )


def _score(
    y: NDArray[np.int64], pred: NDArray[np.int64], rows: NDArray[np.int64], metric: str
) -> float:
    if metric == "accuracy":
        return float((pred[rows] == y[rows]).mean())
    labels = list(range(int(y.max()) + 1))
    return float(
        f1_score(y[rows], pred[rows], labels=labels, average="macro", zero_division=0)
    )


def split_half_oracle(
    raw: pd.DataFrame,
    truth: dict[int, NDArray[np.int64]],
    metric: str,
    seed: int = 0,
) -> pd.DataFrame:
    """Best-of-R random draw chosen on one half of the rows, scored on the other.

    Rows of each (dataset, mult, learner) are split once by ``seed``; the draw
    chosen on half A is scored on half B and vice versa, and the two are averaged.
    A row selected by a draw is excluded from that draw's scoring.
    """
    out = []
    rng = np.random.default_rng(seed)
    rand = raw[raw.strategy == BASELINE]
    for key, g in rand.groupby(KEYS):
        y = truth[int(cast(Any, key)[0])]
        half = rng.permutation(len(y)) % 2
        scores = np.array(
            [
                [
                    _score(
                        y,
                        np.asarray(r.pred),
                        np.setdiff1d(np.flatnonzero(half == h), cast(Any, r.labeled)),
                        metric,
                    )
                    for h in (0, 1)
                ]
                for r in g.itertuples()
            ]
        )
        pick_on_a, pick_on_b = scores[:, 0].argmax(), scores[:, 1].argmax()
        out.append(
            dict(
                zip(KEYS, key),
                split_half=(scores[pick_on_a, 1] + scores[pick_on_b, 0]) / 2,
                in_sample_best=float(g[metric].max()),
                mean_random=float(g[metric].mean()),
            )
        )
    return pd.DataFrame(out).set_index("dataset", drop=False)


def never_selected_rows(
    frame: pd.DataFrame, dataset: int, mult: int, n: int
) -> NDArray[np.int64]:
    """Rows that no strategy labeled for this (dataset, mult) in this frame."""
    sub = frame[(frame.dataset == dataset) & (frame.mult == mult)]
    chosen = set().union(*map(set, sub.labeled)) if len(sub) else set()
    return np.array(sorted(set(range(n)) - chosen), dtype=np.int64)


def rescore_on(
    frame: pd.DataFrame, truth: dict[int, NDArray[np.int64]], metric: str
) -> pd.DataFrame:
    """Sensitivity: rescore on rows no strategy selected in the same repetition."""
    rows = []
    for (d, mm, _), g in frame.groupby(["dataset", "mult", "rep"]):
        dataset, mult = int(cast(Any, d)), int(cast(Any, mm))
        y = truth[dataset]
        common = never_selected_rows(g, dataset, mult, len(y))
        for r in g.itertuples():
            rows.append(
                dict(
                    dataset=dataset,
                    mult=mult,
                    learner=r.learner,
                    strategy=r.strategy,
                    rep=r.rep,
                    **{
                        f"{metric}_common": _score(
                            y, np.asarray(r.pred), common, metric
                        )
                    },
                )
            )
    return pd.DataFrame(rows)
