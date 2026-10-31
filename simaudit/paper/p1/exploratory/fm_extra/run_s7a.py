"""Exploratory replication of Study 7 Part (a) on DEV with one extra FM.

Same labelled sets, seeds and protocol as fm1-fm5: the (strategy, mult, rep, seed,
labeled) rows are read from the stored ``results/study7/a/{did}_fm2.parquet``
(read-only; the selections are learner-independent). Each fit goes through
``selection.strategies.predict_with_budget`` and is scored with
``analysis.study7.complement_metrics``, i.e. ``runner.study7._fit`` re-implemented
without importing the runner (it pulls in fm1-fm5's packages). Full X, no column drop,
as in the main Part (a) run.

Usage (from simaudit/, in the model's venv, one process = one worker):
    OMP_NUM_THREADS=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 nice -n 10 \
      .venv_fm_extra/bin/python paper/p1/exploratory/fm_extra/run_s7a.py fm6 \
      [--strategies stratified,random] [--datasets 1510,42544] [--out out/s7a]

Output: ``{out}/{did}_{fm}.parquet`` (atomic, resumable from ``*.partial.parquet``,
skipped when present). Nothing is written outside paper/p1/exploratory/fm_extra/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import adapters as ad  # noqa: E402
from simaudit.analysis.study7 import complement_metrics  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.selection import strategies as sel  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

S7A = settings.results_dir / "study7" / "a"
REFERENCE_FM = "fm2"
CORE = ("stratified", "random")
ALL = ("stratified", "random", "kmedoids", "kmeans_central", "typiclust", "lucos")
CHECKPOINT_EVERY = 10


def dev_ids() -> list[int]:
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_i2.yaml").read_text())
    return [int(d["did"]) for d in pool["dev"]]


def stored_cells(did: int, strategies: tuple[str, ...], mults: tuple[int, ...] = (1, 3, 5, 10)) -> pd.DataFrame:
    """The Part (a) labelled sets of one table, in the stored order."""
    cols = ["strategy", "mult", "rep", "seed", "labeled"]
    df = pd.read_parquet(S7A / f"{did}_{REFERENCE_FM}.parquet", columns=cols)
    df = df[df.strategy.isin(strategies) & df.mult.isin(mults)]
    order = {s: i for i, s in enumerate(strategies)}
    return df.sort_values(["strategy", "mult", "rep"], key=lambda c: c.map(order) if c.name == "strategy" else c)


def fit_one(predict, X, y, labeled: np.ndarray, seed: int) -> dict:
    """``runner.study7._fit`` without the runner import (same fields)."""
    n_classes = int(y.max()) + 1
    start = time.perf_counter()
    proba = sel.predict_with_budget(predict, X, labeled, y[labeled], n_classes, seed)
    return dict(
        n_classes=n_classes,
        labeled=labeled.tolist(),
        proba=proba.astype(np.float32).ravel(),
        coverage=sel.class_coverage(y, labeled),
        seen=np.unique(y[labeled]).tolist(),
        seconds=time.perf_counter() - start,
        **complement_metrics(y, proba, labeled),
    )


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def run_table(fm: str, did: int, strategies: tuple[str, ...], out_dir: Path,
              mults: tuple[int, ...] = (1, 3, 5, 10)) -> Path:
    out = out_dir / f"{did}_{fm}.parquet"
    if out.exists():
        return out
    partial = out.with_name(out.name.replace(".parquet", ".partial.parquet"))
    rows: list[dict] = []
    done: set[str] = set()
    if partial.exists():
        old = pd.read_parquet(partial)
        rows, done = old.to_dict("records"), set(old["_key"])
    predict = ad.PREDICTORS[fm][1]
    data = load(did, settings.data_cache_dir)
    fresh = 0
    for r in stored_cells(did, strategies, mults).itertuples():
        key = f"{r.strategy}|{r.mult}|{r.rep}"
        if key in done:
            continue
        labeled = np.asarray(r.labeled, dtype=np.int64)
        rec = fit_one(predict, data.X, data.y, labeled, int(r.seed))
        rows.append(dict(strategy=r.strategy, mult=int(r.mult), rep=int(r.rep), seed=int(r.seed),
                         budget=len(labeled), _key=key, **rec))
        fresh += 1
        if fresh % CHECKPOINT_EVERY == 0:
            _write(pd.DataFrame(rows), partial)
    frame = pd.DataFrame(rows).drop(columns="_key")
    frame.insert(0, "learner", fm)
    frame.insert(0, "dataset", did)
    _write(frame, out)
    partial.unlink(missing_ok=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fm", choices=sorted(ad.PREDICTORS))
    ap.add_argument("--strategies", default=",".join(CORE))
    ap.add_argument("--datasets", default="")
    ap.add_argument("--mults", default="1,3,5,10")
    ap.add_argument("--out", default=str(HERE / "out" / "s7a"))
    a = ap.parse_args()
    strategies = tuple(a.strategies.split(","))
    dids = [int(x) for x in a.datasets.split(",")] if a.datasets else dev_ids()
    out_dir = Path(a.out)
    for did in dids:
        t = time.perf_counter()
        try:
            run_table(a.fm, did, strategies, out_dir, tuple(int(m) for m in a.mults.split(",")))
            print(f"done {a.fm} {did} {time.perf_counter() - t:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - logged, next table goes on
            print(f"FAILED {a.fm} {did}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
