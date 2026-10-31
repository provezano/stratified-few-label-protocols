"""Mitra-v2 seeding check (exploratory, DEV, logged; source for the P4 pitfall numbers).

AutoGluon 1.6's Mitra no longer seeds each fit, so its random preprocessing (feature
mirroring etc.) draws from the global numpy/torch RNG. This script fits the SAME
cell (same X_fit, y_fit, X_query, seed) R times in one process:

(a) unseeded: ``adapters.mitra_predict`` as in the original DEV run (out/s7a);
(b) seeded:   AutoGluon's ``set_seed(seed)`` before every fit (``run_s7a_v2.seeded_mitra``,
              as the Study 13 worker).

Cells: 8 = DEV tables {1510 (C=2), 42544 (C=8)} x m {1, 5} x {stratified, random},
rep 0, with the stored Study 7 Part (a) labelled sets and seeds
(results/study7/a/{did}_fm2.parquet, read-only). Prediction on all n rows, through
``predict_with_budget`` exactly as run_s7a.py. R = 3 repeats per condition.

Per cell and condition: max |p_r - p_1| over repeats, rows and classes; share of rows
whose argmax differs from repeat 1; max |macro-F1_r - macro-F1_1| on the complement;
``bit_identical`` = every repeat ``np.array_equal`` to repeat 1.

Usage (Mitra venv, 1 worker):
    OMP_NUM_THREADS=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 nice -n 10 \
      .venv_fm_extra_mitra/bin/python paper/p1/exploratory/fm_extra/mitra_seed_check.py
Writes out/mitra_seed/cells.csv, out/mitra_seed/summary.csv, out/mitra_seed/run.log.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import adapters as ad  # noqa: E402
from run_s7a_v2 import seeded_mitra  # noqa: E402
from simaudit.analysis.study7 import complement_metrics  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.selection import strategies as sel  # noqa: E402
from simaudit.settings import settings  # noqa: E402

OUT = HERE / "out" / "mitra_seed"
TABLES = (1510, 42544)
MULTS = (1, 5)
STRATS = ("stratified", "random")
REPEATS = 3
CONDITIONS = {"unseeded": ad.mitra_predict, "seeded": seeded_mitra}


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "run.log").open("a") as f:
        f.write(line + "\n")


def main() -> None:
    import torch

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"start: tables={TABLES} mults={MULTS} strategies={STRATS} rep=0 repeats={REPEATS}")
    rows = []
    for did in TABLES:
        data = load(did, settings.data_cache_dir)
        C = int(data.y.max()) + 1
        stored = pd.read_parquet(settings.results_dir / "study7" / "a" / f"{did}_fm2.parquet",
                                 columns=["strategy", "mult", "rep", "seed", "labeled"])
        for m in MULTS:
            for s in STRATS:
                r = stored[(stored.strategy == s) & (stored.mult == m) & (stored.rep == 0)].iloc[0]
                lab, seed = np.asarray(r.labeled, dtype=np.int64), int(r.seed)
                for cond, fn in CONDITIONS.items():
                    probas, f1s = [], []
                    for _ in range(REPEATS):
                        p = sel.predict_with_budget(fn, data.X, lab, data.y[lab], C, seed)
                        probas.append(p)
                        f1s.append(complement_metrics(data.y, p, lab)["macro_f1"])
                    rest = np.setdiff1d(np.arange(len(data.y)), lab)
                    ref = probas[0]
                    rec = dict(dataset=did, C=C, mult=m, strategy=s, rep=0, seed=seed, condition=cond,
                               repeats=REPEATS, seen=int(len(np.unique(data.y[lab]))),
                               max_abs_dp=float(max(np.abs(q - ref).max() for q in probas[1:])),
                               share_argmax_changed=float(max(
                                   np.mean(q[rest].argmax(1) != ref[rest].argmax(1)) for q in probas[1:])),
                               max_abs_dmacro_f1=float(max(abs(f - f1s[0]) for f in f1s[1:])),
                               bit_identical=bool(all(np.array_equal(q, ref) for q in probas[1:])))
                    rows.append(rec)
                    log(" ".join(f"{k}={v}" for k, v in rec.items()))
    cells = pd.DataFrame(rows)
    cells.to_csv(OUT / "cells.csv", index=False)
    summ = cells.groupby("condition").agg(
        cells=("seed", "size"), fitted_cells=("seen", lambda x: int((x > 1).sum())),
        max_abs_dp=("max_abs_dp", "max"), mean_max_abs_dp=("max_abs_dp", "mean"),
        max_share_argmax_changed=("share_argmax_changed", "max"),
        max_abs_dmacro_f1=("max_abs_dmacro_f1", "max"),
        bit_identical_cells=("bit_identical", "sum")).reset_index()
    summ.to_csv(OUT / "summary.csv", index=False)
    log("summary:\n" + summ.to_string(index=False))


if __name__ == "__main__":
    main()
