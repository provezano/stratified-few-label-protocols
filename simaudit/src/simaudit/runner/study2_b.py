"""Study 2, Stage B: honest selection (R2) by cross-validation on labeled rows.

For every (dataset, method, m >= 3, rep) on the Stage A splits, each of the 16
configurations is refit J times with one fold of labeled rows hidden (treated
as unlabeled), and scored by the macro-F1 pooled over the held-out rows
(prereg/STUDY2_BASELINES.md, Sec. 5). The selected configuration's test score
is read from the Stage A output, which used the same splits and seeds.
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from simaudit.data.openml_loader import load
from simaudit.data.splits import draw_split
from simaudit.runner import study2 as s2
from simaudit.runner.pilot import _provenance_name, cell_seed, source_digest

FOLDS = {3: 3, 5: 5, 10: 5}
FOLD_NAMESPACE = 11
STAGE_B_M = (3, 5, 10)
# Open (method, m) pairs after the a fortiori rule on Stage A (prereg Sec. 9.4).
STAGE_B_PAIRS: list[tuple[str, tuple[int, ...]]] = [
    ("cs_logreg", (3, 5)),
    ("appnp", (3,)),
    ("llgc_fm1", (3, 5, 10)),
    ("cs_fm1", (3, 5)),
    ("llgc_fm5", (3, 5, 10)),
    ("cs_fm5", (3, 5, 10)),
    ("scarf", (3, 5)),
    ("stunt", (3, 5)),  # added once STUNT Stage A finished (Sec. 9.4)
]


def cv_folds(y_lab: NDArray[np.int64], m: int, seed: int) -> list[NDArray[np.int64]]:
    """Positions in the labeled set held out by each fold (Sec. 5).

    Within each class a seeded permutation ranks the labeled rows; fold j holds
    out the ranks j, j + J, ...
    """
    if m not in FOLDS:
        raise ValueError("m = 1 has no honest selection (R2 = R0)" if m == 1 else m)
    J = FOLDS[m]
    rng = np.random.default_rng([seed, FOLD_NAMESPACE])
    rank = np.empty(len(y_lab), dtype=np.int64)
    for c in np.unique(y_lab):
        rows = np.flatnonzero(y_lab == c)
        rank[rows[rng.permutation(len(rows))]] = np.arange(len(rows))
    return [np.flatnonzero(rank % J == j) for j in range(J)]


def select_config(cv_f1: Any) -> int:
    """Best pooled CV macro-F1; ties go to the default (0), then the lower index."""
    return int(np.argmax(np.asarray(cv_f1, dtype=np.float64)))


def _cv_scores(
    method: str,
    grid: list[Any],
    X: NDArray[np.float64],
    lab: NDArray[np.int64],
    y_lab: NDArray[np.int64],
    seed: int,
    encoder_for: Any,
) -> list[float]:
    m = len(y_lab) // len(np.unique(y_lab))
    n_classes = int(y_lab.max()) + 1
    pooled: list[list[int]] = [[] for _ in grid]
    truth: list[int] = []
    for held in cv_folds(y_lab, m, seed):
        train = np.setdiff1d(np.arange(len(lab)), held)
        cell = s2._Cell(X, lab[train], y_lab[train], seed)
        truth.extend(y_lab[held].tolist())
        for index, config in enumerate(grid):
            proba = s2._predict(method, config, cell, encoder_for(index, config))
            pooled[index].extend(proba[lab[held]].argmax(axis=1).tolist())
    y_true = np.asarray(truth, dtype=np.int64)
    return [s2._macro_f1(y_true, np.asarray(p), n_classes) for p in pooled]


def run_job(
    did: int,
    method: str,
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    grids: dict[str, list[Any]] | None = None,
    reps: int = s2.STAGE_A_REPS,
    m_values: tuple[int, ...] = STAGE_B_M,
) -> Path:
    """CV scores of every configuration for one (dataset, method)."""
    out = out_dir / f"{did}_{method}_m{'-'.join(map(str, m_values))}.parquet"
    if out.exists():
        return out
    grid = (grids or s2.GRIDS)[method]
    data = load(did, cache_dir)
    n_classes = int(data.y.max()) + 1
    shared: dict[int, Any] = {}
    rows = []
    for rep in range(reps):
        encoders: dict[int, Any] = shared if method == "stunt" else {}

        def encoder_for(index: int, config: Any) -> Any:
            return s2._encoder(
                method, data.X, config, index, encoders, base_seed, did, rep, n_classes
            )

        for m in m_values:
            seed = cell_seed(base_seed, did, m, rep)
            lab, _ = draw_split(data.y, m, seed)
            start = time.perf_counter()
            scores = _cv_scores(
                method, grid, data.X, lab, data.y[lab], seed, encoder_for
            )
            rows.append(
                dict(
                    dataset=did,
                    m=m,
                    rep=rep,
                    seed=seed,
                    method=method,
                    cv_f1=scores,
                    selected=select_config(scores),
                    seconds=time.perf_counter() - start,
                )
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(out)
    return out


def run_stage_b(
    pairs: list[tuple[str, tuple[int, ...]]],
    dataset_ids: list[int],
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    workers: int,
) -> list[Path]:
    """Run the (method, m values) pairs that the a fortiori rule left open."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _provenance_name()).write_text(
        json.dumps(
            dict(
                base_seed=base_seed,
                folds=FOLDS,
                fold_namespace=FOLD_NAMESPACE,
                pairs=pairs,
                datasets=dataset_ids,
                source_sha256=source_digest(Path(__file__).resolve().parents[1]),
            ),
            indent=2,
        ),
    )
    os.environ.setdefault("OMP_NUM_THREADS", str(s2.TORCH_THREADS))
    jobs = [(did, meth, ms) for meth, ms in pairs for did in dataset_ids]
    done = []
    with ProcessPoolExecutor(max_workers=workers, initializer=s2._init_worker) as pool:
        futures = {
            pool.submit(
                run_job,
                did,
                meth,
                cache_dir,
                out_dir,
                base_seed,
                None,
                s2.STAGE_A_REPS,
                ms,
            ): (did, meth)
            for did, meth, ms in jobs
        }
        for future in as_completed(futures):
            did, meth = futures[future]
            try:
                done.append(future.result())
            except Exception as exc:  # noqa: BLE001 - logged; other jobs go on
                s2._record_failure(out_dir, did, meth, exc)
                print(f"FAILED {(did, meth)}: {exc!r}", flush=True)
                continue
            print(f"done {(did, meth)} ({len(done)}/{len(jobs)})", flush=True)
    return done
