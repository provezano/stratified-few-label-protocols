"""Resumable, parallel execution of pilot I2 (one result file per dataset and m)."""

import hashlib
import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simaudit.data.openml_loader import load
from simaudit.data.splits import draw_split
from simaudit.graph.knn import mutual_knn_graph, zscore
from simaudit.runner.cell import CellConfig, run_cell, score_cell

M_VALUES = (1, 3, 5, 10)
N_REPS = 10
TORCH_THREADS = 2


def cell_seed(base_seed: int, did: int, m: int, rep: int) -> int:
    """Return the seed for one (dataset, m, repetition), independent of run order."""
    state = np.random.SeedSequence([base_seed, did, m, rep]).generate_state(1)
    return int(state[0] % (2**31 - 1))


def _provenance_name() -> str:
    """One provenance file per invocation, so re-runs never overwrite history."""
    return f"provenance_{datetime.now().strftime('%Y%m%dT%H%M%S%f')}.json"


def source_digest(root: Path) -> str:
    """SHA-256 over the package sources, recorded as provenance."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _init_worker() -> None:
    import torch

    torch.set_num_threads(TORCH_THREADS)


def run_dataset_m(
    did: int,
    m: int,
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    config: CellConfig,
) -> Path:
    """Run every repetition for one (dataset, m) and write one parquet file."""
    out = out_dir / f"{did}_m{m}.parquet"
    if out.exists():
        return out
    data = load(did, cache_dir)
    W_feat = mutual_knn_graph(zscore(data.X), config.k)
    n_classes = int(data.y.max()) + 1
    rows = []
    for rep in range(N_REPS):
        seed = cell_seed(base_seed, did, m, rep)
        lab, unl = draw_split(data.y, m, seed)
        start = time.perf_counter()
        result = run_cell(data.X, data.y, lab, unl, seed, W_feat, config)
        seconds = time.perf_counter() - start
        scores = score_cell(result, data.y[unl], n_classes)
        for arm, metrics in scores.items():
            rows.append(
                dict(
                    dataset=did,
                    m=m,
                    rep=rep,
                    seed=seed,
                    arm=arm,
                    **metrics,
                    fallback=result.fallbacks.get(arm, 0),
                    **result.diagnostics,
                    seconds=seconds,
                ),
            )
    tmp = out.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(out)
    return out


def run_pilot(
    dataset_ids: list[int],
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    workers: int,
    config: CellConfig | None = None,
    job: Callable[..., Path] | None = None,
    job_args: tuple[Any, ...] = (),
) -> list[Path]:
    """Run all (dataset, m) jobs in parallel, skipping those already written.

    ``job`` defaults to :func:`run_dataset_m`; it is called as
    ``job(did, m, cache_dir, out_dir, base_seed, config, *job_args)``.
    """
    job = job or run_dataset_m
    config = config or CellConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _provenance_name()).write_text(
        json.dumps(
            dict(
                config=asdict(config),
                base_seed=base_seed,
                m_values=M_VALUES,
                n_reps=N_REPS,
                torch_threads=TORCH_THREADS,
                source_sha256=source_digest(Path(__file__).resolve().parents[1]),
                datasets=dataset_ids,
                job=job.__name__,
                job_args=list(job_args),
            ),
            indent=2,
        ),
    )
    jobs = [(did, m) for did in dataset_ids for m in M_VALUES]
    os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_THREADS))
    done = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {
            pool.submit(
                job, did, m, cache_dir, out_dir, base_seed, config, *job_args
            ): (did, m)
            for did, m in jobs
        }
        for future in as_completed(futures):
            done.append(future.result())
            print(f"done {futures[future]} ({len(done)}/{len(jobs)})", flush=True)
    return done
