"""Study 12: Study 7 Part (a) replicated on T1' (prereg/STUDY12_PROTOCOL_T1PRIME.md).

Per dataset, rows are permuted by a fixed seeded permutation (namespace 33).
Labeled sets are drawn from all rows (stratified 5 reps, random 20 reps,
m in {1, 3, 5, 10}); the evaluation panel E is the first min(n, 2000) permuted
rows and every metric is computed on E minus the labeled rows. The redraw arm
reuses each covered random draw (same rows, seed and fit) and redraws only the
draws that miss a class. One job = one (dataset, learner); its parquet is
written atomically, checkpointed and skipped when present.
"""

import json
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from simaudit.analysis.study7 import prior_correct
from simaudit.analysis.study12 import panel_metrics
from simaudit.data.openml_loader import Dataset, load
from simaudit.runner import study4
from simaudit.runner.pilot import _provenance_name, source_digest
from simaudit.runner.study9 import dataset_key, predict_on, seed_did

SEED_NAMESPACE = 32
PERM_NAMESPACE = 33
MULTS = (1, 3, 5, 10)
STRATEGIES = ("stratified", "random")
REPS = {"stratified": 5, "random": 20}
PANEL_MAX = 2000
MAX_ATTEMPTS = 100_000
LEARNERS = ("fm2", "fm5", "fm1", "fm3")  # Sec. 5 priority order
POSTHOC = ("fm1", "fm5")  # fm1_bal / fm5_bal = p / pi-hat of these (Sec. 2.2)
CHECKPOINT_EVERY = 25
TORCH_THREADS = 1

Predictor = Callable[..., NDArray[np.float64]]
Loader = Callable[[int, Path], Dataset]
IntArray = NDArray[np.int64]


# ---------------------------------------------------------------------- seeds
def _seed(*entropy: int) -> int:
    state = np.random.SeedSequence(list(entropy))
    return int(state.generate_state(1)[0] % (2**31 - 1))


def selection_seed(base_seed: int, did: int, mult: int, rep: int, name: str) -> int:
    """Seed of one selection and its FM fit (namespace 32)."""
    return _seed(
        base_seed, seed_did(did), mult, rep, study4.ORDER.index(name), SEED_NAMESPACE
    )


def redraw_seed(base_seed: int, did: int, mult: int, rep: int, attempt: int) -> int:
    """Seed of redraw attempt ``attempt`` >= 1 of a random rep (Sec. 2.3)."""
    return _seed(
        base_seed,
        seed_did(did),
        mult,
        rep,
        attempt,
        study4.ORDER.index("random"),
        SEED_NAMESPACE,
        1,
    )


def row_permutation(n: int, did: int, base_seed: int) -> IntArray:
    """Fixed per-dataset row permutation (namespace 33)."""
    rng = np.random.default_rng([base_seed, seed_did(did), PERM_NAMESPACE])
    return rng.permutation(n).astype(np.int64)


def panel_size(n: int) -> int:
    """Panel size e = min(n, 2000); E is the first e permuted rows."""
    return min(n, PANEL_MAX)


# ---------------------------------------------------------------------- selection
def covers(y: IntArray, labeled: IntArray) -> bool:
    """Whether the labeled rows contain every class of y."""
    return len(np.unique(y[labeled])) == len(np.unique(y))


def redraw(
    y: IntArray, mult: int, rep: int, first: IntArray, did: int, base_seed: int
) -> tuple[IntArray, int, int]:
    """Labeled rows, attempt and seed of the redraw run of one random rep.

    Attempt 0 is the random draw itself; later attempts redraw uniformly until
    every class is present. If no attempt covers every class within
    MAX_ATTEMPTS, the rep is infeasible (prereg Sec. 2.3): empty rows, attempt -1.
    """
    if covers(y, first):
        return first, 0, selection_seed(base_seed, did, mult, rep, "random")
    budget = mult * (int(y.max()) + 1)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        seed = redraw_seed(base_seed, did, mult, rep, attempt)
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(len(y), size=budget, replace=False)).astype(np.int64)
        if covers(y, rows):
            return rows, attempt, seed
    return np.zeros(0, dtype=np.int64), -1, -1


def selections(
    key: str,
    did: int,
    X: NDArray[np.float64],
    y: IntArray,
    base_seed: int,
    out_dir: Path,
) -> pd.DataFrame:
    """Stratified, random and redraw selections, in permuted positions; cached."""
    path = out_dir / "selections" / f"{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    for mult in MULTS:
        for name in STRATEGIES:
            for rep in range(REPS[name]):
                seed = selection_seed(base_seed, did, mult, rep, name)
                picked = np.sort(study4.select(name, X, y, mult, seed)).astype(np.int64)
                rows.append(
                    dict(
                        strategy=name,
                        mult=mult,
                        rep=rep,
                        attempt=0,
                        seed=seed,
                        labeled_pos=picked.tolist(),
                    )
                )
                if name == "random":
                    lab, attempt, rseed = redraw(y, mult, rep, picked, did, base_seed)
                    rows.append(
                        dict(
                            strategy="redraw",
                            mult=mult,
                            rep=rep,
                            attempt=attempt,
                            seed=rseed,
                            labeled_pos=lab.tolist(),
                        )
                    )
    frame = pd.DataFrame(rows)
    _atomic_write(frame, path)
    return frame


# ---------------------------------------------------------------------- fitting
def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def fit_record(
    predict: Predictor,
    X: NDArray[np.float64],
    y: IntArray,
    labeled: IntArray,
    seed: int,
    perm: IntArray,
) -> dict[str, Any]:
    """Fit on the labeled rows, predict E minus L, store probabilities + metrics.

    ``pc_*`` metrics apply the post-hoc prior correction p / pi-hat (TabPFN's
    balance option, Study 7 Part (b)) to the same probabilities.
    """
    n_classes = int(y.max()) + 1
    query = np.setdiff1d(np.arange(panel_size(len(y)), dtype=np.int64), labeled)
    start = time.perf_counter()
    proba = predict_on(predict, X, labeled, y, query, n_classes, seed)
    seconds = time.perf_counter() - start
    seen = np.unique(y[labeled]).astype(np.int64)
    corrected = panel_metrics(y[query], prior_correct(proba, y[labeled]), seen)
    return dict(
        n_classes=n_classes,
        budget=len(labeled),
        labeled=perm[labeled].tolist(),
        query=perm[query].tolist(),
        proba=proba.astype(np.float32).ravel(),
        coverage=len(seen) / n_classes,
        seen=seen.tolist(),
        seconds=seconds,
        **panel_metrics(y[query], proba, seen),
        **{f"pc_{k}": v for k, v in corrected.items()},
    )


def _cells(
    chosen: pd.DataFrame,
) -> Iterator[tuple[str, dict[str, Any]]]:
    for sel in cast(list[dict[str, Any]], chosen.to_dict("records")):
        yield f"{sel['strategy']}|{sel['mult']}|{sel['rep']}", sel


def output_path(out_dir: Path, key: str, learner: str) -> Path:
    """``results/study12/{key}_{learner}.parquet``."""
    return out_dir / f"{key}_{learner}.parquet"


def _cell_record(
    reuse: bool,
    sel: dict[str, Any],
    meta: dict[str, Any],
    random_fits: dict[tuple[int, int], dict[str, Any]],
    fn: Predictor,
    X: NDArray[np.float64],
    y: IntArray,
    perm: IntArray,
) -> dict[str, Any]:
    """Fit record of one cell; a reused redraw copies its random run's fit."""
    if reuse:  # identical rows, seed and fit as the random run of this rep
        base = random_fits[(meta["mult"], meta["rep"])]
        return {
            k: v
            for k, v in base.items()
            if k not in meta and k not in ("_key", "reused")
        }
    labeled = np.asarray(sel["labeled_pos"], dtype=np.int64)
    return fit_record(fn, X, y, labeled, meta["seed"], perm)


def run_job(
    did: int,
    learner: str,
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    *,
    loader: Loader = load,
    predict: Predictor | None = None,
) -> Path:
    """Every strategy x mult x rep (and redraw) for one dataset and learner."""
    key = dataset_key(did)
    out = output_path(out_dir, key, learner)
    if out.exists():
        return out
    data = loader(did, cache_dir)
    perm = row_permutation(len(data.y), did, base_seed)
    X, y = data.X[perm], data.y[perm]
    chosen = selections(key, did, X, y, base_seed, out_dir)
    fn = predict or study4._predictor(learner)
    partial = out.with_name(out.name.replace(".parquet", ".partial.parquet"))
    rows: list[dict[str, Any]] = []
    if partial.exists():
        rows = cast(list[dict[str, Any]], pd.read_parquet(partial).to_dict("records"))
    done = {r["_key"] for r in rows}
    random_fits: dict[tuple[int, int], dict[str, Any]] = {
        (int(r["mult"]), int(r["rep"])): r for r in rows if r["strategy"] == "random"
    }
    fresh = 0
    for cell, sel in _cells(chosen):
        if cell in done or int(sel["attempt"]) < 0:  # infeasible redraw: no fit
            continue
        meta = dict(
            strategy=sel["strategy"],
            mult=int(sel["mult"]),
            rep=int(sel["rep"]),
            attempt=int(sel["attempt"]),
            seed=int(sel["seed"]),
        )
        reuse = sel["strategy"] == "redraw" and sel["attempt"] == 0
        record = _cell_record(reuse, sel, meta, random_fits, fn, X, y, perm)
        row = dict(meta, reused=reuse, _key=cell, **record)
        rows.append(row)
        if sel["strategy"] == "random":
            random_fits[(meta["mult"], meta["rep"])] = row
        fresh += 1
        if fresh % CHECKPOINT_EVERY == 0:
            _atomic_write(pd.DataFrame(rows), partial)
    frame = pd.DataFrame(rows).drop(columns="_key")
    frame.insert(0, "learner", learner)
    frame.insert(0, "dataset", key)
    _atomic_write(frame, out)
    partial.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------------- driver
def _init_worker() -> None:  # pragma: no cover - runs in worker processes
    import torch

    torch.set_num_threads(TORCH_THREADS)


def _write_provenance(
    out_dir: Path, base_seed: int, ids: list[int], learners: list[str]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _provenance_name()).write_text(
        json.dumps(
            dict(
                prereg="prereg/STUDY12_PROTOCOL_T1PRIME.md",
                base_seed=base_seed,
                seed_namespace=SEED_NAMESPACE,
                perm_namespace=PERM_NAMESPACE,
                mults=MULTS,
                reps=REPS,
                panel_max=PANEL_MAX,
                learners=learners,
                datasets=ids,
                source_sha256=source_digest(Path(__file__).resolve().parents[1]),
            ),
            indent=2,
        )
    )


def _log_failure(out_dir: Path, job: tuple[Any, ...], exc: Exception) -> None:
    with (out_dir / "failures.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(job=list(job), error=repr(exc))) + "\n")
    print(f"FAILED {job}: {exc!r}", flush=True)


def run(
    dataset_ids: list[int],
    learners: list[str],
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    workers: int,
) -> list[Path]:
    """Run every (learner, dataset) job in priority order; failures retried once."""
    _write_provenance(out_dir, base_seed, dataset_ids, learners)
    os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_THREADS))
    jobs = [(did, lr) for lr in learners for did in dataset_ids]
    done: list[Path] = []
    for attempt in (1, 2):
        failed = []
        with ProcessPoolExecutor(  # pragma: no cover - multiprocess path
            max_workers=workers, initializer=_init_worker
        ) as pool:
            futures = {
                pool.submit(run_job, *job, cache_dir, out_dir, base_seed): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    done.append(future.result())
                except Exception as exc:  # noqa: BLE001 - logged; retried once
                    _log_failure(out_dir, (*job, attempt), exc)
                    failed.append(job)
                    continue
                print(
                    f"done {job} ({len(done)}/{len(dataset_ids) * len(learners)})",
                    flush=True,
                )
        jobs = failed
        if not jobs:
            break
    return done


def main() -> None:  # pragma: no cover - command-line entry
    """Command-line entry point.

    ``python -m simaudit.runner.study12 [--workers 7] [--limit N] [--learners fm2]``
    """
    import argparse

    import yaml

    from simaudit.settings import PROJECT_ROOT, settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--learners", default=",".join(LEARNERS))
    args = parser.parse_args()
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())
    ids = [int(d["did"]) for d in pool["datasets"]]
    ids = ids[: args.limit] if args.limit else ids
    run(
        ids,
        args.learners.split(","),
        settings.data_cache_dir,
        settings.results_dir / "study12",
        settings.base_seed,
        args.workers,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
