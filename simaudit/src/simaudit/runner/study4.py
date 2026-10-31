"""Study 4, Stage A: cold-start label selection x budgets, scored with FM5.

One job = one dataset. For each budget B = mult * C and each strategy, the
selected rows are labeled, FM5 (TabPFN v3.5) predicts every row, and LLGC on the
FM5 prior refines it (RQ-L3). Scores are on the complement of the selected rows;
labeled indices and predictions are stored so that the sensitivity analysis and
the split-half oracle can be computed afterwards (prereg/STUDY4_LABEL_SELECTION.md).
"""

import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from simaudit.data.openml_loader import load
from simaudit.data.splits import draw_split
from simaudit.graph.knn import mutual_knn_graph, zscore
from simaudit.graph.propagation import llgc
from simaudit.runner.pilot import _provenance_name, source_digest
from simaudit.selection import baselines
from simaudit.selection import strategies as sel
from simaudit.selection.cache import disk_cached
from simaudit.selection.lucos import select_lucos

SEED_NAMESPACE = 5
BUDGET_MULTS = (1, 3, 5, 10)
REPS = {"random": 10}
DEFAULT_REPS = 5
LLGC_ALPHA, LLGC_K = 0.9, 10
TORCH_THREADS = 1
# Strategy order fixes the seed of each strategy; append only (Amendment 1).
ORDER = (
    "stratified",
    "random",
    "kmedoids",
    "typiclust",
    "probcover",
    "kcenter",
    "outlier_filtered",
    "sigma_opt",
    "stratified_medoids",
    "kmeans_central",
    "lucos",  # Amendment 2
    "usl",  # Amendment 3
    "rdss",  # Amendment 3
    "zcore",  # Amendment 3
    "kmedoids_raw",  # Amendment 4
)
# The strategies of the original Stage A / Stage B runs (Amendment 1).
STAGE_A = ORDER[: ORDER.index("lucos")]
# Label-free strategies registered here rather than in selection.strategies.
# Amendment 3 selections are disk-cached, so every learner reuses the same rows.
EXTRA: dict[str, sel.Strategy] = {
    "lucos": select_lucos,
    **{name: disk_cached(name, fn) for name, fn in baselines.BASELINES.items()},
}
Predictor = Callable[..., NDArray[np.float64]]


def selection_seed(base_seed: int, did: int, mult: int, rep: int, name: str) -> int:
    """Seed of one selection, in namespace 5 (disjoint from pilot and search)."""
    state = np.random.SeedSequence(
        [base_seed, did, mult, rep, ORDER.index(name), SEED_NAMESPACE]
    )
    return int(state.generate_state(1)[0] % (2**31 - 1))


def select(
    name: str,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    mult: int,
    seed: int,
) -> NDArray[np.int64]:
    """Rows to label; only the two reference samplers ever look at y."""
    n_classes = int(y.max()) + 1
    if name == "stratified":
        return np.sort(draw_split(y, mult, seed)[0]).astype(np.int64)
    if name == "stratified_medoids":
        return sel.select_stratified_medoids(X, y, mult, seed)
    budget = mult * n_classes
    if name in sel.CLASS_AWARE:
        return sel.CLASS_AWARE[name](X, budget, seed, n_classes=n_classes)
    return {**sel.STRATEGIES, **EXTRA}[name](X, budget, seed)


def _scores(
    y: NDArray[np.int64], proba: NDArray[np.float64], rest: NDArray[np.int64]
) -> dict[str, float]:
    from sklearn.metrics import f1_score

    n_classes = proba.shape[1]
    pred = proba[rest].argmax(axis=1)
    return dict(
        macro_f1=float(
            f1_score(
                y[rest],
                pred,
                labels=list(range(n_classes)),
                average="macro",
                zero_division=0,
            )
        ),
        accuracy=float(np.mean(pred == y[rest])),
    )


def _llgc_refine(
    X: NDArray[np.float64],
    proba: NDArray[np.float64],
    labeled: NDArray[np.int64],
    y_lab: NDArray[np.int64],
) -> NDArray[np.float64]:
    prior = proba.copy()
    prior[labeled] = np.eye(proba.shape[1])[y_lab]
    graph = mutual_knn_graph(zscore(X), LLGC_K)
    return np.asarray(llgc(graph, prior, LLGC_ALPHA), dtype=np.float64)


def _predictor(learner: str) -> Predictor:
    """FM5 as in Stage A; FM1/FM2/FM3 (Stage B) via the harness wrappers."""
    from simaudit.autoresearch import harness
    from simaudit.runner.study2 import _fm_base

    stage_b = dict(fm1=harness.fm_tabpfn_v2, fm2=harness.fm_tabicl, fm3=harness.fm_kumo)
    return _fm_base("v35") if learner == "fm5" else stage_b[learner]


def output_name(did: int, learner: str, strategies: tuple[str, ...]) -> str:
    """``{did}[_{learner}][_{added}].parquet``; ``added`` = strategies after Stage A.

    Runs of the original strategies keep their file names, and later strategies
    (LUCoS) go to separate files that never touch the existing outputs.
    """
    added = [name for name in strategies if name not in STAGE_A]
    parts = [str(did)] + ([] if learner == "fm5" else [learner]) + added
    return "_".join(parts) + ".parquet"


def run_job(
    did: int,
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    predictor: Predictor | None = None,
    strategies: tuple[str, ...] = STAGE_A,
    mults: tuple[int, ...] = BUDGET_MULTS,
    learner: str = "fm5",
) -> Path:
    """Every strategy x budget x repetition for one dataset (skipped if done)."""
    out = out_dir / output_name(did, learner, strategies)
    if out.exists():
        return out
    predict = predictor or _predictor(learner)
    data = load(did, cache_dir)
    n_classes = int(data.y.max()) + 1
    rows = []
    for mult in mults:
        for name in strategies:
            for rep in range(REPS.get(name, DEFAULT_REPS)):
                rows.extend(
                    _one(
                        did,
                        data.X,
                        data.y,
                        n_classes,
                        mult,
                        name,
                        rep,
                        base_seed,
                        predict,
                        learner,
                    )
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(out)
    return out


def rerun_cells(
    did: int,
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    learner: str,
    strategy: str,
    mult: int,
    reps: tuple[int, ...],
    predictor: Predictor | None = None,
) -> Path:
    """Recompute some repetitions of one cell in place (Amendment 4 bug fix).

    The file that holds the cell is copied to ``superseded/`` with a timestamp,
    then its rows for (strategy, mult, reps) are replaced; nothing else changes.
    """
    path = out_dir / output_name(did, learner, (strategy,))
    if strategy in STAGE_A:
        path = out_dir / output_name(did, learner, STAGE_A)
    frame = pd.read_parquet(path)
    backup = (
        out_dir / "superseded" / f"{path.stem}.{time.strftime('%Y%m%dT%H%M%S')}.parquet"
    )
    backup.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(backup, index=False)
    data = load(did, cache_dir)
    predict = predictor or _predictor(learner)
    n_classes = int(data.y.max()) + 1
    new = [
        row
        for rep in reps
        for row in _one(
            did,
            data.X,
            data.y,
            n_classes,
            mult,
            strategy,
            rep,
            base_seed,
            predict,
            learner,
        )
    ]
    stale = (
        (frame.strategy == strategy) & (frame["mult"] == mult) & frame.rep.isin(reps)
    )
    merged = pd.concat([frame[~stale], pd.DataFrame(new)], ignore_index=True)
    tmp = path.with_suffix(".tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def _one(
    did: int,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    n_classes: int,
    mult: int,
    name: str,
    rep: int,
    base_seed: int,
    predict: Predictor,
    learner: str = "fm5",
) -> list[dict[str, Any]]:
    seed = selection_seed(base_seed, did, mult, rep, name)
    start = time.perf_counter()
    labeled = select(name, X, y, mult, seed)
    rest = np.setdiff1d(np.arange(len(y)), labeled)
    fm = sel.predict_with_budget(predict, X, labeled, y[labeled], n_classes, seed)
    refined = _llgc_refine(X, fm, labeled, y[labeled])
    common = dict(
        dataset=did,
        mult=mult,
        budget=len(labeled),
        strategy=name,
        rep=rep,
        seed=seed,
        coverage=sel.class_coverage(y, labeled),
        labeled=labeled.tolist(),
        seconds=time.perf_counter() - start,
    )
    return [
        dict(
            common,
            learner=name_,
            **_scores(y, proba, rest),
            pred=proba.argmax(axis=1).astype(np.int64).tolist(),
        )
        for name_, proba in ((learner, fm), (f"llgc_{learner}", refined))
    ]


def _init_worker() -> None:
    import torch

    torch.set_num_threads(TORCH_THREADS)


def run_stage_a(
    dataset_ids: list[int],
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    workers: int,
    learner: str = "fm5",
    strategies: tuple[str, ...] = STAGE_A,
) -> list[Path]:
    """Run every dataset in parallel; failures are logged and the rest go on."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _provenance_name()).write_text(
        json.dumps(
            dict(
                base_seed=base_seed,
                seed_namespace=SEED_NAMESPACE,
                mults=BUDGET_MULTS,
                strategies=list(strategies),
                order=ORDER,
                reps=dict(REPS, default=DEFAULT_REPS),
                llgc=dict(alpha=LLGC_ALPHA, k=LLGC_K),
                datasets=dataset_ids,
                learner=learner,
                source_sha256=source_digest(Path(__file__).resolve().parents[1]),
            ),
            indent=2,
        ),
    )
    os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_THREADS))
    done = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {
            pool.submit(
                run_job,
                did,
                cache_dir,
                out_dir,
                base_seed,
                None,
                strategies,
                BUDGET_MULTS,
                learner,
            ): did
            for did in dataset_ids
        }
        for future in as_completed(futures):
            did = futures[future]
            try:
                done.append(future.result())
            except Exception as exc:  # noqa: BLE001 - logged; other jobs go on
                with (out_dir / "failures.jsonl").open("a") as handle:
                    handle.write(json.dumps(dict(dataset=did, error=repr(exc))) + "\n")
                print(f"FAILED {did}: {exc!r}", flush=True)
                continue
            print(f"done {did} ({len(done)}/{len(dataset_ids)})", flush=True)
    return done
