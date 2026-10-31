"""Fixed evaluation harness for Study 5 (Workstream B autoresearch: label selection).

A candidate is a Python file defining ``select(task: SelectTask) -> indices``:
which ``task.budget`` rows to label, chosen from the features alone. The harness
reveals the labels of the chosen rows, fits the loop learner (TabICL, FM2) on
them, and scores the complement (prereg/STUDY5_SELECTION_LOOP.md). Candidates
never see a label and never call a foundation model.
"""

import hashlib
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score

from simaudit.autoresearch import harness
from simaudit.data.openml_loader import load
from simaudit.selection.strategies import class_coverage, predict_with_budget

MULTS = (1, 3, 5, 10)
REPS = 3
SEED_NAMESPACE = 6  # disjoint from Study 4 selection seeds (namespace 5)
NULL_NAMESPACE = 7  # label permutation of the null control loop
RESCORE_NAMESPACE = 9  # fresh seeds for the final re-score (Study 6 Sec. 3)
RESCORE_REPS = 5
FORBIDDEN = (*harness.FORBIDDEN, "tabpfn", "tabicl", "kumo", "simaudit")
FAILED_ROW = dict(macro_f1=0.0, accuracy=0.0, chance_f1=0.0, coverage=0.0)
SECONDARY = ("accuracy", "coverage", "chance_f1")
Predictor = Callable[..., NDArray[np.float64]]


@dataclass(frozen=True)
class SelectTask:
    """Everything a candidate may use: features, budget, class count and seed."""

    X: NDArray[np.float64]
    budget: int
    n_classes: int
    seed: int


def cell_seed(
    base_seed: int, did: int, mult: int, rep: int, namespace: int = SEED_NAMESPACE
) -> int:
    """Seed of one (dataset, budget, repetition) cell, in namespace 6 by default."""
    state = np.random.SeedSequence([base_seed, did, mult, rep, namespace])
    return int(state.generate_state(1)[0] % (2**31 - 1))


def null_labels(y: NDArray[np.int64], did: int) -> NDArray[np.int64]:
    """A fixed permutation of all labels: the features carry no information."""
    rng = np.random.default_rng([did, NULL_NAMESPACE])
    return np.asarray(y[rng.permutation(len(y))], dtype=np.int64)


def _learner() -> Predictor:
    return harness.fm_tabicl


def _load_select(path: Path) -> Callable[[SelectTask], Any]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"selector_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn: Callable[[SelectTask], Any] = module.select
    return fn


def run_selector(candidate: Path, task: SelectTask) -> NDArray[np.int64]:
    """Run a candidate in a worker process that only ever receives the task."""
    chosen = np.asarray(_load_select(candidate)(task))
    return validate(chosen, task, len(task.X))


def validate(chosen: Any, task: SelectTask, n: int) -> NDArray[np.int64]:
    """Return sorted indices, or raise if the selection breaks the budget rules."""
    flat = np.asarray(chosen).ravel()
    if not np.all(np.equal(np.mod(flat, 1), 0)):
        raise ValueError("selection: indices must be integers")
    idx = np.unique(flat.astype(np.int64))
    if len(idx) != task.budget or len(flat) != task.budget:
        raise ValueError(f"selection: need {task.budget} distinct rows, got {flat}")
    if idx[0] < 0 or idx[-1] >= n:
        raise ValueError("selection: index out of range")
    return idx


def check_source(path: Path) -> str:
    """Return the first forbidden token in a selector's source, or ''."""
    source = path.read_text()
    return next((token for token in FORBIDDEN if token in source), "")


def _cache_file(
    cache: Path, did: int, seed: int, idx: NDArray[np.int64], null: bool
) -> Path:
    key = f"{did}|{seed}|{int(null)}|{','.join(map(str, idx.tolist()))}"
    return cache / f"{did}_{hashlib.sha256(key.encode()).hexdigest()[:20]}.npz"


def fm_proba(
    did: int,
    data_cache: Path,
    fm_cache: Path,
    seed: int,
    idx: NDArray[np.int64],
    null: bool,
) -> NDArray[np.float64]:
    """Learner probabilities for every row given the chosen rows (disk-cached)."""
    path = _cache_file(fm_cache, did, seed, idx, null)
    if path.exists():
        return np.asarray(np.load(path)["proba"], dtype=np.float64)
    data = load(did, data_cache)
    y = null_labels(data.y, did) if null else data.y
    n_classes = int(data.y.max()) + 1
    proba = predict_with_budget(_learner(), data.X, idx, y[idx], n_classes, seed)
    fm_cache.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, proba=proba)
    tmp.replace(path)
    return proba


def _metrics(
    proba: NDArray[np.float64], idx: NDArray[np.int64], y: NDArray[np.int64]
) -> dict[str, float]:
    n_classes = proba.shape[1]
    rest = np.setdiff1d(np.arange(len(y)), idx)
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
        accuracy=float((pred == y[rest]).mean()),
        chance_f1=harness._chance_f1(y[rest], pred, n_classes),
        coverage=class_coverage(y, idx),
    )


Cell = tuple[int, int, int]


def _tasks(
    cells: list[Cell], data_cache: Path, base_seed: int, namespace: int
) -> dict[Cell, SelectTask]:
    out = {}
    for did, mult, rep in cells:
        data = load(did, data_cache)
        n_classes = int(data.y.max()) + 1
        out[(did, mult, rep)] = SelectTask(
            X=data.X.copy(),
            budget=mult * n_classes,
            n_classes=n_classes,
            seed=cell_seed(base_seed, did, mult, rep, namespace),
        )
    return out


def _selections(
    candidate: Path, tasks: dict[Cell, SelectTask], workers: int, trusted: bool = False
) -> dict[Cell, NDArray[np.int64] | str]:
    violation = "" if trusted else check_source(candidate)
    if violation:
        return {c: f"forbidden token {violation!r}" for c in tasks}
    out: dict[Cell, NDArray[np.int64] | str] = {}
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as pool:
        futures = {c: pool.submit(run_selector, candidate, t) for c, t in tasks.items()}
        for cell, future in futures.items():
            try:
                out[cell] = future.result()
            except Exception:  # noqa: BLE001 - any candidate failure scores 0
                out[cell] = traceback.format_exc(limit=3)
    return out


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def _probas(
    jobs: dict[Cell, tuple[int, NDArray[np.int64]]],
    data_cache: Path,
    fm_cache: Path,
    workers: int,
    null: bool,
) -> dict[Cell, NDArray[np.float64]]:
    args = {
        c: (c[0], data_cache, fm_cache, seed, idx, null)
        for c, (seed, idx) in jobs.items()
    }
    if workers <= 1:
        return {c: fm_proba(*a) for c, a in args.items()}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {c: pool.submit(fm_proba, *a) for c, a in args.items()}
        return {c: f.result() for c, f in futures.items()}


def _rows(
    cells: list[Cell],
    tasks: dict[Cell, SelectTask],
    chosen: dict[Cell, NDArray[np.int64] | str],
    probas: dict[Cell, NDArray[np.float64]],
    labels: dict[int, NDArray[np.int64]],
) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        did, mult, rep = cell
        pick = chosen[cell]
        if isinstance(pick, str):
            scores, error = dict(FAILED_ROW), pick
        else:
            scores, error = _metrics(probas[cell], pick, labels[did]), ""
        rows.append(
            dict(
                dataset=did,
                m=mult,
                rep=rep,
                budget=tasks[cell].budget,
                **scores,
                error=error,
            )
        )
    return rows


def evaluate(
    candidate: Path,
    dataset_ids: list[int],
    data_cache: Path,
    fm_cache: Path,
    ledger: Path,
    workers: int,
    note: str = "",
    null: bool = False,
    base_seed: int = 20261020,
    trusted: bool = False,
    rescore: bool = False,
) -> dict[str, Any]:
    """Score a selector on every (dataset, budget, rep) cell; append one ledger line.

    Labels stay in this process and in the learner's workers; the candidate's
    workers receive only ``SelectTask`` objects. ``trusted`` skips the
    forbidden-token check, for literature references that call project code
    (LUCoS needs TabClustPFN); such entries carry ``trusted: true``. Loop
    candidates are never trusted. ``rescore`` draws fresh seeds (namespace 9,
    5 reps) for the final re-score of branch bests.
    """
    start = time.perf_counter()
    reps, namespace = (
        (RESCORE_REPS, RESCORE_NAMESPACE) if rescore else (REPS, SEED_NAMESPACE)
    )
    cells = [(d, m, r) for d in dataset_ids for m in MULTS for r in range(reps)]
    tasks = _tasks(cells, data_cache, base_seed, namespace)
    chosen = _selections(candidate, tasks, workers, trusted)
    jobs = {
        c: (tasks[c].seed, pick)
        for c, pick in chosen.items()
        if not isinstance(pick, str)
    }
    probas = _probas(jobs, data_cache, fm_cache, workers, null)
    labels = {
        d: null_labels(load(d, data_cache).y, d) if null else load(d, data_cache).y
        for d in dataset_ids
    }
    rows = _rows(cells, tasks, chosen, probas, labels)
    by_m = harness.score_by_m(rows)
    entry: dict[str, Any] = dict(
        candidate=str(candidate),
        sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        score=float(np.mean(list(by_m.values()))),
        by_m=by_m,
        secondary={k: harness.score_by_m(rows, k) for k in SECONDARY},
        by_dataset=harness.by_dataset(rows),
        n_cells=len(rows),
        n_failed=sum(bool(r["error"]) for r in rows),
        first_error=next((r["error"] for r in rows if r["error"]), ""),
        seconds=time.perf_counter() - start,
        note=note,
        null=null,
    )
    if trusted:
        entry["trusted"] = True
    if rescore:
        entry["rescore"] = True
    harness._log(entry, candidate, ledger)
    return entry
