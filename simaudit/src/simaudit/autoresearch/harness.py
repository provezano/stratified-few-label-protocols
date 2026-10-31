"""Fixed evaluation harness for Study 3 (the analogue of autoresearch's prepare.py).

A candidate is a Python file defining ``fit_predict(task: Task) -> proba`` with
``proba`` of shape (n, n_classes) over *all* rows. The harness owns the data,
the splits and the metric; the candidate sees features, the labeled indices,
their labels and cached FM outputs, never the unlabeled labels
(prereg/STUDY3_AUTORESEARCH.md).
"""

import importlib.util
import json
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score

from simaudit.data.openml_loader import load
from simaudit.data.splits import draw_split

SEARCH_M = (1, 3, 5, 10)
SEARCH_REPS = 3
SEARCH_NAMESPACE = 3  # disjoint from pilot cell seeds (tested)
NULL_NAMESPACE = 4  # label permutation of the null control loop
FM_NAMES = ("tabpfn_v2", "tabicl", "kumo", "tabpfn_v35")


@dataclass(frozen=True)
class Task:
    """Everything a candidate may use for one search cell."""

    X: NDArray[np.float64]
    labeled: NDArray[np.int64]
    y_labeled: NDArray[np.int64]
    n_classes: int
    seed: int
    fm_proba: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    embeddings: NDArray[np.float64] | None = None


def search_seed(base_seed: int, did: int, m: int, rep: int) -> int:
    """Seed of one search cell, in a namespace disjoint from the pilot's."""
    state = np.random.SeedSequence([base_seed, did, m, rep, SEARCH_NAMESPACE])
    return int(state.generate_state(1)[0] % (2**31 - 1))


def cell_path(cache_dir: Path, did: int, m: int, rep: int) -> Path:
    """Location of the cached FM outputs for one search cell."""
    return cache_dir / f"{did}_m{m}_r{rep}.npz"


def build_cell(
    did: int,
    m: int,
    rep: int,
    data_cache: Path,
    search_cache: Path,
    base_seed: int,
    fm_fns: dict[str, Callable[..., NDArray[np.float64]]],
    embed_fn: Callable[..., NDArray[np.float64]] | None,
) -> Path:
    """Cache the split, FM probabilities and embeddings of one search cell."""
    path = cell_path(search_cache, did, m, rep)
    if path.exists():
        return path
    data = load(did, data_cache)
    seed = search_seed(base_seed, did, m, rep)
    lab, unl = draw_split(data.y, m, seed)
    arrays = {"labeled": lab, "seed": np.array(seed)}
    for name, fn in fm_fns.items():
        arrays[f"fm_{name}"] = fn(data.X[lab], data.y[lab], data.X, seed)
    if embed_fn is not None:
        arrays["embeddings"] = embed_fn(data.X[lab], data.y[lab], data.X, seed)
    search_cache.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)  # type: ignore[arg-type]
    tmp.replace(path)
    return path


def load_task(
    did: int,
    m: int,
    rep: int,
    data_cache: Path,
    search_cache: Path,
    null: bool = False,
) -> tuple[Task, NDArray[np.int64]]:
    """Return the candidate-visible task and, separately, the hidden labels.

    With ``null``, the hidden labels are permuted among the unlabeled rows, so
    the candidate's inputs are unchanged but the scoring target is noise.
    """
    data = load(did, data_cache)
    z = np.load(cell_path(search_cache, did, m, rep), allow_pickle=False)
    lab = z["labeled"]
    fm_proba = {k[3:]: z[k] for k in z.files if k.startswith("fm_")}
    task = Task(
        X=data.X.copy(),
        labeled=lab.copy(),
        y_labeled=data.y[lab].copy(),
        n_classes=int(data.y.max()) + 1,
        seed=int(z["seed"]),
        fm_proba=fm_proba,
        embeddings=z["embeddings"] if "embeddings" in z.files else None,
    )
    return task, _null_labels(data.y, lab, task.seed) if null else data.y


def _null_labels(
    y: NDArray[np.int64], labeled: NDArray[np.int64], seed: int
) -> NDArray[np.int64]:
    out = y.copy()
    hidden = np.setdiff1d(np.arange(len(y)), labeled)
    out[hidden] = y[np.random.default_rng([seed, NULL_NAMESPACE]).permutation(hidden)]
    return out


def load_candidate(path: Path) -> Callable[[Task], NDArray[np.float64]]:
    """Import ``fit_predict`` from a candidate file."""
    spec = importlib.util.spec_from_file_location(f"candidate_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn: Callable[[Task], NDArray[np.float64]] = module.fit_predict
    return fn


FORBIDDEN = (
    "open(",
    "os.",
    "sys.",
    "pathlib",
    "inspect",
    "importlib",
    "subprocess",
    "socket",
    "gc.",
    "__import__",
    "eval(",
    "exec(",
    "globals(",
    "np.load",
    "pickle",
    "openml",
    "simaudit.data",
    "simaudit.settings",
    "builtins",
    "_getframe",
)


def check_source(path: Path) -> str:
    """Return the first forbidden token in a candidate's source, or ''.

    Candidates may not do I/O or introspection: they must not reach the hidden
    labels on disk or in the harness's memory.
    """
    source = path.read_text()
    return next((token for token in FORBIDDEN if token in source), "")


def run_candidate(candidate: Path, task: Task) -> NDArray[np.float64]:
    """Run a candidate in a worker process that only ever receives the Task."""
    return np.asarray(load_candidate(candidate)(task), dtype=np.float64)


SECONDARY = ("accuracy", "chance_f1", "log_loss")
FAILED_ROW = dict(macro_f1=0.0, accuracy=0.0, chance_f1=0.0, log_loss=float("nan"))


def _chance_f1(y: NDArray[np.int64], pred: NDArray[np.int64], n_classes: int) -> float:
    """Expected macro-F1 of a label-blind predictor with the same predicted shares.

    For class k with prevalence pi and predicted share q, that F1 is 2 pi q/(pi + q).
    """
    pi = np.bincount(y, minlength=n_classes) / len(y)
    q = np.bincount(pred, minlength=n_classes) / len(pred)
    total = pi + q
    per_class = np.divide(2 * pi * q, total, out=np.zeros(n_classes), where=total > 0)
    return float(per_class.mean())


def _metrics(
    proba: NDArray[np.float64] | None,
    task: Task,
    y: NDArray[np.int64],
) -> dict[str, float]:
    """Macro-F1, accuracy, chance macro-F1 and log-loss on the unlabeled rows."""
    unl = np.setdiff1d(np.arange(len(y)), task.labeled)
    if proba is None or proba.shape != (len(y), task.n_classes):
        raise ValueError(f"bad output shape {None if proba is None else proba.shape}")
    if not np.isfinite(proba).all():
        raise ValueError("non-finite probabilities")
    pred = proba[unl].argmax(axis=1)
    clipped = np.clip(proba[unl], 1e-15, None)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return dict(
        macro_f1=float(
            f1_score(
                y[unl],
                pred,
                labels=list(range(task.n_classes)),
                average="macro",
                zero_division=0,
            )
        ),
        accuracy=float((pred == y[unl]).mean()),
        chance_f1=_chance_f1(y[unl], pred, task.n_classes),
        log_loss=float(-np.log(clipped[np.arange(len(unl)), y[unl]]).mean()),
    )


def score_cell(
    candidate: Path,
    did: int,
    m: int,
    rep: int,
    data_cache: Path,
    search_cache: Path,
) -> dict[str, Any]:
    """Macro-F1 of a candidate on the unlabeled rows of one cell (0 on failure)."""
    task, y = load_task(did, m, rep, data_cache, search_cache)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_candidate, candidate, task)
        return _row(future, task, y, did, m, rep, start)


def _row(
    future: Any,
    task: Task,
    y: NDArray[np.int64],
    did: int,
    m: int,
    rep: int,
    start: float,
) -> dict[str, Any]:
    try:
        scores, error = _metrics(future.result(), task, y), ""
    except Exception:  # noqa: BLE001 - any candidate failure scores 0 and is logged
        scores, error = dict(FAILED_ROW), traceback.format_exc(limit=3)
    return dict(
        dataset=did,
        m=m,
        rep=rep,
        **scores,
        seconds=time.perf_counter() - start,
        error=error,
    )


def _run_cells(
    candidate: Path,
    cells: list[tuple[int, int, int]],
    data_cache: Path,
    search_cache: Path,
    workers: int,
    start: float,
    null: bool = False,
) -> list[dict[str, Any]]:
    violation = check_source(candidate)
    if violation:
        error = f"forbidden token {violation!r}"
        return [
            dict(dataset=d, m=m, rep=r, **FAILED_ROW, seconds=0.0, error=error)
            for d, m, r in cells
        ]
    loaded = {c: load_task(*c, data_cache, search_cache, null) for c in cells}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            c: pool.submit(run_candidate, candidate, loaded[c][0]) for c in cells
        }
        return [_row(futures[c], loaded[c][0], loaded[c][1], *c, start) for c in cells]


def score_by_m(
    rows: list[dict[str, Any]],
    metric: str = "macro_f1",
) -> dict[int, float]:
    """Mean over datasets of the per-dataset mean of a metric, for each m."""
    per_dataset_m: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        key = (int(row["dataset"]), int(row["m"]))
        per_dataset_m.setdefault(key, []).append(float(row[metric]))
    return {
        m: float(
            np.mean([np.mean(v) for (_, mm), v in per_dataset_m.items() if mm == m])
        )
        for m in sorted({m for _, m in per_dataset_m})
    }


def by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (dataset, m) means of macro-F1 and accuracy over repetitions."""
    frame: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        frame.setdefault((int(row["dataset"]), int(row["m"])), []).append(row)
    return [
        dict(
            dataset=d,
            m=m,
            macro_f1=float(np.mean([r["macro_f1"] for r in group])),
            accuracy=float(np.mean([r["accuracy"] for r in group])),
        )
        for (d, m), group in sorted(frame.items())
    ]


def _log(entry: dict[str, Any], candidate: Path, ledger: Path) -> None:
    archive = ledger.parent / ledger.stem.replace("ledger", "candidates")
    archive.mkdir(parents=True, exist_ok=True)
    n_logged = len(ledger.read_text().splitlines()) if ledger.exists() else 0
    copy = archive / f"{n_logged:04d}_{entry['sha256'][:8]}.py"
    copy.write_text(candidate.read_text())
    entry["archived"] = str(copy)
    with ledger.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")


def evaluate(
    candidate: Path,
    dataset_ids: list[int],
    data_cache: Path,
    search_cache: Path,
    ledger: Path,
    workers: int,
    note: str = "",
    null: bool = False,
) -> dict[str, Any]:
    """Score a candidate on every search cell and append one ledger line.

    Hidden labels stay in this process; workers receive only Tasks.
    """
    start = time.perf_counter()
    cells = [
        (d, m, r) for d in dataset_ids for m in SEARCH_M for r in range(SEARCH_REPS)
    ]
    rows = _run_cells(candidate, cells, data_cache, search_cache, workers, start, null)
    by_m = score_by_m(rows)
    entry: dict[str, Any] = dict(
        candidate=str(candidate),
        sha256=_sha256(candidate),
        score=float(np.mean(list(by_m.values()))),
        by_m=by_m,
        secondary={k: score_by_m(rows, k) for k in SECONDARY},
        by_dataset=by_dataset(rows),
        n_cells=len(rows),
        n_failed=sum(bool(r["error"]) for r in rows),
        first_error=next((r["error"] for r in rows if r["error"]), ""),
        seconds=time.perf_counter() - start,
        note=note,
        null=null,
    )
    _log(entry, candidate, ledger)
    return entry


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def fm_tabpfn_v2(X_fit: Any, y_fit: Any, X: Any, seed: int) -> NDArray[np.float64]:
    """Return TabPFN v2 probabilities for every row (labeled-only context)."""
    from simaudit.arms import fm

    return fm.fit_predict_proba(fm.make_tabpfn(seed), X_fit, y_fit, X)


def fm_tabicl(X_fit: Any, y_fit: Any, X: Any, seed: int) -> NDArray[np.float64]:
    """Return TabICL probabilities for every row."""
    from simaudit.arms import fm

    return fm.fit_predict_proba(fm.make_tabicl(seed), X_fit, y_fit, X)


def fm_kumo(X_fit: Any, y_fit: Any, X: Any, seed: int) -> NDArray[np.float64]:
    """Return Kumo-Tabular probabilities for every row."""
    from simaudit.arms import fm

    return fm.kumo_predict_proba(X_fit, y_fit, X, seed)


def fm_tabpfn_v35(X_fit: Any, y_fit: Any, X: Any, seed: int) -> NDArray[np.float64]:
    """Return TabPFN v3.5 probabilities for every row (research-only weights)."""
    from simaudit.arms import fm

    return fm.fit_predict_proba(fm.make_tabpfn_v35(seed), X_fit, y_fit, X)


REAL_FMS: dict[str, Callable[..., NDArray[np.float64]]] = dict(
    tabpfn_v2=fm_tabpfn_v2, tabicl=fm_tabicl, kumo=fm_kumo, tabpfn_v35=fm_tabpfn_v35
)


def real_embed_fn(X_fit: Any, y_fit: Any, X: Any, seed: int) -> NDArray[np.float64]:
    """Embed every row with TabPFN v2, using the labeled rows as context."""
    from simaudit.arms import fm

    model = fm.make_tabpfn(seed)
    model.fit(X_fit, y_fit)
    return fm.tabpfn_embeddings(model, X)


def build_search_cache(
    dataset_ids: list[int],
    data_cache: Path,
    search_cache: Path,
    base_seed: int,
    workers: int,
    fm_fns: dict[str, Callable[..., NDArray[np.float64]]] | None = None,
    embed_fn: Callable[..., NDArray[np.float64]] | None = real_embed_fn,
) -> int:
    """Cache every search cell's split, FM outputs and embeddings (resumable).

    With ``workers <= 1`` the cells are built in this process.
    """
    fns = REAL_FMS if fm_fns is None else fm_fns
    cells = [
        (d, m, r) for d in dataset_ids for m in SEARCH_M for r in range(SEARCH_REPS)
    ]
    args = [
        (d, m, r, data_cache, search_cache, base_seed, fns, embed_fn)
        for d, m, r in cells
    ]
    if workers <= 1:
        for a in args:
            build_cell(*a)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(build_cell, *a) for a in args]:
                future.result()
    return len(cells)
