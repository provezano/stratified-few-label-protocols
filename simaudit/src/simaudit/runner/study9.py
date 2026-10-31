"""Study 9: how many labels does a tabular FM need? (prereg/STUDY9_HOW_MANY_LABELS.md).

Per dataset, rows are first permuted by a fixed seeded permutation (namespace
13) so that no index carries label information; every index below is in that
permuted order and is mapped back to the original order only when stored.
Rows [0, s) are the stop set S (never labeled), rows [s, n) the labeling pool
P, rows [0, e) the evaluation panel E. Each strategy yields nested labeled sets
(budget-m set = prefix of the budget-m' set); every FM fit predicts E minus the
labeled rows and stores the probabilities.

One job = one (set, dataset, learner); its parquet is written atomically and
skipped when present. Selections are cached once per dataset and shared.
"""

import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from simaudit.analysis.study7 import metric_set
from simaudit.data.openml_loader import Dataset, load
from simaudit.graph.knn import zscore
from simaudit.runner.pilot import _provenance_name, source_digest
from simaudit.selection.strategies import expand_proba

SEED_NAMESPACE = 12
PERM_NAMESPACE = 13
GRID = (1, 2, 3, 5, 7, 10, 15, 20, 30)
# Append-only: the position fixes each strategy's seeds.
STRATEGIES = ("random", "stratified", "kmedoids_nested", "lucos_nested")
REPS = {"random": 10, "stratified": 5, "kmedoids_nested": 3, "lucos_nested": 3}
DETERMINISTIC = ("kmedoids_nested", "lucos_nested")
STOP_MAX, PANEL_MAX = 500, 2000
CHECKPOINT_EVERY = 25
TORCH_THREADS = 1

Predictor = Callable[..., NDArray[np.float64]]
Embed = Callable[[NDArray[np.float64]], NDArray[Any]]
Loader = Callable[[int, Path], Dataset]
IntArray = NDArray[np.int64]


# ---------------------------------------------------------------------- seeds
def seed_did(did: int) -> int:
    """Non-negative seed entropy for a dataset ID (PMLB IDs are negative)."""
    return did if did >= 0 else 2**40 + did


def _seed(*entropy: int) -> int:
    state = np.random.SeedSequence([*entropy, SEED_NAMESPACE])
    return int(state.generate_state(1)[0] % (2**31 - 1))


def selection_seed(base_seed: int, did: int, rep: int, name: str) -> int:
    """Seed of one nested selection (one per rep, shared by every budget)."""
    return _seed(base_seed, seed_did(did), rep, STRATEGIES.index(name))


def fm_seed(base_seed: int, did: int, m: int, rep: int, name: str) -> int:
    """Seed of one FM fit."""
    return _seed(base_seed, seed_did(did), m, rep, STRATEGIES.index(name))


def row_permutation(n: int, did: int, base_seed: int) -> IntArray:
    """Fixed per-dataset row permutation (namespace 13) against row-order leaks."""
    rng = np.random.default_rng([base_seed, seed_did(did), PERM_NAMESPACE])
    return rng.permutation(n).astype(np.int64)


# ---------------------------------------------------------------------- layout
def layout(n: int) -> tuple[int, int]:
    """Stop-set size s and evaluation-panel size e (S = [0, s), E = [0, e))."""
    return min(STOP_MAX, n // 5), min(n, PANEL_MAX)


def feasible_ms(y: IntArray, pool: IntArray, name: str) -> list[int]:
    """Grid budgets with B = mC <= n/2 (and m < smallest pool class, stratified)."""
    n_classes = int(y.max()) + 1
    ms = [m for m in GRID if m * n_classes <= len(y) // 2]
    if name == "stratified":
        smallest = int(np.bincount(y[pool], minlength=n_classes).min())
        ms = [m for m in ms if m <= smallest - 1]
    return ms


# ---------------------------------------------------------------------- selection
def greedy_facility_location(Z: NDArray[Any], candidates: IntArray, k: int) -> IntArray:
    """PAM BUILD: k greedy picks from ``candidates`` covering all rows of Z.

    The first pick minimises the total distance; each next pick maximises the
    reduction of sum_i min_s d(i, s). Ties go to the lowest index.
    """
    Z32 = np.asarray(Z, dtype=np.float32)
    sq = (Z32**2).sum(axis=1)
    cz = Z32[candidates]
    D = np.sqrt(np.maximum(sq[:, None] + sq[candidates][None, :] - 2 * Z32 @ cz.T, 0))
    first = int(np.argmin(D.sum(axis=0)))
    picks = [first]
    best = D[:, first].copy()
    for _ in range(1, k):
        gain = np.maximum(best[:, None] - D, 0).sum(axis=0)
        gain[picks] = -1.0
        j = int(np.argmax(gain))
        picks.append(j)
        np.minimum(best, D[:, j], out=best)
    return candidates[np.array(picks, dtype=np.int64)]


def _max_budget(y: IntArray, pool: IntArray, name: str) -> tuple[list[int], int]:
    ms = feasible_ms(y, pool, name)
    return ms, max(ms) * (int(y.max()) + 1)


def nested_sets(
    name: str,
    X: NDArray[np.float64],
    y: IntArray,
    pool: IntArray,
    seed: int,
    embed: Embed | None,
) -> dict[int, IntArray]:
    """Labeled set per feasible m, nested within the repetition (permuted order)."""
    ms, top = _max_budget(y, pool, name)
    n_classes = int(y.max()) + 1
    if name == "stratified":
        rng = np.random.default_rng(seed)
        by_class = [rng.permutation(pool[y[pool] == c]) for c in range(n_classes)]
        return {m: np.sort(np.concatenate([b[:m] for b in by_class])) for m in ms}
    if name == "random":
        order = np.random.default_rng(seed).permutation(pool)
    elif name == "kmedoids_nested":
        order = greedy_facility_location(zscore(X), pool, top)
    elif name == "lucos_nested" and embed is not None:
        order = greedy_facility_location(np.asarray(embed(X)), pool, top)
    else:
        raise ValueError(f"unknown strategy {name!r}")
    return {m: np.sort(order[: m * n_classes]) for m in ms}


def _default_embed(X: NDArray[np.float64]) -> NDArray[Any]:  # pragma: no cover
    from simaudit.selection.lucos import embed

    return embed(X)


def selections(
    key: str,
    did: int,
    X: NDArray[np.float64],
    y: IntArray,
    base_seed: int,
    out_dir: Path,
    embed: Embed | None = _default_embed,
) -> pd.DataFrame:
    """All nested selections of one (permuted) dataset, computed once and cached."""
    path = out_dir / "selections" / f"{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    s, _ = layout(len(y))
    pool = np.arange(s, len(y), dtype=np.int64)
    rows = []
    for name in STRATEGIES:
        sets: dict[int, IntArray] | None = None
        for rep in range(REPS[name]):
            seed = selection_seed(base_seed, did, rep, name)
            if sets is None or name not in DETERMINISTIC:
                sets = nested_sets(name, X, y, pool, seed, embed)
            rows += [
                dict(strategy=name, rep=rep, m=m, sel_seed=seed, labeled_pos=L.tolist())
                for m, L in sets.items()
            ]
    frame = pd.DataFrame(rows)
    _atomic_write(frame, path)
    return frame


# ---------------------------------------------------------------------- fitting
def predict_on(
    predict: Predictor,
    X: NDArray[np.float64],
    labeled: IntArray,
    y: IntArray,
    query: IntArray,
    n_classes: int,
    seed: int,
) -> NDArray[np.float64]:
    """Fit on the labeled rows (seen classes only) and predict ``query`` rows."""
    seen, y_fit = np.unique(y[labeled], return_inverse=True)
    if len(seen) == 1:
        return expand_proba(np.ones((len(query), 1)), seen, n_classes)
    proba = predict(X[labeled], y_fit.astype(np.int64), X[query], seed)
    return expand_proba(np.asarray(proba), seen, n_classes)


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _fit_row(
    sel: dict[str, Any],
    X: NDArray[np.float64],
    y: IntArray,
    perm: IntArray,
    predict: Predictor,
    ctx: dict[str, int],
) -> dict[str, Any]:
    n_classes = int(y.max()) + 1
    labeled = np.asarray(sel["labeled_pos"], dtype=np.int64)
    panel = np.arange(ctx["e"], dtype=np.int64)
    query = np.setdiff1d(panel, labeled)
    seed = fm_seed(ctx["base_seed"], ctx["did"], sel["m"], sel["rep"], sel["strategy"])
    start = time.perf_counter()
    proba = predict_on(predict, X, labeled, y, query, n_classes, seed)
    seen = np.unique(y[labeled]).astype(np.int64)
    metrics = metric_set(y[query], proba, np.arange(n_classes, dtype=np.int64), seen)
    return dict(
        strategy=sel["strategy"],
        m=int(sel["m"]),
        budget=len(labeled),
        rep=int(sel["rep"]),
        sel_seed=int(sel["sel_seed"]),
        fm_seed=seed,
        labeled=perm[labeled].tolist(),
        query=perm[query].tolist(),
        query_pos=query.tolist(),
        proba=proba.astype(np.float32).ravel(),
        n_classes=n_classes,
        n_stop=ctx["s"],
        coverage=len(seen) / n_classes,
        seen=seen.tolist(),
        seconds=time.perf_counter() - start,
        **metrics,
    )


def output_path(out_dir: Path, key: str, learner: str) -> Path:
    """``results/study9/{set}/{key}_{learner}.parquet``."""
    return out_dir / f"{key}_{learner}.parquet"


def _predictor(learner: str) -> Predictor:  # pragma: no cover - loads FMs
    from simaudit.runner import study4

    return study4._predictor(learner)


def dataset_key(did: int) -> str:
    """``openml_{did}`` or ``pmlb_{-did}`` (file-name safe)."""
    return f"openml_{did}" if did >= 0 else f"pmlb_{-did}"


def run_job(
    set_name: str,
    did: int,
    learner: str,
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    *,
    loader: Loader = load,
    predict: Predictor | None = None,
    embed: Embed | None = _default_embed,
) -> Path:
    """Every strategy x rep x budget for one dataset and learner (resumable)."""
    key = dataset_key(did)
    out = output_path(out_dir, key, learner)
    if out.exists():
        return out
    data = loader(did, cache_dir)
    perm = row_permutation(len(data.y), did, base_seed)
    X, y = data.X[perm], data.y[perm]
    chosen = selections(key, did, X, y, base_seed, out_dir, embed)
    fn = predict or _predictor(learner)
    s, e = layout(len(y))
    ctx = dict(base_seed=base_seed, did=did, s=s, e=e)
    partial = out.with_name(out.name.replace(".parquet", ".partial.parquet"))
    rows: list[dict[str, Any]] = []
    if partial.exists():
        rows = pd.read_parquet(partial).to_dict("records")  # type: ignore[assignment]
    done = {(r["strategy"], r["rep"], r["m"]) for r in rows}
    records = cast(list[dict[str, Any]], chosen.to_dict("records"))
    for sel in records:
        if (sel["strategy"], sel["rep"], sel["m"]) in done:
            continue
        rows.append(_fit_row(sel, X, y, perm, fn, ctx))
        if len(rows) % CHECKPOINT_EVERY == 0:
            _atomic_write(pd.DataFrame(rows), partial)
    frame = pd.DataFrame(rows)
    frame.insert(0, "learner", learner)
    frame.insert(0, "dataset", key)
    frame.insert(0, "set", set_name)
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
                base_seed=base_seed,
                seed_namespace=SEED_NAMESPACE,
                perm_namespace=PERM_NAMESPACE,
                grid=GRID,
                strategies=STRATEGIES,
                reps=REPS,
                stop_max=STOP_MAX,
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
    set_name: str,
    dataset_ids: list[int],
    learners: list[str],
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    workers: int,
) -> list[Path]:
    """Run every (learner, dataset) job; failures are logged and the rest go on."""
    _write_provenance(out_dir, base_seed, dataset_ids, learners)
    os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_THREADS))
    jobs = [(set_name, did, lr) for lr in learners for did in dataset_ids]
    done: list[Path] = []
    if workers == 1:
        for job in jobs:
            try:
                done.append(run_job(*job, cache_dir, out_dir, base_seed))
            except Exception as exc:  # noqa: BLE001 - logged; other jobs go on
                _log_failure(out_dir, job, exc)
        return done
    with ProcessPoolExecutor(  # pragma: no cover - multiprocess path
        max_workers=workers, initializer=_init_worker
    ) as pool:
        futures = {
            pool.submit(run_job, *job, cache_dir, out_dir, base_seed): job
            for job in jobs
        }
        for future in as_completed(futures):
            try:
                done.append(future.result())
            except Exception as exc:  # noqa: BLE001 - logged; other jobs go on
                _log_failure(out_dir, futures[future], exc)
                continue
            print(f"done {futures[future]} ({len(done)}/{len(jobs)})", flush=True)
    return done
