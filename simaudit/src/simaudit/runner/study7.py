"""Study 7: protocol and mechanism (prereg/STUDY7_PROTOCOL_MECHANISM.md, DEV only).

Part (a) re-runs the Study 4 core (six strategies x four FMs x four budgets) with
new seeds and stored probability matrices. Part (b) adds TabPFN v2 / v3.5 with
``balance_probabilities=True`` under random and stratified sampling. Part (c)
fits every FM on contexts with full coverage and prescribed class shares (the
dose-response test of the context-balance mechanism).

One job = one (part, learner, dataset); its parquet is written atomically and
skipped when present. Long jobs checkpoint to ``*.partial.parquet`` and resume.
"""

import json
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from simaudit.analysis.study7 import complement_metrics, prior_correct
from simaudit.data.openml_loader import load
from simaudit.runner import study4
from simaudit.runner.pilot import _provenance_name, source_digest
from simaudit.selection import strategies as sel

SEED_NAMESPACE = 10
MULTS = (1, 3, 5, 10)
DOSE_MULTS = (3, 5, 10)
LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9)
STRATEGIES = (
    "stratified",
    "random",
    "kmedoids",
    "kmeans_central",
    "typiclust",
    "lucos",
)
BALANCE_STRATEGIES = ("stratified", "random")
# Addendum 1 (S3): strategies whose selection may depend on row order.
ROW_ORDER_STRATEGIES = ("kmedoids", "kmeans_central", "typiclust", "lucos")
PART_STRATEGIES = {
    "a": STRATEGIES,
    "b": BALANCE_STRATEGIES,
    "p": ROW_ORDER_STRATEGIES,
}
REPS = {"random": 20}
DEFAULT_REPS = 5
DOSE_REPS = 10
PART_LEARNERS = {
    "a": ("fm1", "fm2", "fm3", "fm5"),
    "b": ("fm1_bal", "fm5_bal"),
    "c": ("fm1", "fm2", "fm3", "fm5"),
    "p": ("fm2",),  # Addendum 1 (S3), run after PLAN
}
# Pre-registered run order: (b) and (c) with the cheap FMs first, then (a),
# with Kumo last (prereg Sec. 0 / task brief).
PLAN = (
    ("c", "fm2"),
    ("c", "fm1"),
    ("b", "fm1_bal"),
    ("b", "fm5_bal"),
    ("c", "fm5"),
    ("a", "fm2"),
    ("a", "fm1"),
    ("a", "fm5"),
    ("a", "fm3"),
    ("c", "fm3"),
)
CHECKPOINT_EVERY = 20
TORCH_THREADS = 1
Predictor = Callable[..., NDArray[np.float64]]


def selection_seed(base_seed: int, did: int, mult: int, rep: int, name: str) -> int:
    """Seed of one Part (a)/(b) selection, namespace 10 (Study 4 used 5)."""
    state = np.random.SeedSequence(
        [base_seed, did, mult, rep, study4.ORDER.index(name), SEED_NAMESPACE]
    )
    return int(state.generate_state(1)[0] % (2**31 - 1))


def dose_seed(
    base_seed: int, did: int, mult: int, rep: int, share: float, counts: list[int]
) -> int:
    """Seed of one Part (c) context; identical designs share it."""
    state = np.random.SeedSequence(
        [base_seed, did, mult, rep, round(100 * share), *counts, SEED_NAMESPACE]
    )
    return int(state.generate_state(1)[0] % (2**31 - 1))


def _cell_rng(
    base_seed: int, did: int, mult: int, rep: int, share: float, upweighted: int
) -> np.random.Generator:
    """RNG of a (dataset, mult, rep, level, upweighted class) cell: remainders."""
    return np.random.default_rng(
        [base_seed, did, mult, rep, round(100 * share), upweighted, SEED_NAMESPACE, 1]
    )


def natural_classes(y: NDArray[np.int64]) -> tuple[int, int]:
    """Natural majority (ties: lowest index) and minority (ties: highest index)."""
    counts = np.bincount(y)
    majority = int(np.argmax(counts))
    minority = int(len(counts) - 1 - np.argmin(counts[::-1]))
    return majority, minority


def dose_counts(
    available: NDArray[np.int64],
    upweighted: int,
    share: float,
    mult: int,
    rng: np.random.Generator,
) -> NDArray[np.int64] | None:
    """Per-class counts for share s of the upweighted class; None if infeasible.

    n_u = round-half-up(s * B) clipped to [1, B - (C - 1)]; the rest is split
    equally over the other classes, extra rows to random classes. Infeasible if
    a class would need more than n_c - 1 rows.
    """
    n_classes = len(available)
    budget = mult * n_classes
    n_up = min(max(int(np.floor(share * budget + 0.5)), 1), budget - (n_classes - 1))
    others = np.array([c for c in range(n_classes) if c != upweighted])
    base, extra = divmod(budget - n_up, n_classes - 1)
    counts = np.zeros(n_classes, dtype=np.int64)
    counts[upweighted] = n_up
    counts[others] = base
    counts[rng.choice(others, size=extra, replace=False)] += 1
    return None if np.any(counts > available - 1) else counts


def draw_context(
    y: NDArray[np.int64], counts: NDArray[np.int64], seed: int
) -> NDArray[np.int64]:
    """Rows drawn uniformly at random within each class, ``counts[c]`` of class c."""
    rng = np.random.default_rng(seed)
    chosen = [
        rng.choice(np.flatnonzero(y == c), size=int(k), replace=False)
        for c, k in enumerate(counts)
    ]
    return np.sort(np.concatenate(chosen)).astype(np.int64)


@dataclass(frozen=True)
class DoseCell:
    """One Part (c) context design."""

    arm: str
    share: float
    mult: int
    rep: int
    upweighted: int
    counts: tuple[int, ...] | None
    seed: int

    @property
    def key(self) -> str:
        """Identify the row in checkpoints."""
        return f"{self.arm}|{self.share}|{self.mult}|{self.rep}"


def dose_cells(y: NDArray[np.int64], did: int, base_seed: int) -> list[DoseCell]:
    """Every Part (c) design: two arms x levels x mults x reps (+ multiclass anchor)."""
    available = np.bincount(y).astype(np.int64)
    majority, minority = natural_classes(y)
    arms = (("majority", majority), ("minority", minority))
    cells = []
    for mult, rep in product(DOSE_MULTS, range(DOSE_REPS)):
        for (arm, up), share in product(arms, LEVELS):
            rng = _cell_rng(base_seed, did, mult, rep, share, up)
            counts = dose_counts(available, up, share, mult, rng)
            cell = DoseCell(arm, share, mult, rep, up, None, -1)
            cells.append(_with_seed(cell, counts, did, base_seed))
        if len(available) > 2:
            cells.append(_anchor(available, mult, rep, did, base_seed))
    return cells


def _anchor(
    available: NDArray[np.int64], mult: int, rep: int, did: int, base_seed: int
) -> DoseCell:
    """Multiclass anchor: mult rows per class (share 1/C), if feasible."""
    n_classes = len(available)
    counts = np.full(n_classes, mult, dtype=np.int64)
    feasible = counts if np.all(counts <= available - 1) else None
    cell = DoseCell("anchor", 1 / n_classes, mult, rep, -1, None, -1)
    return _with_seed(cell, feasible, did, base_seed)


def _with_seed(
    cell: DoseCell, counts: NDArray[np.int64] | None, did: int, base_seed: int
) -> DoseCell:
    """Attach the counts and the design seed (unchanged if infeasible)."""
    if counts is None:
        return cell
    listed = [int(c) for c in counts]
    seed = dose_seed(base_seed, did, cell.mult, cell.rep, cell.share, listed)
    return replace(cell, counts=tuple(listed), seed=seed)


# ---------------------------------------------------------------------- learners
def tabpfn_predictor(version: str, balance: bool) -> Predictor:
    """Return TabPFN v2 or v3.5 as in Study 4, optionally with balancing on."""

    def predict(X_fit: Any, y_fit: Any, X_query: Any, seed: int) -> NDArray[np.float64]:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        model_version = ModelVersion.V2 if version == "v2" else ModelVersion.V3_5
        model = TabPFNClassifier.create_default_for_version(
            model_version,
            device="cpu",
            random_state=seed,
            balance_probabilities=balance,
        )
        model.fit(X_fit, y_fit)
        proba = np.zeros((len(X_query), int(y_fit.max()) + 1))
        proba[:, model.classes_] = model.predict_proba(X_query)
        return proba

    return predict


def predictor(learner: str) -> Predictor:
    """FM1-FM5 exactly as in Study 4; ``*_bal`` turn the balance option on."""
    if learner == "fm1_bal":
        return tabpfn_predictor("v2", balance=True)
    if learner == "fm5_bal":
        return tabpfn_predictor("v35", balance=True)
    return study4._predictor(learner)


# ---------------------------------------------------------------------- selections
def selection_path(out_dir: Path, did: int, name: str, permuted: bool = False) -> Path:
    """Cache of one strategy's selections for one dataset (learner-independent)."""
    suffix = "_perm" if permuted else ""
    return out_dir / "selections" / f"{did}_{name}{suffix}.parquet"


def row_permutation(base_seed: int, did: int, n: int) -> NDArray[np.int64]:
    """Fixed per-dataset row permutation of Addendum 1 (S3)."""
    rng = np.random.default_rng([base_seed, did, SEED_NAMESPACE, 99])
    return rng.permutation(n).astype(np.int64)


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def selections(
    did: int,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    name: str,
    base_seed: int,
    out_dir: Path,
    permuted: bool = False,
) -> pd.DataFrame:
    """Selections of one strategy for every mult and rep (computed once, cached).

    ``permuted`` selects on rows shuffled by ``row_permutation`` (same seeds) and
    maps the chosen rows back to the original indices.
    """
    path = selection_path(out_dir, did, name, permuted)
    if path.exists():
        return pd.read_parquet(path)
    order = row_permutation(base_seed, did, len(y)) if permuted else np.arange(len(y))
    rows = []
    for mult in MULTS:
        for rep in range(REPS.get(name, DEFAULT_REPS)):
            seed = selection_seed(base_seed, did, mult, rep, name)
            picked = study4.select(name, X[order], y[order], mult, seed)
            labeled = np.sort(order[picked])
            rows.append(dict(mult=mult, rep=rep, seed=seed, labeled=labeled.tolist()))
    frame = pd.DataFrame(rows)
    _atomic_write(frame, path)
    return frame


# ---------------------------------------------------------------------- jobs
def output_path(out_dir: Path, part: str, did: int, learner: str) -> Path:
    """``results/study7/{part}/{did}_{learner}.parquet``."""
    return out_dir / part / f"{did}_{learner}.parquet"


def _fit(
    predict: Predictor,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    labeled: NDArray[np.int64],
    seed: int,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    n_classes = int(y.max()) + 1
    start = time.perf_counter()
    proba = sel.predict_with_budget(predict, X, labeled, y[labeled], n_classes, seed)
    record = dict(
        n_classes=n_classes,
        labeled=labeled.tolist(),
        proba=proba.astype(np.float32).ravel(),
        coverage=sel.class_coverage(y, labeled),
        seen=np.unique(y[labeled]).tolist(),
        seconds=time.perf_counter() - start,
        **complement_metrics(y, proba, labeled),
    )
    return proba, record


def _strategy_cells(
    part: str,
    did: int,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    base_seed: int,
    out_dir: Path,
    predict: Predictor,
) -> Iterator[tuple[str, Callable[[], list[dict[str, Any]]]]]:
    for name in PART_STRATEGIES[part]:
        chosen = selections(did, X, y, name, base_seed, out_dir, part == "p")
        for r in chosen.to_dict("records"):
            labeled = np.asarray(r["labeled"], dtype=np.int64)
            meta = dict(strategy=name, mult=r["mult"], rep=r["rep"], seed=r["seed"])

            def run(
                labeled: NDArray[np.int64] = labeled, meta: dict[str, Any] = meta
            ) -> list[dict[str, Any]]:
                _, record = _fit(predict, X, y, labeled, int(meta["seed"]))
                return [dict(meta, budget=len(labeled), **record)]

            yield f"{name}|{r['mult']}|{r['rep']}", run


def _dose_cells(
    did: int,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    base_seed: int,
    predict: Predictor,
) -> Iterator[tuple[str, Callable[[], list[dict[str, Any]]]]]:
    """Group Part (c) designs by seed so identical designs are fitted once."""
    by_seed: dict[int, list[DoseCell]] = {}
    for cell in dose_cells(y, did, base_seed):
        if cell.counts is None:
            yield cell.key, partial(_infeasible_rows, cell, did)
        else:
            by_seed.setdefault(cell.seed, []).append(cell)
    for seed, group in by_seed.items():

        def run(
            seed: int = seed, group: list[DoseCell] = group
        ) -> list[dict[str, Any]]:
            counts = np.array(group[0].counts, dtype=np.int64)
            labeled = draw_context(y, counts, seed)
            proba, record = _fit(predict, X, y, labeled, seed)
            corrected = complement_metrics(y, prior_correct(proba, y[labeled]), labeled)
            record.update({f"pc_{k}": v for k, v in corrected.items()})
            return [
                dict(_dose_meta(c, did, feasible=True), shared=len(group) > 1, **record)
                for c in group
            ]

        yield "+".join(c.key for c in group), run


def _infeasible_rows(cell: DoseCell, did: int) -> list[dict[str, Any]]:
    """Log an infeasible design without fitting."""
    return [_dose_meta(cell, did, feasible=False)]


def _dose_meta(cell: DoseCell, did: int, feasible: bool) -> dict[str, Any]:
    counts = list(cell.counts or ())
    budget = sum(counts)
    real = float("nan")
    if counts:
        anchor = cell.upweighted < 0
        real = 1 / len(counts) if anchor else counts[cell.upweighted] / budget
    return dict(
        dataset=did,
        cell=cell.key,
        arm=cell.arm,
        share=cell.share,
        mult=cell.mult,
        rep=cell.rep,
        upweighted=cell.upweighted,
        seed=cell.seed,
        feasible=feasible,
        counts=counts,
        budget=budget,
        share_real=real,
    )


def _resume(
    out: Path, cells: Iterator[tuple[str, Callable[[], list[dict[str, Any]]]]]
) -> pd.DataFrame:
    """Run cells not in the checkpoint, flushing it every CHECKPOINT_EVERY cells."""
    partial = out.with_name(out.name.replace(".parquet", ".partial.parquet"))
    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if partial.exists():
        old = pd.read_parquet(partial)
        rows = cast(list[dict[str, Any]], old.to_dict("records"))
        done = set(old["_key"])
    fresh = 0
    for key, run in cells:
        if key in done:
            continue
        rows.extend(dict(r, _key=key) for r in run())
        fresh += 1
        if fresh % CHECKPOINT_EVERY == 0:
            _atomic_write(pd.DataFrame(rows), partial)
    frame = pd.DataFrame(rows).drop(columns="_key")
    _atomic_write(frame, out)
    partial.unlink(missing_ok=True)
    return frame


def run_job(
    part: str,
    learner: str,
    did: int,
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    predict: Predictor | None = None,
    drop_columns: tuple[int, ...] = (),
) -> Path:
    """One (part, learner, dataset) job; skipped if its parquet exists.

    ``drop_columns`` removes feature columns first (Addendum 1, S4: 694's ID).
    """
    if learner not in PART_LEARNERS[part]:
        raise ValueError(f"learner {learner} is not in part {part}")
    out = output_path(out_dir, part, did, learner)
    if out.exists():
        return out
    fn = predict or predictor(learner)
    data = load(did, cache_dir)
    X = np.delete(data.X, list(drop_columns), axis=1) if drop_columns else data.X
    if part == "c":
        cells = _dose_cells(did, X, data.y, base_seed, fn)
    else:
        cells = _strategy_cells(part, did, X, data.y, base_seed, out_dir, fn)
    frame = _resume(out, cells)
    frame["n_classes"] = int(data.y.max()) + 1  # also on infeasible rows
    frame.insert(0, "learner", learner)
    if "dataset" not in frame:
        frame.insert(0, "dataset", did)
    _atomic_write(frame, out)
    return out


def _init_worker() -> None:
    import torch

    torch.set_num_threads(TORCH_THREADS)


def _write_provenance(
    out_dir: Path, base_seed: int, dataset_ids: list[int], plan: tuple[Any, ...]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _provenance_name()).write_text(
        json.dumps(
            dict(
                base_seed=base_seed,
                seed_namespace=SEED_NAMESPACE,
                mults=MULTS,
                dose_mults=DOSE_MULTS,
                levels=LEVELS,
                strategies=STRATEGIES,
                reps=dict(REPS, default=DEFAULT_REPS, dose=DOSE_REPS),
                plan=[list(p) for p in plan],
                datasets=dataset_ids,
                source_sha256=source_digest(Path(__file__).resolve().parents[1]),
            ),
            indent=2,
        )
    )


def run(
    dataset_ids: list[int],
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    workers: int,
    plan: tuple[tuple[str, str], ...] = PLAN,
    drop: dict[int, tuple[int, ...]] | None = None,
) -> list[Path]:
    """Submit jobs in plan order; failures are logged and the rest go on.

    ``drop`` maps a dataset to feature columns removed before selection and fit.
    """
    drop = drop or {}
    _write_provenance(out_dir, base_seed, dataset_ids, plan)
    os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_THREADS))
    done = []
    jobs = [(part, learner, did) for part, learner in plan for did in dataset_ids]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {
            pool.submit(
                run_job,
                *job,
                cache_dir,
                out_dir,
                base_seed,
                None,
                drop.get(job[2], ()),
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                done.append(future.result())
            except Exception as exc:  # noqa: BLE001 - logged; other jobs go on
                with (out_dir / "failures.jsonl").open("a") as handle:
                    handle.write(json.dumps(dict(job=job, error=repr(exc))) + "\n")
                print(f"FAILED {job}: {exc!r}", flush=True)
                continue
            print(f"done {job} ({len(done)}/{len(jobs)})", flush=True)
    return done
