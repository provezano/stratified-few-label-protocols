"""Study 10 runner: confirmation of the consensus-core family (prereg Sec. 12).

prereg/STUDY10_SELECTOR_CONFIRMATION.md fixes everything here. Per dataset,
rows are permuted once (namespace 14) before any selection or fit; every index
below is in that permuted order and is mapped back only when stored. Every
family file and the scalable-competitor module are pinned by SHA-256 and
checked before a run. Selections see only the permuted X, the budget and C
(never y); they are cached once per (set, dataset, arm) and shared by learners.
Each fit predicts the panel rows E_d minus the labeled rows and stores the
probabilities, so the N2 common-set rule (Sec. 3.3) is applied at analysis
time. ``--set heldout`` refuses to run unless the prereg is FROZEN and every
dataset matches its pool hash.
"""

import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import re
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from simaudit.analysis.study7 import metric_set
from simaudit.autoresearch.select_harness import SelectTask
from simaudit.data.openml_loader import Dataset, load
from simaudit.graph.knn import zscore
from simaudit.runner import study4, study9_heldout, study13
from simaudit.runner.pilot import _provenance_name, source_digest
from simaudit.runner.study9 import dataset_key, predict_on, seed_did
from simaudit.selection import lucos, scalable
from simaudit.selection import strategies as sel

BASE_SEED = 20261020
SEED_NAMESPACE = 8  # reserved for this confirmation by Study 5 Sec. 6
PERM_NAMESPACE = 14  # Study 9 uses 13
MULTS = (1, 3, 5, 10)
REPS = {"random": 10}
DEFAULT_REPS = 5
PANEL_MAX = 2000
N2_MAX_N = 160  # Sec. 3.3: common set for TypiClust / k-center when n <= 160
N2_ARMS = ("typiclust", "kcenter")
LANDMARK_FLOOR, LANDMARK_PER_B = 2000, 40
CHECKPOINT_EVERY = 25
TORCH_THREADS = 1
# TabICL orders its ensemble groups by list(set(...)) of strings, so its float32
# output follows the string-hash seed; spawned workers get a fixed one (G4, T7).
HASH_SEED = "0"
CANDIDATE = Path("autoresearch/workstream_b_v3/rescore/main_D.py")
CANDIDATE_SHA256 = "ac735dd73caad3ea974af06dffa4a1c8617769b5723a256e7b7b87705ddae25e"
PREREG = Path("prereg/STUDY10_SELECTOR_CONFIRMATION.md")
# Sec. 4.1: every family file and the scalable module, pinned (G3).
PINNED: dict[str, tuple[str, str]] = {
    "d22_v3": (str(CANDIDATE), CANDIDATE_SHA256),
    "f1_e5": (
        "autoresearch/workstream_b_v3/efficiency/E5_F1_landmark.py",
        "58cee1cc27950af2adfc7f5b43a8fa1254fd971d72d49c3d4af41fd15745b274",
    ),
    "f1_sub": (
        "autoresearch/workstream_b_v3/efficiency/knobs/F1_lmcore.py",
        "aa9fa02815d6786059fbb932e83029aa0f695e7c57e544e5740d11b9c7ec5395",
    ),
    "f3_e7": (
        "autoresearch/workstream_b_v3/efficiency/E7_F3_streaming.py",
        "95c049a73feb80556a149457a20462a426ceac375a584a0e458ae553aba5b10a",
    ),
    "e9_ik": (
        "autoresearch/workstream_b_scalable/E9_IK.py",
        "8d931a219120fc9fb1106cc7a72c89268232aac80b5cf39205ab4ff87ccdf54a",
    ),
    "scalable": (
        "src/simaudit/selection/scalable.py",
        "e70978415cf8bb1fea20eb6574ab9583594c381fa11ea1b58c85125a763e4970",
    ),
}
# Family arm -> (pinned file key, module constants set after loading).
FAMILY: dict[str, tuple[str, dict[str, Any]]] = {
    "d22_v3": ("d22_v3", {}),
    "f1_e5": ("f1_e5", {}),
    "f1_sub": ("f1_sub", {}),
    "f3_e7": ("f3_e7", {}),
    "e9_ik": ("e9_ik", {}),
    "e9_ikproj": ("e9_ik", {"IK_SPACE": "proj"}),  # the make_perm.py knob
}
PRIMARY_COMPARATORS = ("lucos_fps", "kmedoids")
COMPARATORS = (
    "lucos_fps",
    "kmedoids",
    "lucos",
    "random",
    "kmeans_central",
    "typiclust",
    "kcenter",
    "probcover",
    "usl",
    "rdss",
    "zcore",
)
SENSITIVITY_ARMS = ("lucos_n10",)
ARMS = (*FAMILY, *COMPARATORS, *SENSITIVITY_ARMS)
ARM_ORDER = (  # append only: the position fixes each arm's seeds
    *study4.ORDER,
    "d22_v3",
    "lucos_n10",
    "f1_e5",
    "f1_sub",
    "f3_e7",
    "e9_ik",
    "e9_ikproj",
    "lucos_fps",
)
PRIMARY_LEARNER = "fm2"
OPEN_LEARNERS = ("fm1", "fm3", "logreg")
SENSITIVITY_LEARNERS = ("fm5",)
LEARNERS = (PRIMARY_LEARNER, *OPEN_LEARNERS, *SENSITIVITY_LEARNERS)
SETS = ("dev", "heldout", "t1prime")

IntArray = NDArray[np.int64]
Predictor = Callable[..., NDArray[np.float64]]
Loader = Callable[[int, Path], Dataset]
_MODULES: dict[tuple[str, str], ModuleType] = {}


# ---------------------------------------------------------------------- pins
def check_file(path: Path, expected: str) -> None:
    """Raise unless ``path`` is byte-identical to the pinned file."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"{path} has SHA-256 {digest}, expected {expected}")


def check_arms(root: Path) -> None:
    """Raise, naming the arm, if any pinned file differs from its hash (T1)."""
    for name, (rel, expected) in PINNED.items():
        try:
            check_file(root / rel, expected)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc


def check_frozen(prereg: Path) -> None:
    """Raise unless the prereg's status line reads FROZEN (HELD-OUT guard)."""
    status = re.search(r"^Status:\s*\**\s*(\w+)", prereg.read_text(), re.MULTILINE)
    if status is None or status.group(1) != "FROZEN":
        raise PermissionError(f"{prereg} is not FROZEN; HELD-OUT stays sealed")


def check_heldout_hash(data: Dataset, expected: dict[int, str]) -> None:
    """Raise unless the dataset's content hash equals its pool entry."""
    want = expected.get(data.did)
    if want is None or data.sha256 != want:
        raise ValueError(f"HELD-OUT {data.did}: hash {data.sha256} != pool {want}")


def heldout_loader(
    root: Path, sealed: Path, out_dir: Path, env: Mapping[str, str] | None = None
) -> study13.SealedLoader:
    """The joint-opening HELD-OUT loader of Studies 9 and 13 (prereg Sec. 3.5).

    Checks, in this order and before any sealed file is touched: the joint go
    (``study13.check_heldout_go``: ``P1_HELDOUT_GO=1`` and Study 13 frozen) and
    this prereg's FROZEN status line; then ``study13.verify_manifest``. Returns
    ``study13.SealedLoader(normalise_labels=True)``: every load is logged to
    ``out_dir/access_log.jsonl`` and re-checks the pool content hash, and the
    label step of ``data.labels`` runs (merged spellings; unverified tables keep
    their stored labels and are flagged). ``label_normalisation.csv`` is written
    before any selection, as ``runner/study9_heldout.run`` does.
    """
    study13.check_heldout_go(root, env)
    check_frozen(root / PREREG)
    pool = study9_heldout.heldout_pool(root)
    log_path = out_dir / "access_log.jsonl"
    study13.verify_manifest(sealed, sorted(pool), study13.AccessLog(log_path))
    loader = study13.SealedLoader(sealed, pool, log_path, normalise_labels=True)
    study13.label_log(sorted(pool), loader, out_dir)
    return loader


# ---------------------------------------------------------------------- seeds
def row_permutation(n: int, did: int, base_seed: int = BASE_SEED) -> IntArray:
    """Fixed per-dataset row permutation (namespace 14) against row-order leaks."""
    rng = np.random.default_rng([base_seed, seed_did(did), PERM_NAMESPACE])
    return rng.permutation(n).astype(np.int64)


def cell_seed(
    did: int, mult: int, rep: int, arm: str, base_seed: int = BASE_SEED
) -> int:
    """Seed of one (dataset, mult, rep, arm) cell; drives selection and learner."""
    state = np.random.SeedSequence(
        [base_seed, seed_did(did), mult, rep, ARM_ORDER.index(arm), SEED_NAMESPACE]
    )
    return int(state.generate_state(1)[0] % (2**31 - 1))


def reps(arm: str) -> int:
    """Repetitions per arm (Sec. 4.1): random 10, every other arm 5."""
    return REPS.get(arm, DEFAULT_REPS)


def landmark_active(n: int, budget: int) -> bool:
    """True when the landmark members subsample (s < n)."""
    return n > max(LANDMARK_FLOOR, LANDMARK_PER_B * budget)


# ---------------------------------------------------------------------- arms
def load_arm(name: str, root: Path) -> ModuleType:
    """Load a family file by path (hash-checked) and apply its module constants."""
    key, overrides = FAMILY[name]
    rel, expected = PINNED[key]
    cache_key = (name, str(root))
    if cache_key not in _MODULES:
        path = root / rel
        check_file(path, expected)
        spec = importlib.util.spec_from_file_location(f"study10_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr, value in overrides.items():
            setattr(module, attr, value)
        _MODULES[cache_key] = module
    return _MODULES[cache_key]


def _lucos_n10(X: NDArray[np.float64], budget: int, seed: int) -> IntArray:
    return lucos.select_lucos(X, budget, seed, n_init=10)


def select_arm(
    name: str, X: NDArray[np.float64], n_classes: int, mult: int, seed: int, root: Path
) -> IntArray:
    """Rows chosen by one arm from the (permuted) features; labels are never passed."""
    budget = mult * n_classes
    if name in FAMILY:
        task = SelectTask(X, budget, n_classes, seed)
        return np.asarray(load_arm(name, root).select(task)).ravel().astype(np.int64)
    if name in sel.CLASS_AWARE:
        return sel.CLASS_AWARE[name](X, budget, seed, n_classes=n_classes)
    named: dict[str, sel.Strategy] = {
        "lucos_fps": scalable.select_lucos_fps,
        "lucos_n10": _lucos_n10,
        "lucos": lucos.select_lucos,
    }
    fn = named.get(name) or {**sel.STRATEGIES, **study4.EXTRA}[name]
    return np.asarray(fn(X, budget, seed)).ravel().astype(np.int64)


def repair(
    chosen: NDArray[Any], budget: int, X: NDArray[np.float64]
) -> tuple[IntArray, int]:
    """Common validity rule (Sec. 4.3): keep first occurrences, fill farthest-first."""
    n = len(X)
    kept = list(
        dict.fromkeys(int(i) for i in np.asarray(chosen).ravel() if 0 <= int(i) < n)
    )
    kept = kept[:budget]
    filled = budget - len(kept)
    if filled:
        Z = zscore(X)
        if not kept:
            kept = [0]  # nothing valid: start the fill at row 0
        dist = np.min([np.linalg.norm(Z - Z[i], axis=1) for i in kept], axis=0)
        dist[kept] = -1.0
        while len(kept) < budget:
            nxt = int(np.argmax(dist))  # ties -> lowest index
            kept.append(nxt)
            dist = np.minimum(dist, np.linalg.norm(Z - Z[nxt], axis=1))
            dist[kept] = -1.0
    return np.sort(np.asarray(kept, dtype=np.int64)), filled


# ---------------------------------------------------------------------- selections
def selection_path(out_dir: Path, did: int, arm: str) -> Path:
    """Cache of one arm's selections for one dataset (learner-independent)."""
    return out_dir / "selections" / f"{dataset_key(did)}_{arm}.parquet"


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def selections(
    did: int,
    X: NDArray[np.float64],
    n_classes: int,
    arm: str,
    out_dir: Path,
    root: Path,
) -> pd.DataFrame:
    """Selections of one arm for every mult and rep (computed once, cached)."""
    path = selection_path(out_dir, did, arm)
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    for mult in MULTS:
        for rep in range(reps(arm)):
            seed = cell_seed(did, mult, rep, arm)
            start = time.process_time()
            picked = select_arm(arm, X, n_classes, mult, seed, root)
            cpu = time.process_time() - start
            fixed, filled = repair(picked, mult * n_classes, X)
            rows.append(
                dict(
                    arm=arm,
                    mult=mult,
                    rep=rep,
                    seed=seed,
                    labeled_pos=fixed.tolist(),
                    repaired=filled > 0 or len(picked) != len(fixed),
                    n_filled=filled,
                    select_cpu_s=cpu,
                )
            )
    frame = pd.DataFrame(rows)
    _atomic_write(frame, path)
    return frame


def common_seed_identity(
    f0: pd.DataFrame, X: NDArray[np.float64], n_classes: int, root: Path
) -> pd.DataFrame:
    """H3 identity share (Sec. 7.2): F1 and F1-sub re-selected at F0's cell seeds.

    Selections only (no fit, no label). One row per F0 cell, with whether F1's
    selection equals F0's, whether F1-sub's equals F1's, and how many rows differ.
    """
    rows = []
    for cell in f0.to_dict("records"):
        mult, seed = int(cell["mult"]), int(cell["seed"])
        f0_rows = np.asarray(cell["labeled_pos"], dtype=np.int64)
        f1, _ = repair(
            select_arm("f1_e5", X, n_classes, mult, seed, root), mult * n_classes, X
        )
        sub, _ = repair(
            select_arm("f1_sub", X, n_classes, mult, seed, root), mult * n_classes, X
        )
        rows.append(
            dict(
                mult=mult,
                rep=int(cell["rep"]),
                seed=seed,
                f1_eq_f0=bool(np.array_equal(f1, f0_rows)),
                f1sub_eq_f1=bool(np.array_equal(sub, f1)),
                n_rows_differ=len(np.setdiff1d(f1, f0_rows)),
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- learners
def zscore_all(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Label-free z-scoring with the statistics of all n rows (logreg input)."""
    return np.asarray(zscore(X), dtype=np.float64)


def logreg(
    X_fit: NDArray[np.float64], y_fit: IntArray, X_query: NDArray[np.float64], seed: int
) -> NDArray[np.float64]:
    """The non-FM learner of Sec. 5 (C = 1, lbfgs, max_iter 2000)."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
    model.fit(X_fit, y_fit)
    return np.asarray(model.predict_proba(X_query), dtype=np.float64)


def fit_predict(
    predict: Predictor,
    X: NDArray[np.float64],
    labeled: IntArray,
    y: IntArray,
    query: IntArray,
    n_classes: int,
    seed: int,
) -> NDArray[np.float64]:
    """Fit on the labeled rows (seen classes only) and predict ``query``."""
    return predict_on(predict, X, labeled, y, query, n_classes, seed)


def panel_proba(
    predict: Predictor,
    X: NDArray[np.float64],
    labeled: IntArray,
    y: IntArray,
    seed: int,
) -> tuple[IntArray, NDArray[np.float64]]:
    """Query rows E_d minus L and their probabilities (G4, T7).

    The learner predicts the whole panel E_d (labeled rows included) and the
    query rows are taken from it. TabICL's float32 output depends on which rows
    share its test batch (up to about 1e-6), so predicting the query rows alone
    would not reproduce the harness, which predicted every row; on DEV and
    HELD-OUT (n <= PANEL_MAX) E_d is every row, as in the harness.
    """
    n_classes = int(y.max()) + 1
    panel = np.arange(min(len(y), PANEL_MAX), dtype=np.int64)
    query = np.setdiff1d(panel, labeled)
    proba = fit_predict(predict, X, labeled, y, panel, n_classes, seed)
    return query, proba[query]


def _predictor(learner: str) -> Predictor:  # pragma: no cover - loads FMs
    return logreg if learner == "logreg" else study4._predictor(learner)


def _metrics(
    y: IntArray, proba: NDArray[np.float64], query: IntArray, labeled: IntArray
) -> dict[str, float]:
    n_classes = proba.shape[1]
    seen = np.unique(y[labeled]).astype(np.int64)
    full = metric_set(y[query], proba, np.arange(n_classes, dtype=np.int64), seen)
    keep = np.isin(y[query], seen)
    seen_only = metric_set(y[query][keep], proba[keep], seen, seen)
    return {**full, **{f"seen_{k}": v for k, v in seen_only.items()}}


def _fit_row(
    sel_row: dict[str, Any],
    X: NDArray[np.float64],
    y: IntArray,
    perm: IntArray,
    predict: Predictor,
) -> dict[str, Any]:
    n_classes = int(y.max()) + 1
    labeled = np.asarray(sel_row["labeled_pos"], dtype=np.int64)
    start = time.perf_counter()
    query, proba = panel_proba(predict, X, labeled, y, int(sel_row["seed"]))
    seen = np.unique(y[labeled])
    return dict(
        arm=sel_row["arm"],
        mult=int(sel_row["mult"]),
        budget=len(labeled),
        rep=int(sel_row["rep"]),
        seed=int(sel_row["seed"]),
        labeled_pos=labeled.tolist(),
        labeled=perm[labeled].tolist(),
        query_pos=query.tolist(),
        proba=proba.astype(np.float32).ravel(),
        n_classes=n_classes,
        coverage=len(seen) / n_classes,
        repaired=bool(sel_row["repaired"]),
        n_filled=int(sel_row["n_filled"]),
        select_cpu_s=float(sel_row["select_cpu_s"]),
        landmark_active=landmark_active(len(y), len(labeled)),
        seconds=time.perf_counter() - start,
        **_metrics(y, proba, query, labeled),
    )


# ---------------------------------------------------------------------- jobs
def output_path(out_dir: Path, did: int, learner: str) -> Path:
    """``results/study10/{set}/{key}_{learner}.parquet``."""
    return out_dir / f"{dataset_key(did)}_{learner}.parquet"


def run_job(
    set_name: str,
    did: int,
    learner: str,
    cache_dir: Path,
    out_dir: Path,
    *,
    loader: Loader = load,
    predict: Predictor | None = None,
    root: Path = Path("."),
    arms: tuple[str, ...] = ARMS,
    heldout_hashes: dict[int, str] | None = None,
) -> Path:
    """Every arm x mult x rep for one dataset and learner (resumable)."""
    out = output_path(out_dir, did, learner)
    if out.exists():
        return out
    data = loader(did, cache_dir)
    if heldout_hashes is not None:
        check_heldout_hash(data, heldout_hashes)
    perm = row_permutation(len(data.y), did)
    X, y = data.X[perm], data.y[perm]
    n_classes = int(y.max()) + 1
    chosen = [selections(did, X, n_classes, a, out_dir, root) for a in arms]
    fn = predict or _predictor(learner)
    X_fit = zscore_all(X) if learner == "logreg" else X
    partial = out.with_name(out.name.replace(".parquet", ".partial.parquet"))
    rows: list[dict[str, Any]] = []
    if partial.exists():
        rows = cast(list[dict[str, Any]], pd.read_parquet(partial).to_dict("records"))
    done = {(r["arm"], r["mult"], r["rep"]) for r in rows}
    for frame in chosen:
        for sel_row in cast(list[dict[str, Any]], frame.to_dict("records")):
            if (sel_row["arm"], sel_row["mult"], sel_row["rep"]) in done:
                continue
            rows.append(_fit_row(sel_row, X_fit, y, perm, fn))
            if len(rows) % CHECKPOINT_EVERY == 0:
                _atomic_write(pd.DataFrame(rows), partial)
    result = pd.DataFrame(rows)
    result.insert(0, "learner", learner)
    result.insert(0, "n", len(y))
    result.insert(0, "dataset", dataset_key(did))
    result.insert(0, "set", set_name)
    _atomic_write(result, out)
    partial.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------------- scoring
def scoring_rows(
    panel: IntArray,
    labeled_a: IntArray,
    labeled_b: IntArray,
    arm_a: str,
    arm_b: str,
    n: int,
) -> IntArray:
    """Rows on which arm a is scored in the contrast a - b (N2 rule, Sec. 3.3)."""
    if n <= N2_MAX_N and (arm_a in N2_ARMS or arm_b in N2_ARMS):
        return np.setdiff1d(panel, np.union1d(labeled_a, labeled_b))
    return np.setdiff1d(panel, labeled_a)


# ---------------------------------------------------------------------- driver
def _init_worker() -> None:  # pragma: no cover - runs in worker processes
    import torch

    torch.set_num_threads(TORCH_THREADS)


def _write_provenance(
    out_dir: Path, ids: list[int], learners: list[str], root: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _provenance_name()).write_text(
        json.dumps(
            dict(
                base_seed=BASE_SEED,
                seed_namespace=SEED_NAMESPACE,
                perm_namespace=PERM_NAMESPACE,
                mults=MULTS,
                arms=ARMS,
                arm_order=ARM_ORDER,
                reps=dict(REPS, default=DEFAULT_REPS),
                panel_max=PANEL_MAX,
                pinned={k: v[1] for k, v in PINNED.items()},
                learners=learners,
                datasets=ids,
                python_hash_seed=HASH_SEED,
                source_sha256=source_digest(Path(__file__).resolve().parents[1]),
                root=str(root),
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
    workers: int,
    root: Path,
    heldout_hashes: dict[int, str] | None = None,
    loader: Loader = load,
) -> list[Path]:
    """Run every (learner, dataset) job in learner order; failures are logged.

    HELD-OUT needs the pool hashes and the sealed loader of ``heldout_loader``.
    """
    if set_name == "heldout":
        check_frozen(root / PREREG)
        if heldout_hashes is None:
            raise ValueError("HELD-OUT needs the pool hashes")
        if not isinstance(loader, study13.SealedLoader) or not loader.normalise_labels:
            raise PermissionError("HELD-OUT needs heldout_loader (Sec. 3.5)")
    check_arms(root)
    _write_provenance(out_dir, dataset_ids, learners, root)
    os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_THREADS))
    os.environ["PYTHONHASHSEED"] = HASH_SEED  # inherited by spawned workers
    jobs = [(set_name, did, lr) for lr in learners for did in dataset_ids]
    kw: dict[str, Any] = dict(root=root, heldout_hashes=heldout_hashes, loader=loader)
    done: list[Path] = []
    if workers == 1:
        for job in jobs:
            try:
                done.append(run_job(*job, cache_dir, out_dir, **kw))
            except Exception as exc:  # noqa: BLE001 - logged; other jobs go on
                _log_failure(out_dir, job, exc)
        return done
    with ProcessPoolExecutor(  # pragma: no cover - multiprocess path
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
    ) as pool:
        futures = {
            pool.submit(run_job, *job, cache_dir, out_dir, **kw): job for job in jobs
        }
        for future in as_completed(futures):
            try:
                done.append(future.result())
            except Exception as exc:  # noqa: BLE001 - logged; other jobs go on
                _log_failure(out_dir, futures[future], exc)
                continue
            print(f"done {futures[future]} ({len(done)}/{len(jobs)})", flush=True)
    return done
