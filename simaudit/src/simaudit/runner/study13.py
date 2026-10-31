"""Study 13: Study 7 Part (a) confirmed on HELD-OUT (prereg Study 13).

Prereg: prereg/STUDY13_HELDOUT_CONFIRMATION.md.

The prereg is a DRAFT until the owner freezes it (file renamed from
``DRAFT_STUDY13_HELDOUT_CONFIRMATION.md``). Per table, rows are permuted once
(namespace 35). Arm T is Study 7 Part (a): stratified (5 reps) and random (20
reps) labeled sets drawn from all rows, scored on the complement, plus the
redraw arm (covered random draws). Arm F uses a fixed class-stratified test
set T_d (namespace 36) defined before any draw: labeled sets are drawn from
the other rows and every fit is scored on T_d. Learners are the four FMs and,
as one job group, the six classical learners of exploratory E1.

HELD-OUT guard: ``set_name == "heldout"`` raises unless ``P1_HELDOUT_GO=1``,
the frozen prereg exists under its frozen name with a FROZEN status line, the
draft file is gone, and the frozen file is tracked by git and unchanged
against HEAD. Only then is ``sealed_heldout/MANIFEST.json`` read and every npz
hashed (bytes, unparsed) before any is loaded. Every access is logged.

Amendment 1 (2026-10-28, before any HELD-OUT access):
(A) on HELD-OUT, ``SealedLoader(normalise_labels=True)`` recodes label
spellings that differ only in case, whitespace, ``_``, ``-`` or ``.``
(``data.labels``), after the content-hash check and before the screen or any
draw; every table and mapping goes to the access log and to
``label_normalisation.csv``. (B) secondary FMs fm6 (TabDPT v1.3) and fm7
(Mitra-v2) run arm T stratified/random only, each in its own pinned venv
through ``runner/fm_extra_worker.py``; fm7 is skipped on tables with more than
10 classes. (C) every fit also stores kappa, MCC and CITL
(``analysis.study13.e17_panel``).

Amendment 3 (2026-10-29, before any HELD-OUT access): fm8 (LimiX-16M) joins
fm6 and fm7 as a third secondary FM, arm T stratified/random only, in
``.venv_fm_extra_limix`` through the same worker; fm8 is skipped on tables
with more than 10 classes (its decoder has 10 outputs).
"""

import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from simaudit.analysis.study7 import prior_correct
from simaudit.analysis.study12 import panel_metrics
from simaudit.analysis.study13 import e17_panel
from simaudit.data import labels
from simaudit.data.integrity import _monotone, screen
from simaudit.data.openml_loader import Dataset, content_hash, load
from simaudit.runner import fm_extra_worker as fmw
from simaudit.runner import study4
from simaudit.runner.study9 import dataset_key, predict_on, seed_did

SEED_NAMESPACE = 34
PERM_NAMESPACE = 35
SPLIT_NAMESPACE = 36
ARM_T, ARM_F = 0, 2
REDRAW_TAG = 1
MULTS = (1, 3, 5, 10)
STRATEGIES = ("stratified", "random")
REPS = {"stratified": 5, "random": 20}
TEST_SHARE = 0.3
MAX_MULT = max(MULTS)
MAX_ATTEMPTS = 100_000
FMS = ("fm2", "fm5", "fm1", "fm3")  # prereg Sec. 7 order
CLASSICAL = ("lr", "knn1", "knn3", "gnb", "lda", "hgb")
EXTRA_FMS = ("fm6", "fm7", "fm8")  # Amendments 1 (B) and 3: secondary, arm T only
GROUPS = (*FMS, "classical", *EXTRA_FMS)
EXTRA_VENVS = {
    "fm6": ".venv_fm_extra",
    "fm7": ".venv_fm_extra_mitra",
    "fm8": ".venv_fm_extra_limix",
}
MITRA_MAX_CLASSES = 10
LIMIX_MAX_CLASSES = 10  # Amendment 3: LimiX-16M decoder outputs
ROOT = Path(__file__).resolve().parents[3]
WORKER = Path(__file__).resolve().with_name("fm_extra_worker.py")
SETS = ("dev", "heldout")
CHECKPOINT_EVERY = 25
TORCH_THREADS = 1

FROZEN_NAME = "STUDY13_HELDOUT_CONFIRMATION.md"
DRAFT_NAME = "DRAFT_STUDY13_HELDOUT_CONFIRMATION.md"
DRAFT_MARKER = "DRAFT — NOT FROZEN"
GO_ENV = "P1_HELDOUT_GO"

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Predictor = Callable[..., FloatArray]
Loader = Callable[[int, Path], Dataset]


# ---------------------------------------------------------------------- guard
def check_heldout_go(root: Path, env: Mapping[str, str] | None = None) -> Path:
    """Raise unless every HELD-OUT precondition of prereg Sec. 9 holds.

    ``root`` is the repository root (holds ``prereg/``). Returns the frozen
    prereg path.
    """
    env = os.environ if env is None else env
    if env.get(GO_ENV) != "1":
        raise PermissionError(f"{GO_ENV}=1 is not set; HELD-OUT stays sealed")
    frozen = root / "prereg" / FROZEN_NAME
    if not frozen.exists():
        raise PermissionError(f"{frozen} does not exist; the prereg is not frozen")
    if (root / "prereg" / DRAFT_NAME).exists():
        raise PermissionError("the DRAFT prereg still exists; rename it when freezing")
    text = frozen.read_text()
    status = re.search(r"^Status:\s*\**\s*(\w+)", text, re.MULTILINE)
    if status is None or status.group(1) != "FROZEN" or DRAFT_MARKER in text:
        raise PermissionError(f"{frozen} is not FROZEN; HELD-OUT stays sealed")
    rel = frozen.relative_to(root)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(rel)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise PermissionError(f"{rel} is not committed")
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(rel)], cwd=root)
    if clean.returncode != 0:
        raise PermissionError(f"{rel} differs from HEAD; commit the frozen prereg")
    return frozen


class AccessLog:
    """Append-only JSONL log of every HELD-OUT access (UTC timestamps)."""

    def __init__(self, path: Path) -> None:
        """Log to ``path`` (parents created)."""
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(
        self, target: Path, action: str, detail: Mapping[str, Any] | None = None
    ) -> None:
        """Append one access entry (``detail``: e.g. a label mapping)."""
        entry: dict[str, Any] = dict(
            utc=datetime.now(UTC).isoformat(timespec="milliseconds"),
            path=str(target),
            action=action,
            pid=os.getpid(),
        )
        if detail is not None:
            entry["detail"] = dict(detail)
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes (the file is not parsed)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(sealed: Path, ids: list[int], log: AccessLog) -> dict[int, str]:
    """Check MANIFEST.json against the pool IDs and every npz's bytes.

    Raises before any npz is parsed if IDs or hashes differ.
    """
    manifest_path = sealed / "MANIFEST.json"
    log(manifest_path, "read manifest")
    manifest = {
        int(k): str(v)
        for k, v in json.loads(manifest_path.read_text())["npz_sha256"].items()
    }
    if sorted(manifest) != sorted(ids):
        raise ValueError("MANIFEST IDs differ from the pool_i2 HELD-OUT IDs")
    bad = []
    for did in sorted(manifest):
        path = sealed / "npz" / f"{did}.npz"
        log(path, "sha256 of file bytes (not parsed)")
        if file_sha256(path) != manifest[did]:
            bad.append(did)
    if bad:
        raise ValueError(f"MANIFEST mismatch for HELD-OUT npz {bad}; nothing parsed")
    return manifest


class SealedLoader:
    """Picklable loader for manifest-verified sealed npz files.

    Every load is logged and re-checks the content hash against pool_i2.
    With ``normalise_labels`` (Study 13 Amendment 1 (A)), the label step of
    ``data.labels`` then runs on the checked table: raw label strings are read
    from the sealed raw OpenML cache, verified against the stored codes, and
    colliding spellings are merged. The mapping is logged on every load.
    """

    def __init__(
        self,
        sealed: Path,
        pool_hashes: Mapping[int, str],
        log_path: Path,
        normalise_labels: bool = False,
    ) -> None:
        """Bind the sealed folder, the pool hashes and the access log."""
        self.sealed = sealed
        self.pool_hashes = dict(pool_hashes)
        self.log_path = log_path
        self.normalise_labels = normalise_labels

    def label_record(self, data: Dataset) -> labels.LabelRecord:
        """Raw labels of a checked table, verified and mapped; logged."""
        log = AccessLog(self.log_path)
        raw = labels.raw_labels(self.sealed, data.did, log)
        record = labels.label_record(data.did, data.y, raw)
        log(self.sealed, "label normalisation", record.as_row())
        return record

    def load_with_record(self, did: int) -> tuple[Dataset, labels.LabelRecord | None]:
        """Checked table (labels normalised if enabled) and its label record."""
        path = self.sealed / "npz" / f"{did}.npz"
        AccessLog(self.log_path)(path, "load npz")
        with np.load(path, allow_pickle=False) as z:
            data = Dataset(
                did,
                str(z["name"]),
                int(z["version"]),
                np.asarray(z["X"], dtype=np.float64),
                np.asarray(z["y"], dtype=np.int64),
                str(z["sha"]),
            )
        check_content(data, self.pool_hashes)
        if not self.normalise_labels:
            return data, None
        record = self.label_record(data)
        return replace(data, y=labels.apply(data.y, record)), record

    def __call__(self, did: int, cache_dir: Path) -> Dataset:  # noqa: ARG002
        """Load table ``did`` (``cache_dir`` is unused: Loader signature)."""
        return self.load_with_record(did)[0]


def sealed_loader(
    sealed: Path,
    pool_hashes: Mapping[int, str],
    log_path: Path,
    normalise_labels: bool = False,
) -> Loader:
    """Loader for verified sealed npz files that re-checks the content hash."""
    return SealedLoader(sealed, pool_hashes, log_path, normalise_labels)


def label_log(ids: list[int], loader: Loader, out_dir: Path) -> Path | None:
    """Write ``label_normalisation.csv``: one row per table, before any draw.

    Only for a :class:`SealedLoader` with ``normalise_labels`` (otherwise
    None). The loads are logged like any other; an existing file is kept.
    """
    if not isinstance(loader, SealedLoader) or not loader.normalise_labels:
        return None
    path = out_dir / "label_normalisation.csv"
    if path.exists():
        return path
    rows = []
    for did in ids:
        record = loader.load_with_record(did)[1]
        assert record is not None  # normalise_labels is on
        rows.append(dict(key=dataset_key(did), **record.as_row()))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def check_content(data: Dataset, pool_hashes: Mapping[int, str]) -> None:
    """Raise unless the stored and recomputed content hash equal the pool's."""
    want = pool_hashes.get(data.did)
    got = content_hash(data.X, data.y)
    if want is None or data.sha256 != want or got != want:
        raise ValueError(f"HELD-OUT {data.did}: content hash != pool_i2 entry")


# ---------------------------------------------------------------------- screen
def screen_table(data: Dataset) -> dict[str, Any]:
    """Study 8 A2.7 screen in stored order and the two sensitivity flags.

    The screen excludes nothing and drops no column: every table stays in the
    primary analysis (owner decision 2026-10-27). ``flag_id`` marks an
    identifier/counter column (leaky ID-like, or ID-like and monotone in the
    stored row index): SENS-ID. ``flagged`` marks any screen finding (an A2.7
    exclusion or column drop, an identifier/counter column, or any conflicting
    duplicate group): SENS-INT.
    """
    result = screen(data.X, data.y).as_row()
    leaky = [int(c) for c in str(result["drop_columns"]).split(";") if c]
    id_like = [int(c) for c in str(result["id_like"]).split(";") if c]
    counters = [c for c in id_like if _monotone(data.X[:, c])]
    id_columns = sorted({*leaky, *counters})
    counts = np.bincount(data.y)
    flag_id = bool(id_columns)
    flag_conflict = int(result["conflict_groups"]) > 0
    return dict(
        key=dataset_key(data.did),
        did=data.did,
        n=len(data.y),
        C=len(counts),
        class_counts=";".join(map(str, counts)),
        id_columns=";".join(map(str, id_columns)),
        flag_id=flag_id,
        flag_conflict=flag_conflict,
        flagged=flag_id or flag_conflict or result["decision"] != "keep",
        **result,
    )


# ---------------------------------------------------------------------- seeds
def _seed(*entropy: int) -> int:
    state = np.random.SeedSequence(list(entropy))
    return int(state.generate_state(1)[0] % (2**31 - 1))


def cell_seed(
    base_seed: int, did: int, mult: int, rep: int, name: str, arm: int
) -> int:
    """Seed of one selection and all its fits (namespace 34)."""
    return _seed(
        base_seed,
        seed_did(did),
        mult,
        rep,
        study4.ORDER.index(name),
        SEED_NAMESPACE,
        arm,
    )


def redraw_seed(base_seed: int, did: int, mult: int, rep: int, attempt: int) -> int:
    """Seed of redraw attempt >= 1 of a random rep (arm T)."""
    return _seed(
        base_seed,
        seed_did(did),
        mult,
        rep,
        attempt,
        study4.ORDER.index("random"),
        SEED_NAMESPACE,
        REDRAW_TAG,
    )


def row_permutation(n: int, did: int, base_seed: int) -> IntArray:
    """Fixed per-table permutation (namespace 35)."""
    rng = np.random.default_rng([base_seed, seed_did(did), PERM_NAMESPACE])
    return rng.permutation(n).astype(np.int64)


# ---------------------------------------------------------------------- arm F
def test_counts(counts: IntArray) -> IntArray:
    """t_c = max(1, min(floor(0.3 n_c), n_c - 10)) (prereg Sec. 2.4)."""
    raw = np.minimum(np.floor(TEST_SHARE * counts), counts - MAX_MULT)
    return np.maximum(1, raw).astype(np.int64)


def fixed_split(y: IntArray, did: int, base_seed: int) -> IntArray:
    """Sorted positions of the fixed test set T_d (namespace 36)."""
    counts = np.bincount(y)
    if counts.min() < MAX_MULT + 1:
        raise ValueError(f"did={did}: a class has < {MAX_MULT + 1} rows")
    rng = np.random.default_rng([base_seed, seed_did(did), SPLIT_NAMESPACE])
    picks = [
        rng.choice(np.flatnonzero(y == c), size=t, replace=False)
        for c, t in enumerate(test_counts(counts))
    ]
    return np.sort(np.concatenate(picks)).astype(np.int64)


def draw_in_pool(
    name: str, y: IntArray, pool: IntArray, mult: int, seed: int
) -> IntArray:
    """Stratified (m per class) or random (m*C) rows of ``pool``, sorted."""
    rng = np.random.default_rng(seed)
    yp = y[pool]
    if name == "stratified":
        picks = [
            rng.choice(np.flatnonzero(yp == c), size=mult, replace=False)
            for c in np.unique(yp)
        ]
        local = np.concatenate(picks)
    else:
        local = rng.choice(len(pool), size=mult * (int(y.max()) + 1), replace=False)
    return np.sort(pool[local]).astype(np.int64)


# ---------------------------------------------------------------------- arm T
def covers(y: IntArray, labeled: IntArray) -> bool:
    """Whether the labeled rows contain every class of y."""
    return len(np.unique(y[labeled])) == len(np.unique(y))


def redraw(
    y: IntArray,
    mult: int,
    rep: int,
    first: IntArray,
    did: int,
    base_seed: int,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[IntArray | None, int, int]:
    """(rows, attempt, seed) of a covered draw; rows None if infeasible."""
    if covers(y, first):
        return first, 0, cell_seed(base_seed, did, mult, rep, "random", ARM_T)
    budget = mult * (int(y.max()) + 1)
    for attempt in range(1, max_attempts + 1):
        seed = redraw_seed(base_seed, did, mult, rep, attempt)
        rows = np.sort(
            np.random.default_rng(seed).choice(len(y), size=budget, replace=False)
        ).astype(np.int64)
        if covers(y, rows):
            return rows, attempt, seed
    return None, max_attempts, -1


def _selection(
    arm: str,
    cell: tuple[str, int, int],
    seed: int,
    rows: IntArray | None,
    attempt: int = 0,
) -> dict[str, Any]:
    name, mult, rep = cell
    return dict(
        arm=arm,
        strategy=name,
        mult=mult,
        rep=rep,
        attempt=attempt,
        seed=seed,
        labeled_pos=[] if rows is None else rows.tolist(),
        infeasible=rows is None,
    )


def _cell_selections(
    X: FloatArray,
    y: IntArray,
    pool: IntArray,
    did: int,
    base_seed: int,
    cell: tuple[str, int, int],
) -> list[dict[str, Any]]:
    """Arm T draw (+ its redraw for random) and arm F draw of one cell."""
    name, mult, rep = cell
    seed = cell_seed(base_seed, did, mult, rep, name, ARM_T)
    picked = np.sort(study4.select(name, X, y, mult, seed)).astype(np.int64)
    out = [_selection("T", cell, seed, picked)]
    if name == "random":
        lab, attempt, rseed = redraw(y, mult, rep, picked, did, base_seed)
        out.append(_selection("T", ("redraw", mult, rep), rseed, lab, attempt))
    fseed = cell_seed(base_seed, did, mult, rep, name, ARM_F)
    out.append(_selection("F", cell, fseed, draw_in_pool(name, y, pool, mult, fseed)))
    return out


def selections(
    key: str, did: int, X: FloatArray, y: IntArray, base_seed: int, out_dir: Path
) -> pd.DataFrame:
    """Arm T (with redraw) and arm F selections in permuted positions, cached."""
    path = out_dir / "selections" / f"{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    test = fixed_split(y, did, base_seed)
    pool = np.setdiff1d(np.arange(len(y), dtype=np.int64), test)
    rows = [
        row
        for mult in MULTS
        for name in STRATEGIES
        for rep in range(REPS[name])
        for row in _cell_selections(X, y, pool, did, base_seed, (name, mult, rep))
    ]
    frame = pd.DataFrame(rows)
    _atomic_write(frame, path)
    _atomic_write(pd.DataFrame(dict(test_pos=test)), _test_path(out_dir, key))
    return frame


def _test_path(out_dir: Path, key: str) -> Path:
    return out_dir / "selections" / f"{key}_test.parquet"


def test_positions(out_dir: Path, key: str) -> IntArray:
    """Stored fixed-split positions (permuted order)."""
    frame = pd.read_parquet(_test_path(out_dir, key))
    return frame["test_pos"].to_numpy(dtype=np.int64)


# ---------------------------------------------------------------------- learners
def zscore_all(X: FloatArray) -> FloatArray:
    """Label-free z-score with all rows' mean and SD (constant columns: SD 1)."""
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def _lda_proba(Xf: FloatArray, yf: IntArray, Xq: FloatArray) -> FloatArray:
    from sklearn.covariance import ledoit_wolf

    k = int(yf.max()) + 1
    means = np.array([Xf[yf == c].mean(axis=0) for c in range(k)])
    resid = Xf - means[yf]
    d = Xf.shape[1]
    if np.allclose(resid, 0):
        cov = np.eye(d)
    else:
        cov = ledoit_wolf(resid, assume_centered=True)[0] + 1e-6 * np.eye(d)
    W = means @ np.linalg.pinv(cov)
    prior = np.bincount(yf, minlength=k) / len(yf)
    s = Xq @ W.T - 0.5 * np.einsum("kd,kd->k", W, means) + np.log(prior)
    p = np.exp(s - s.max(axis=1, keepdims=True))
    return cast(FloatArray, p / p.sum(axis=1, keepdims=True))


def classical(name: str) -> Predictor:
    """E1 classical learner (prereg Sec. 2.3) as a ``predict_on`` predictor."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier

    if name not in CLASSICAL:
        raise ValueError(name)

    def predict(Xf: FloatArray, yf: IntArray, Xq: FloatArray, seed: int) -> FloatArray:
        if name == "lda":
            return _lda_proba(Xf, yf, Xq)
        model: Any
        if name == "lr":
            model = LogisticRegression(C=1.0, max_iter=5000)
        elif name == "knn1":
            model = KNeighborsClassifier(n_neighbors=1)
        elif name == "knn3":
            model = KNeighborsClassifier(n_neighbors=min(3, len(yf)))
        elif name == "gnb":
            model = GaussianNB(var_smoothing=0.1)
        else:
            model = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=3,
                min_samples_leaf=1,
                learning_rate=0.1,
                early_stopping=False,
                random_state=int(seed) % (2**31 - 1),
            )
        model.fit(Xf, yf)
        proba = np.zeros((len(Xq), int(yf.max()) + 1))
        proba[:, model.classes_] = model.predict_proba(Xq)
        return proba

    return predict


# ---------------------------------------------------------------------- fitting
def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def fit_record(
    predict: Predictor,
    X: FloatArray,
    y: IntArray,
    labeled: IntArray,
    query: IntArray,
    seed: int,
    perm: IntArray,
) -> dict[str, Any]:
    """Fit on ``labeled``, predict ``query``; probabilities, metrics, pc_* metrics."""
    n_classes = int(y.max()) + 1
    start = time.perf_counter()
    proba = predict_on(predict, X, labeled, y, query, n_classes, seed)
    seconds = time.perf_counter() - start
    seen = np.unique(y[labeled]).astype(np.int64)
    balanced = prior_correct(proba, y[labeled])
    corrected = {
        **panel_metrics(y[query], balanced, seen),
        **e17_panel(y[query], balanced),
    }
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
        **e17_panel(y[query], proba),
        **{f"pc_{k}": v for k, v in corrected.items()},
    )


def _nan_record(y: IntArray) -> dict[str, Any]:
    return dict(
        n_classes=int(y.max()) + 1,
        budget=0,
        labeled=[],
        query=[],
        proba=np.zeros(0, dtype=np.float32),
        coverage=float("nan"),
        seen=[],
        seconds=0.0,
        macro_f1=float("nan"),
        accuracy=float("nan"),
    )


def _cells(chosen: pd.DataFrame) -> Iterator[tuple[str, dict[str, Any]]]:
    for sel in cast(list[dict[str, Any]], chosen.to_dict("records")):
        yield f"{sel['arm']}|{sel['strategy']}|{sel['mult']}|{sel['rep']}", sel


def output_path(out_dir: Path, key: str, group: str) -> Path:
    """``results/study13/{set}/{key}_{group}.parquet``."""
    return out_dir / f"{key}_{group}.parquet"


class ExternalFM:
    """Predictor that runs a secondary FM in its own venv (Amendment 1 (B)).

    The worker (``runner/fm_extra_worker.py``) starts on the first call,
    verifies its pins before answering, and serves every fit of the job; the
    main ``.venv`` never imports the model packages. Picklable (the process
    is not pickled).
    """

    def __init__(self, name: str, python: Path, worker: Path = WORKER) -> None:
        """Bind the learner name, the venv's python and the worker script."""
        self.name = name
        self.python = python
        self.worker = worker
        self.pins: dict[str, Any] = {}
        self._proc: subprocess.Popen[bytes] | None = None

    def __getstate__(self) -> dict[str, Any]:
        """State without the running process."""
        return {**self.__dict__, "_proc": None}

    def _start(self) -> subprocess.Popen[bytes]:
        if self._proc is not None:
            return self._proc
        env = dict(os.environ, OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1")
        proc = subprocess.Popen(
            [str(self.python), "-I", str(self.worker), self.name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
        )
        assert proc.stdout is not None
        hello = fmw.read_frame(proc.stdout)
        if hello is None or "hello" not in hello:
            proc.kill()
            reason = "exited" if hello is None else str(hello["error"])
            raise RuntimeError(f"{self.name} worker failed to start: {reason}")
        self.pins = json.loads(str(hello["hello"]))
        self._proc = proc
        return proc

    def __call__(
        self, X_fit: FloatArray, y_fit: IntArray, X_query: FloatArray, seed: int
    ) -> FloatArray:
        """One fit and prediction in the worker."""
        proc = self._start()
        assert proc.stdin is not None and proc.stdout is not None
        fmw.write_frame(
            proc.stdin,
            X_fit=np.asarray(X_fit, dtype=np.float64),
            y_fit=np.asarray(y_fit, dtype=np.int64),
            X_query=np.asarray(X_query, dtype=np.float64),
            seed=np.int64(seed),
        )
        reply = fmw.read_frame(proc.stdout)
        if reply is None:
            self._proc = None
            raise RuntimeError(f"{self.name} worker exited")
        if "error" in reply:
            raise RuntimeError(f"{self.name} worker: {reply['error']}")
        return np.asarray(reply["proba"], dtype=np.float64)

    def close(self) -> None:
        """End the worker (zero-length frame), killing it if it hangs."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            assert proc.stdin is not None
            proc.stdin.write(fmw.HEADER.pack(0))
            proc.stdin.close()
            proc.wait(timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()


def external_fm(name: str, root: Path = ROOT) -> ExternalFM:
    """fm6, fm7 or fm8 in its pinned venv under ``root``."""
    return ExternalFM(name, root / EXTRA_VENVS[name] / "bin" / "python")


def _predictors(group: str, predict: Predictor | None) -> dict[str, Predictor]:
    if group == "classical":
        return {name: classical(name) for name in CLASSICAL}
    if group in EXTRA_FMS:
        return {group: predict or external_fm(group)}
    return {group: predict or study4._predictor(group)}


def _close(predictors: Mapping[str, Predictor]) -> None:
    for fn in predictors.values():
        if isinstance(fn, ExternalFM):
            fn.close()


def job_selections(chosen: pd.DataFrame, group: str) -> pd.DataFrame:
    """Every selection, or arm T stratified/random only for fm6, fm7 and fm8."""
    if group not in EXTRA_FMS:
        return chosen
    return chosen[(chosen.arm == "T") & chosen.strategy.isin(STRATEGIES)]


def skip_job(group: str, n_classes: int) -> bool:
    """Pre-declared skips: fm7 and fm8 on tables with more than 10 classes.

    fm7 is Mitra-v2 (Amendment 1); fm8 is LimiX-16M (Amendment 3).
    """
    if group == "fm8":
        return n_classes > LIMIX_MAX_CLASSES
    return group == "fm7" and n_classes > MITRA_MAX_CLASSES


def _meta(sel: Mapping[str, Any]) -> dict[str, Any]:
    return dict(
        arm=sel["arm"],
        strategy=sel["strategy"],
        mult=int(sel["mult"]),
        rep=int(sel["rep"]),
        attempt=int(sel["attempt"]),
        seed=int(sel["seed"]),
        infeasible=bool(sel["infeasible"]),
    )


def _fit_cell(
    sel: Mapping[str, Any],
    fn: Predictor,
    ctx: Mapping[str, Any],
    base: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """(meta, record, reused) of one selection and learner.

    ``base`` is this learner's arm-T random fit of the same (m, rep), reused
    unchanged by a redraw whose attempt 0 already covered every class.
    """
    meta = _meta(sel)
    reuse = meta["strategy"] == "redraw" and meta["attempt"] == 0
    if meta["infeasible"]:
        return meta, _nan_record(ctx["y"]), False
    if reuse and base is not None:
        skip = {*meta, "_key", "reused", "learner"}
        return meta, {k: v for k, v in base.items() if k not in skip}, True
    labeled = np.asarray(sel["labeled_pos"], dtype=np.int64)
    everything = np.arange(len(ctx["y"]), dtype=np.int64)
    query = ctx["test"] if meta["arm"] == "F" else np.setdiff1d(everything, labeled)
    record = fit_record(
        fn, ctx["X"], ctx["y"], labeled, query, meta["seed"], ctx["perm"]
    )
    return meta, record, False


def _resume(partial: Path) -> list[dict[str, Any]]:
    if not partial.exists():
        return []
    return cast(list[dict[str, Any]], pd.read_parquet(partial).to_dict("records"))


def _random_key(row: Mapping[str, Any]) -> tuple[str, int, int] | None:
    if row["arm"] == "T" and row["strategy"] == "random":
        return str(row["learner"]), int(row["mult"]), int(row["rep"])
    return None


def run_job(
    did: int,
    group: str,
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    *,
    loader: Loader = load,
    predict: Predictor | None = None,
) -> Path:
    """Every arm x strategy x mult x rep for one table and learner group."""
    key = dataset_key(did)
    out = output_path(out_dir, key, group)
    if out.exists():
        return out
    data = loader(did, cache_dir)
    perm = row_permutation(len(data.y), did, base_seed)
    X, y = data.X[perm], data.y[perm]
    chosen = selections(key, did, X, y, base_seed, out_dir)
    ctx = dict(
        X=zscore_all(X) if group == "classical" else X,
        y=y,
        perm=perm,
        test=test_positions(out_dir, key),
    )
    partial = out.with_name(out.name.replace(".parquet", ".partial.parquet"))
    rows = _resume(partial)
    done = {(r["learner"], r["_key"]) for r in rows}
    fits = {k: r for r in rows if (k := _random_key(r)) is not None}
    predictors = _predictors(group, predict)
    todo = [
        (cell, sel, name, fn)
        for cell, sel in _cells(job_selections(chosen, group))
        for name, fn in predictors.items()
        if (name, cell) not in done
    ]
    try:
        for count, (cell, sel, name, fn) in enumerate(todo, start=1):
            base = fits.get((name, int(sel["mult"]), int(sel["rep"])))
            meta, record, reused = _fit_cell(sel, fn, ctx, base)
            row = dict(meta, learner=name, reused=reused, _key=cell, **record)
            rows.append(row)
            if (k := _random_key(row)) is not None:
                fits[k] = row
            if count % CHECKPOINT_EVERY == 0:
                _atomic_write(pd.DataFrame(rows), partial)
    finally:
        _close(predictors)
    frame = pd.DataFrame(rows).drop(columns="_key")
    frame.insert(0, "dataset", key)
    _atomic_write(frame, out)
    partial.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------------- driver
def _init_worker() -> None:  # pragma: no cover - runs in worker processes
    import torch

    torch.set_num_threads(TORCH_THREADS)


def screen_all(
    ids: list[int], loader: Loader, cache_dir: Path, out_dir: Path
) -> dict[int, dict[str, Any]]:
    """Screen every table before any fit; write ``integrity_screen.csv``."""
    path = out_dir / "integrity_screen.csv"
    if path.exists():
        frame = pd.read_csv(path, keep_default_na=False, dtype=str)
        records = cast(list[dict[str, Any]], frame.to_dict("records"))
        return {int(r["did"]): r for r in records}
    rows = {did: screen_table(loader(did, cache_dir)) for did in ids}
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows.values())).to_csv(path, index=False)
    return rows


def _write_provenance(
    out_dir: Path,
    base_seed: int,
    ids: list[int],
    groups: list[str],
    set_name: str,
    prereg: Path | None,
    root: Path,
) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()
    here = Path(__file__).resolve()
    analysis = here.parents[1] / "analysis" / "study13.py"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "provenance.json").write_text(
        json.dumps(
            dict(
                set=set_name,
                prereg=str(prereg) if prereg else None,
                prereg_sha256=file_sha256(prereg) if prereg else None,
                runner_sha256=file_sha256(here),
                analysis_sha256=file_sha256(analysis) if analysis.exists() else None,
                git_head=head,
                base_seed=base_seed,
                namespaces=dict(
                    seed=SEED_NAMESPACE, perm=PERM_NAMESPACE, split=SPLIT_NAMESPACE
                ),
                mults=MULTS,
                reps=REPS,
                test_share=TEST_SHARE,
                groups=groups,
                datasets=ids,
                started_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            ),
            indent=2,
        )
    )


def _write_amendment_provenance(
    out_dir: Path, groups: list[str], skipped: list[tuple[int, str]], loader: Loader
) -> None:
    """Amendment 1: label step on/off, secondary-FM pins and pre-declared skips."""
    normalise = isinstance(loader, SealedLoader) and loader.normalise_labels
    (out_dir / "provenance_amendment1.json").write_text(
        json.dumps(
            dict(
                label_normalisation=normalise,
                labels_sha256=file_sha256(Path(labels.__file__).resolve()),
                worker_sha256=file_sha256(WORKER),
                extra_fm_pins={g: fmw.PINS[g] for g in groups if g in EXTRA_FMS},
                skipped_jobs=[list(job) for job in skipped],
                mitra_max_classes=MITRA_MAX_CLASSES,
                limix_max_classes=LIMIX_MAX_CLASSES,
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
    groups: list[str],
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    workers: int,
    root: Path,
    *,
    loader: Loader = load,
    predict: Predictor | None = None,
) -> list[Path]:
    """Screen, then run every (group, table) job; failures retried once.

    For HELD-OUT, ``loader`` must be :func:`sealed_loader` and the guard has
    already passed (see :func:`main`); the guard is re-checked here.
    """
    if set_name not in SETS:
        raise ValueError(set_name)
    prereg = check_heldout_go(root) if set_name == "heldout" else None
    label_log(dataset_ids, loader, out_dir)  # Amendment 1 (A), before any draw
    screened = screen_all(dataset_ids, loader, cache_dir, out_dir)  # before any fit
    kept = list(dataset_ids)  # the screen only flags (prereg Sec. 1)
    skipped = [
        (did, g) for g in groups for did in kept if skip_job(g, int(screened[did]["C"]))
    ]
    _write_provenance(out_dir, base_seed, kept, groups, set_name, prereg, root)
    _write_amendment_provenance(out_dir, groups, skipped, loader)
    os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_THREADS))
    jobs = [(did, g) for g in groups for did in kept if (did, g) not in skipped]
    common: dict[str, Any] = dict(
        cache_dir=cache_dir,
        out_dir=out_dir,
        base_seed=base_seed,
        loader=loader,
        predict=predict,
    )
    done: list[Path] = []
    for attempt in (1, 2):
        finished, jobs = _attempt(jobs, attempt, workers, **common)
        done.extend(finished)
        if not jobs:
            break
    return done


def _attempt(
    jobs: list[tuple[int, str]],
    attempt: int,
    workers: int,
    *,
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    loader: Loader,
    predict: Predictor | None,
) -> tuple[list[Path], list[tuple[int, str]]]:
    """One pass over ``jobs``; returns (outputs, failed jobs)."""
    done: list[Path] = []
    failed: list[tuple[int, str]] = []
    if workers == 1:
        for job in jobs:
            try:
                done.append(
                    run_job(
                        *job,
                        cache_dir,
                        out_dir,
                        base_seed,
                        loader=loader,
                        predict=predict,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - logged; retried once
                _log_failure(out_dir, (*job, attempt), exc)
                failed.append(job)
        return done, failed
    with ProcessPoolExecutor(  # pragma: no cover - multiprocess path
        max_workers=workers, initializer=_init_worker
    ) as pool:
        futures = {
            pool.submit(
                run_job,
                *job,
                cache_dir,
                out_dir,
                base_seed,
                loader=loader,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):  # pragma: no cover
            try:
                done.append(future.result())
                print(f"done {futures[future]} ({len(done)}/{len(jobs)})", flush=True)
            except Exception as exc:  # noqa: BLE001 - logged; retried once
                _log_failure(out_dir, (*futures[future], attempt), exc)
                failed.append(futures[future])
    return done, failed


def main() -> None:  # pragma: no cover - command-line entry
    """``python -m simaudit.runner.study13 --set {dev,heldout} [--workers 4]``."""
    import argparse

    import yaml

    from simaudit.settings import PROJECT_ROOT, settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=SETS, default="dev")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_i2.yaml").read_text())
    entries = pool[args.set]
    ids = [int(d["did"]) for d in entries]
    ids = ids[: args.limit] if args.limit else ids
    out_dir = settings.results_dir / "study13" / args.set
    loader: Loader = load
    if args.set == "heldout":
        check_heldout_go(PROJECT_ROOT)  # before anything sealed is touched
        sealed = PROJECT_ROOT.parent / "sealed_heldout"
        log_path = out_dir / "access_log.jsonl"
        verify_manifest(sealed, [int(d["did"]) for d in entries], AccessLog(log_path))
        loader = sealed_loader(
            sealed,
            {int(d["did"]): str(d["sha256"]) for d in entries},
            log_path,
            normalise_labels=True,  # Amendment 1 (A)
        )
    run(
        args.set,
        ids,
        args.groups.split(","),
        settings.data_cache_dir,
        out_dir,
        settings.base_seed,
        args.workers,
        PROJECT_ROOT,
        loader=loader,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
