"""EXPLORATORY Study 12-style replication on T1' for fm6 TabDPT / fm7 Mitra-v2 (arm T).

Study 7 Part (a), stratified and random only, with Study 12's exact design:

- the 52 tables of ``data/pool_t1prime.yaml``;
- the fixed row permutation of ``runner.study12.row_permutation`` (namespace 33);
- the stored Study 12 selections and seeds, read-only from
  ``results/study12/selections/{key}.parquet`` (positions in permuted order);
- the evaluation panel E = first min(n, 2000) permuted rows, metrics on E \\ L, via
  ``analysis.study12.panel_metrics`` (+ ``pc_*`` with ``analysis.study7.prior_correct``),
  i.e. ``runner.study12.fit_record`` re-implemented without importing the runner
  (it pulls in fm1-fm5's packages);
- missing classes as in ``runner.study9.predict_on`` (fit on seen classes, unseen = 0).

The redraw arm is not run. Mitra is seeded per fit with AutoGluon's ``set_seed`` (as
``run_s7a_v2.py`` and ``runner/fm_extra_worker.py``). Nothing is written outside
``paper/p1/exploratory/fm_extra/out/t1p/``.

Usage (from simaudit/, in the model's venv):
    OMP_NUM_THREADS=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 nice -n 10 \
      .venv_fm_extra/bin/python paper/p1/exploratory/fm_extra/run_t1p.py fm6 --workers 3

Resumable: per (table, fm) parquet, skipped when present; ``*.partial.parquet``
checkpoints every 10 fits. Clean stop: create ``out/t1p/STOP`` (or ``STOP_fm6``) and
every worker flushes its checkpoint and exits after the current fit. Memory guard:
before each fit a worker waits while available memory (free + inactive +
speculative pages) is below ``--min-free-gb`` (default 4), so the pool shrinks.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from simaudit.analysis.study7 import prior_correct  # noqa: E402
from simaudit.analysis.study12 import panel_metrics  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.selection.strategies import expand_proba  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

S12 = settings.results_dir / "study12"
OUT = HERE / "out" / "t1p"
PERM_NAMESPACE = 33  # runner.study12.PERM_NAMESPACE
PANEL_MAX = 2000  # runner.study12.PANEL_MAX
STRATEGIES = ("stratified", "random")
CHECKPOINT_EVERY = 10


# ------------------------------------------------- copies of runner.study9 / study12
def seed_did(did: int) -> int:
    return did if did >= 0 else 2**40 + did


def dataset_key(did: int) -> str:
    return f"openml_{did}" if did >= 0 else f"pmlb_{-did}"


def row_permutation(n: int, did: int, base_seed: int) -> np.ndarray:
    rng = np.random.default_rng([base_seed, seed_did(did), PERM_NAMESPACE])
    return rng.permutation(n).astype(np.int64)


def predict_on(predict, X, labeled, y, query, n_classes, seed):  # type: ignore[no-untyped-def]
    seen, y_fit = np.unique(y[labeled], return_inverse=True)
    if len(seen) == 1:
        return expand_proba(np.ones((len(query), 1)), seen, n_classes)
    proba = predict(X[labeled], y_fit.astype(np.int64), X[query], seed)
    return expand_proba(np.asarray(proba), seen, n_classes)


def fit_record(predict, X, y, labeled, seed, perm) -> dict:  # type: ignore[no-untyped-def]
    n_classes = int(y.max()) + 1
    query = np.setdiff1d(np.arange(min(len(y), PANEL_MAX), dtype=np.int64), labeled)
    start = time.perf_counter()
    proba = predict_on(predict, X, labeled, y, query, n_classes, seed)
    seconds = time.perf_counter() - start
    seen = np.unique(y[labeled]).astype(np.int64)
    corrected = panel_metrics(y[query], prior_correct(proba, y[labeled]), seen)
    return dict(
        n_classes=n_classes, budget=len(labeled), labeled=perm[labeled].tolist(),
        query=perm[query].tolist(), proba=proba.astype(np.float32).ravel(),
        coverage=len(seen) / n_classes, seen=seen.tolist(), seconds=seconds,
        **panel_metrics(y[query], proba, seen),
        **{f"pc_{k}": v for k, v in corrected.items()},
    )


# ------------------------------------------------- learners, guards
def predictor(fm: str):  # type: ignore[no-untyped-def]
    import adapters as ad

    if fm == "fm7":
        from run_s7a_v2 import seeded_mitra

        return seeded_mitra
    return ad.PREDICTORS[fm][1]


def available_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split(" ")[0])
    pages = {k.strip(): int(v.strip(" .")) for k, v in
             (line.split(":") for line in out.splitlines()[1:] if ":" in line)}
    free = pages["Pages free"] + pages["Pages inactive"] + pages["Pages speculative"]
    return free * page / 2**30


def stop_requested(fm: str) -> bool:
    return (OUT / "STOP").exists() or (OUT / f"STOP_{fm}").exists()


def t1prime() -> list[tuple[int, str]]:
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())
    return [(int(d["did"]), str(d["key"])) for d in pool["datasets"]]


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


# ------------------------------------------------- one job
def run_job(fm: str, did: int, min_free_gb: float) -> str:
    import torch

    torch.set_num_threads(1)
    key = dataset_key(did)
    out = OUT / f"{key}_{fm}.parquet"
    if out.exists():
        return f"skip {key}"
    partial = out.with_name(out.name.replace(".parquet", ".partial.parquet"))
    rows: list[dict] = []
    if partial.exists():
        rows = pd.read_parquet(partial).to_dict("records")
    done = {r["_key"] for r in rows}
    data = load(did, settings.data_cache_dir)
    perm = row_permutation(len(data.y), did, settings.base_seed)
    X, y = data.X[perm], data.y[perm]
    chosen = pd.read_parquet(S12 / "selections" / f"{key}.parquet")
    chosen = chosen[chosen.strategy.isin(STRATEGIES)]
    predict = predictor(fm)
    fresh = 0
    for sel in chosen.to_dict("records"):
        cell = f"{sel['strategy']}|{sel['mult']}|{sel['rep']}"
        if cell in done:
            continue
        while available_gb() < min_free_gb and not stop_requested(fm):
            time.sleep(30)
        if stop_requested(fm):
            if rows:
                _write(pd.DataFrame(rows), partial)
            return f"stopped {key} ({len(rows)} cells kept)"
        labeled = np.asarray(sel["labeled_pos"], dtype=np.int64)
        meta = dict(strategy=sel["strategy"], mult=int(sel["mult"]), rep=int(sel["rep"]),
                    attempt=int(sel["attempt"]), seed=int(sel["seed"]))
        rows.append(dict(meta, _key=cell, **fit_record(predict, X, y, labeled, meta["seed"], perm)))
        fresh += 1
        if fresh % CHECKPOINT_EVERY == 0:
            _write(pd.DataFrame(rows), partial)
    frame = pd.DataFrame(rows).drop(columns="_key")
    frame.insert(0, "learner", fm)
    frame.insert(0, "dataset", key)
    _write(frame, out)
    partial.unlink(missing_ok=True)
    return f"done {key}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fm", choices=("fm6", "fm7"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--datasets", default="", help="comma-separated keys (default: all 52)")
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    a = ap.parse_args()
    tables = t1prime()
    if a.datasets:
        wanted = set(a.datasets.split(","))
        tables = [t for t in tables if dataset_key(t[0]) in wanted]
    # Largest panels first (better packing); n from the cached npz header.
    sizes = {did: len(load(did, settings.data_cache_dir).y) for did, _ in tables}
    tables.sort(key=lambda t: -min(sizes[t[0]], PANEL_MAX) * sizes[t[0]])
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(run_job, a.fm, did, a.min_free_gb): did for did, _ in tables}
        for future in as_completed(futures):
            did = futures[future]
            try:
                print(f"{a.fm} {future.result()} [{time.strftime('%H:%M:%S')}]", flush=True)
            except Exception as exc:  # noqa: BLE001 - logged; other jobs go on
                print(f"FAILED {a.fm} {dataset_key(did)}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
