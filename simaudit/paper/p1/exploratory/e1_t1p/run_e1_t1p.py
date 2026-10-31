"""EXPLORATORY: E1 classical learners on T1' with Study 12's exact draws (PLAN.md).

Per T1' table: Study 12 row permutation (namespace 33), stored Study 12 selections
(stratified 5 reps, random 20 reps, m in {1,3,5,10}; redraw not run), evaluation rows
E = first min(n, 2000) permuted rows, metrics on E \\ L with ``panel_metrics``; missing
classes as ``runner.study9.predict_on``. Learners = ``e1_classical.predictor`` (DEV E1,
unchanged), features z-scored with whole-table mean/sd. Each cell's labelled rows are
checked against the stored Study 12 fm2 run.

Usage (from simaudit/):
    nice -n 15 env OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python \
        paper/p1/exploratory/e1_t1p/run_e1_t1p.py --workers 3
Writes out/runs/{key}.parquet (skipped when present) and out/DONE at the end.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from e1_classical import LEARNERS, predictor  # noqa: E402
from simaudit.analysis.study7 import prior_correct  # noqa: E402
from simaudit.analysis.study12 import panel_metrics  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.selection.strategies import expand_proba  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

S12 = settings.results_dir / "study12"
OUT = HERE / "out"
RUNS = OUT / "runs"
PERM_NAMESPACE = 33  # runner.study12.PERM_NAMESPACE
PANEL_MAX = 2000  # runner.study12.PANEL_MAX


def seed_did(did: int) -> int:  # runner.study9.seed_did
    return did if did >= 0 else 2**40 + did


def dataset_key(did: int) -> str:  # runner.study9.dataset_key
    return f"openml_{did}" if did >= 0 else f"pmlb_{-did}"


def row_permutation(n: int, did: int, base_seed: int) -> np.ndarray:  # runner.study12
    rng = np.random.default_rng([base_seed, seed_did(did), PERM_NAMESPACE])
    return rng.permutation(n).astype(np.int64)


def predict_on(predict, X, labeled, y, query, n_classes, seed):  # runner.study9.predict_on
    seen, y_fit = np.unique(y[labeled], return_inverse=True)
    if len(seen) == 1:
        return expand_proba(np.ones((len(query), 1)), seen, n_classes)
    proba = predict(X[labeled], y_fit.astype(np.int64), X[query], seed)
    return expand_proba(np.asarray(proba), seen, n_classes)


def t1prime() -> list[int]:
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())
    return [int(d["did"]) for d in pool["datasets"]]


def run_table(did: int) -> str:
    key = dataset_key(did)
    out = RUNS / f"{key}.parquet"
    if out.exists():
        return f"skip {key}"
    t0 = time.perf_counter()
    data = load(did, settings.data_cache_dir)
    perm = row_permutation(len(data.y), did, settings.base_seed)
    X, y = data.X[perm], data.y[perm]
    sd = X.std(axis=0)
    Xs = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    C = int(y.max()) + 1
    sel = pd.read_parquet(S12 / "selections" / f"{key}.parquet")
    sel = sel[sel.strategy.isin(["stratified", "random"])]
    ref = pd.read_parquet(S12 / f"{key}_fm2.parquet", columns=["strategy", "mult", "rep", "seed", "labeled"])
    ref = {(r.strategy, int(r.mult), int(r.rep)): (int(r.seed), list(r.labeled)) for r in ref.itertuples()}
    panel = np.arange(min(len(y), PANEL_MAX), dtype=np.int64)
    rows = []
    for r in sel.itertuples():
        lab = np.asarray(r.labeled_pos, dtype=np.int64)
        seed = int(r.seed)
        s_ref, l_ref = ref[(r.strategy, int(r.mult), int(r.rep))]
        if s_ref != seed or perm[lab].tolist() != [int(v) for v in l_ref]:
            raise AssertionError(f"{key} {r.strategy} m={r.mult} rep={r.rep}: selection != Study 12 fm2")
        query = np.setdiff1d(panel, lab)
        seen = np.unique(y[lab]).astype(np.int64)
        for name in LEARNERS:
            t = time.perf_counter()
            proba = predict_on(predictor(name), Xs, lab, y, query, C, seed)
            secs = time.perf_counter() - t
            met = panel_metrics(y[query], proba, seen)
            pc = panel_metrics(y[query], prior_correct(proba, y[lab]), seen)
            rows.append(dict(
                dataset=key, learner=name, strategy=r.strategy, mult=int(r.mult), rep=int(r.rep),
                seed=seed, n_classes=C, budget=len(lab), coverage=len(seen) / C, seconds=secs,
                macro_f1=met["macro_f1"], accuracy=met["accuracy"],
                balanced_accuracy=met["balanced_accuracy"], auc_all=met["auc_all"],
                pc_macro_f1=pc["macro_f1"], pc_accuracy=pc["accuracy"],
                pc_balanced_accuracy=pc["balanced_accuracy"],
            ))
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(out)
    return f"done {key} ({len(rows)} fits, {time.perf_counter() - t0:.0f} s)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    ids = t1prime()
    ids = ids[: a.limit] if a.limit else ids
    (OUT / "DONE").unlink(missing_ok=True)
    failed = []
    with ProcessPoolExecutor(min(a.workers, 3)) as pool:
        futs = {pool.submit(run_table, did): did for did in ids}
        for fu in as_completed(futs):
            try:
                print(fu.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                failed.append(futs[fu])
                print(f"FAILED {futs[fu]}: {exc!r}", flush=True)
    status = f"failed={failed}" if failed else "ok"
    if not a.limit:
        (OUT / "DONE").write_text(f"{status} n={len(ids)}\n")
    print(status, flush=True)


if __name__ == "__main__":
    main()
