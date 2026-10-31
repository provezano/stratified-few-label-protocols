"""E16 step 3 (exploratory, NEW FITS): class merging within a dataset.

Takes the 10 T1' multiclass tables with C >= 4 and relabels them into binary and ternary
versions of the same rows and features (merge rules below), then fits TabICL (fm2) and
TabPFN v2 (fm1) under the random and stratified protocols at m in {1, 3}, re-using the
Study 9 runner read-only:
  * same row permutation, stop set, labelling pool and seeds (selection_seed, fm_seed);
  * random labelled sets are the same row prefixes as in the stored Study 9 runs (the random
    order does not depend on y), so only the labels change;
  * prediction via runner.study9.predict_on (L0) and metrics via analysis.study7.metric_set.
Deviation (cost): the evaluation panel is capped at its first 500 positions (Study 9: 2000);
the original-C comparison is recomputed from the stored Study 9 probabilities on the same 500
positions (e16_merge_analysis.py), so every version is scored on identical rows.

Merge rules (deterministic, from pool class counts; ties to the lower class code):
  bal_k: classes sorted by pool count (descending) go one by one to the group with the
         smallest current count -> k groups of near-equal mass;
  imb_k: the "rare" group takes the smallest classes (ascending) while their cumulative share
         stays <= 0.15 (at least the smallest class); the other classes are split by bal_{k-1}.
Versions: bin_bal, bin_imb, ter_bal, ter_imb.

Run from simaudit/ (background):
  OMP_NUM_THREADS=1 nice -n 15 .venv/bin/python paper/p1/exploratory/e16_multiclass/e16_merge_run.py --workers 2
Writes e16_multiclass/out/merge/{key}_{fm}.parquet (one job per table x FM, resumable).
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from simaudit.analysis.study7 import metric_set  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.runner.study9 import (  # noqa: E402
    _predictor, dataset_key, fm_seed, layout, nested_sets, predict_on, row_permutation, selection_seed)
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "merge"
PANEL = 500
MS = (1, 3)
REPS = {"random": 10, "stratified": 5}
FMS = ("fm2", "fm1")
VERSIONS = ("bin_bal", "bin_imb", "ter_bal", "ter_imb")
RARE_SHARE = 0.15


def bal_groups(classes: list[int], counts: np.ndarray, k: int) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(k)]
    mass = np.zeros(k)
    for c in sorted(classes, key=lambda c: (-counts[c], c)):
        g = int(np.argmin(mass))
        groups[g].append(c)
        mass[g] += counts[c]
    return groups


def merge_map(counts: np.ndarray, version: str) -> np.ndarray:
    """Old class code -> new class code (0..k-1, sorted by the group's smallest old code)."""
    C = len(counts)
    k = 2 if version.startswith("bin") else 3
    if version.endswith("bal"):
        groups = bal_groups(list(range(C)), counts, k)
    else:
        order = sorted(range(C), key=lambda c: (counts[c], c))
        rare, tot = [order[0]], counts[order[0]]
        for c in order[1:]:
            if (tot + counts[c]) / counts.sum() > RARE_SHARE or len(rare) >= C - (k - 1):
                break
            rare.append(c)
            tot += counts[c]
        groups = [rare] + bal_groups([c for c in range(C) if c not in rare], counts, k - 1)
    groups = sorted(groups, key=min)
    mp = np.empty(C, dtype=np.int64)
    for g, cl in enumerate(groups):
        mp[cl] = g
    return mp


def tables() -> list[tuple[str, int]]:
    rows = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())["datasets"]
    out = []
    for r in rows:
        did = int(r["did"])
        y = load(did, settings.data_cache_dir).y
        if int(y.max()) + 1 >= 4:
            out.append((dataset_key(did), did))
    return out


def run_job(key: str, did: int, fm: str) -> Path:
    path = OUT / f"{key}_{fm}.parquet"
    if path.exists():
        return path
    import torch

    torch.set_num_threads(1)
    data = load(did, settings.data_cache_dir)
    perm = row_permutation(len(data.y), did, settings.base_seed)
    X, y0 = data.X[perm], data.y[perm]
    s, e = layout(len(y0))
    pool = np.arange(s, len(y0), dtype=np.int64)
    panel = np.arange(min(e, PANEL), dtype=np.int64)
    counts = np.bincount(y0[pool], minlength=int(y0.max()) + 1)
    predict = _predictor(fm)
    rows = []
    for version in VERSIONS:
        mp = merge_map(counts, version)
        y = mp[y0]
        C = int(y.max()) + 1
        for name in ("random", "stratified"):
            for rep in range(REPS[name]):
                sets = nested_sets(name, X, y, pool, selection_seed(settings.base_seed, did, rep, name), None)
                for m in MS:
                    if m not in sets:
                        continue
                    lab = sets[m]
                    q = np.setdiff1d(panel, lab)
                    seed = fm_seed(settings.base_seed, did, m, rep, name)
                    t0 = time.perf_counter()
                    P = predict_on(predict, X, lab, y, q, C, seed)
                    seen = np.unique(y[lab]).astype(np.int64)
                    met = metric_set(y[q], P, np.arange(C, dtype=np.int64), seen)
                    pred = P.argmax(1)
                    rows.append(dict(dataset=key, did=did, fm=fm, version=version, merge_map=mp.tolist(), C=C,
                                     strategy=name, m=m, rep=rep, fm_seed=seed,
                                     N=np.bincount(y[lab], minlength=C).tolist(), labeled_pos=lab.tolist(),
                                     cm=np.bincount(y[q] * C + pred, minlength=C * C).tolist(),
                                     soft=P.mean(0).tolist(), seconds=time.perf_counter() - t0, **met))
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(k, d, fm) for fm in FMS for k, d in tables()]
    print(len(jobs), "jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fut = {ex.submit(run_job, *j): j for j in jobs}
        for i, f in enumerate(as_completed(fut), 1):
            try:
                f.result()
                print(f"done {fut[f]} {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {fut[f]}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
