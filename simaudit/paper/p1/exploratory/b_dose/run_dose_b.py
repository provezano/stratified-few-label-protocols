"""Runner of the exploratory dose-response arms F, S and N (DEV only).

EXPLORATORY, not pre-registered. Plan (committed before any fit):
paper/p1/exploratory/b_dose_response_plan.md.

One job = one (FM, DEV table); its parquet is written atomically to
out/raw/{fm}_{did}.parquet and skipped when present. Within a job, finished
(mult, rep) groups are checkpointed to out/raw/{fm}_{did}.partial.parquet.

Usage (from simaudit/):
    nice -n 10 .venv/bin/python paper/p1/exploratory/b_dose/run_dose_b.py --workers 3
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import datetime  # noqa: E402
from itertools import product  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import design  # noqa: E402

FMS = ("fm2", "fm1", "fm5", "fm3")  # plan Sec. 4: cheapest first
MULTS = (3, 5, 10)
REPS = 5
OUT = HERE / "out"
RAW = OUT / "raw"
METRICS = ("macro_f1", "accuracy", "balanced_accuracy")


def _metrics(y, proba, rows, prefix: str) -> dict[str, float]:
    from simaudit.analysis.study7 import metric_set

    n_classes = proba.shape[1]
    if len(rows) == 0 or len(np.unique(y[rows])) < n_classes:
        return {f"{prefix}{m}": float("nan") for m in METRICS}
    cols = np.arange(n_classes, dtype=np.int64)
    out = metric_set(y[rows], proba[rows], cols, cols)
    return {f"{prefix}{m}": out[m] for m in METRICS}


def _score(y, proba, labeled, common) -> dict[str, float]:
    from simaudit.analysis.study7 import prior_correct

    own = np.setdiff1d(np.arange(len(y)), labeled)
    corrected = prior_correct(proba, y[labeled])
    rec: dict[str, float] = {}
    rec.update(_metrics(y, proba, own, ""))
    rec.update(_metrics(y, corrected, own, "pc_"))
    rec.update(_metrics(y, proba, common, "cm_"))
    rec.update(_metrics(y, corrected, common, "cm_pc_"))
    return rec


def _group(fm: str, did: int, X, y, mult: int, rep: int, base_seed: int, predict) -> list[dict[str, Any]]:
    from simaudit.selection import strategies as sel

    cells = design.rep_cells(y, did, mult, rep, base_seed)
    feasible = [c for c in cells if c.feasible]
    union = np.unique(np.concatenate([np.array(c.labeled) for c in feasible])) if feasible else np.array([], int)
    common = np.setdiff1d(np.arange(len(y)), union)
    seed = design.fm_seed(base_seed, did, mult, rep)
    n_classes = int(y.max()) + 1
    fitted: dict[tuple[int, ...], tuple[dict[str, float], float]] = {}
    rows = []
    for c in cells:
        rec: dict[str, Any] = dict(
            learner=fm, dataset=did, arm=c.arm, side=c.side, level=c.level, mult=mult, rep=rep,
            upweighted=c.upweighted, counts=list(c.counts), budget=sum(c.counts),
            share_real=c.share_real, feasible=c.feasible, n_classes=n_classes,
            n_common=len(common), fm_seed=seed,
        )
        if c.feasible:
            if c.labeled not in fitted:
                labeled = np.array(c.labeled, dtype=np.int64)
                t0 = time.perf_counter()
                proba = sel.predict_with_budget(predict, X, labeled, y[labeled], n_classes, seed)
                secs = time.perf_counter() - t0
                fitted[c.labeled] = (_score(y, proba, labeled, common), secs)
                rec["fitted_here"] = True
            else:
                rec["fitted_here"] = False
            scores, secs = fitted[c.labeled]
            rec.update(scores, seconds=secs, context_hash=hash(c.labeled) & 0xFFFFFFFF)
        rows.append(rec)
    return rows


def run_job(fm: str, did: int, base_seed: int) -> str:
    import torch

    torch.set_num_threads(1)
    from simaudit.data.openml_loader import load
    from simaudit.runner.study7 import predictor
    from simaudit.settings import settings

    out = RAW / f"{fm}_{did}.parquet"
    if out.exists():
        return f"skip {fm} {did}"
    partial = RAW / f"{fm}_{did}.partial.parquet"
    rows: list[dict[str, Any]] = []
    done: set[tuple[int, int]] = set()
    if partial.exists():
        old = pd.read_parquet(partial)
        rows = old.to_dict("records")
        done = set(zip(old["mult"].astype(int), old["rep"].astype(int)))
    data = load(did, settings.data_cache_dir)
    predict = predictor(fm)
    t0 = time.perf_counter()
    for rep, mult in product(range(REPS), MULTS):
        if (mult, rep) in done:
            continue
        rows.extend(_group(fm, did, data.X, data.y, mult, rep, base_seed, predict))
        _atomic(pd.DataFrame(rows), partial)
    _atomic(pd.DataFrame(rows), out)
    partial.unlink(missing_ok=True)
    return f"done {fm} {did} ({time.perf_counter() - t0:.0f}s)"


def _atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def main() -> None:
    from simaudit.settings import settings

    sys.path.insert(0, str(HERE.parent))
    from common import dev_ids

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--fms", default=",".join(FMS))
    args = ap.parse_args()
    if args.workers > 3:
        raise SystemExit("at most 3 workers (shared machine)")
    RAW.mkdir(parents=True, exist_ok=True)
    ids = dev_ids()
    fms = args.fms.split(",")
    (OUT / "provenance.json").write_text(json.dumps(dict(
        started=datetime.now().isoformat(timespec="seconds"), base_seed=settings.base_seed,
        seed_namespace=design.SEED_NAMESPACE, mults=MULTS, reps=REPS, levels=design.LEVELS,
        fms=fms, datasets=ids, plan="paper/p1/exploratory/b_dose_response_plan.md",
    ), indent=2))
    jobs = [(fm, did) for fm in fms for did in ids]
    print(f"{datetime.now():%F %T} start {len(jobs)} jobs, workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, fm, did, settings.base_seed): (fm, did) for fm, did in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            try:
                msg = fut.result()
            except Exception as exc:  # noqa: BLE001
                msg = f"FAILED {job}: {exc!r}"
                with (OUT / "failures.jsonl").open("a") as h:
                    h.write(json.dumps(dict(job=job, error=repr(exc))) + "\n")
            print(f"{datetime.now():%F %T} [{n}/{len(jobs)}] {msg}", flush=True)
    print(f"{datetime.now():%F %T} finished", flush=True)


if __name__ == "__main__":
    main()
