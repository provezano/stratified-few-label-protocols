"""Runner of the dose arms F, S and N on a T1' subset (EXPLORATORY, preregistered).

Preregistration (committed before any T1' fit):
paper/p1/exploratory/b_dose/PREREG_T1P_SUBSET.md. Subset: b_dose/t1p_subset.csv.

Same design, FMs, budgets, reps and seed scheme as the DEV run (run_dose_b.py,
design.py), with two T1'-specific details fixed in the prereg:
- seed key of a table: runner.study12.seed_did (non-negative for PMLB/UCI ids);
- evaluation panel: the first min(n, 1000) rows of the Study 12 row permutation
  (namespace 33). The FM predicts only the panel rows outside the context; the
  own-complement set is panel minus context, the common set is panel minus the
  union of all feasible contexts of that (table, mult, rep). Contexts are drawn
  from the whole table, as in DEV.

One job = one (FM, table); output out_t1p/raw/{fm}_{key}.parquet, skipped when
present; checkpoints per (mult, rep) group in *.partial.parquet. A job stops
after the current group once --deadline (epoch seconds) has passed.

Usage (from simaudit/):
    nice -n 15 .venv/bin/python paper/p1/exploratory/b_dose/run_dose_t1p.py --workers 6
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

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
from run_dose_b import _metrics  # noqa: E402
from t1p_subset import PANEL_MAX, seed_did  # noqa: E402

FMS = ("fm2", "fm1", "fm5", "fm3")  # as DEV: cheapest first
MULTS = (3, 5, 10)
REPS = 5
PERM_NAMESPACE = 33  # runner.study12.PERM_NAMESPACE
OUT = HERE / "out_t1p"
RAW = OUT / "raw"


def panel_rows(n: int, did: int, base_seed: int) -> np.ndarray:
    """First min(n, PANEL_MAX) rows of the Study 12 permutation (original indices)."""
    rng = np.random.default_rng([base_seed, seed_did(did), PERM_NAMESPACE])
    return np.sort(rng.permutation(n)[:PANEL_MAX]).astype(np.int64)


def _score(y, proba, labeled, own, common) -> dict[str, float]:
    from simaudit.analysis.study7 import prior_correct

    corrected = prior_correct(proba, y[labeled])
    rec: dict[str, float] = {}
    rec.update(_metrics(y, proba, own, ""))
    rec.update(_metrics(y, corrected, own, "pc_"))
    rec.update(_metrics(y, proba, common, "cm_"))
    rec.update(_metrics(y, corrected, common, "cm_pc_"))
    return rec


def _group(fm, key, did, X, y, panel, mult, rep, base_seed, predict) -> list[dict[str, Any]]:
    from simaudit.selection import strategies as sel

    sd = seed_did(did)
    cells = design.rep_cells(y, sd, mult, rep, base_seed)
    feasible = [c for c in cells if c.feasible]
    union = np.unique(np.concatenate([np.array(c.labeled) for c in feasible])) if feasible else np.array([], int)
    common = np.setdiff1d(panel, union)
    seed = design.fm_seed(base_seed, sd, mult, rep)
    n_classes = int(y.max()) + 1
    fitted: dict[tuple[int, ...], tuple[dict[str, float], float]] = {}
    rows = []
    for c in cells:
        rec: dict[str, Any] = dict(
            learner=fm, dataset=key, did=did, arm=c.arm, side=c.side, level=c.level, mult=mult, rep=rep,
            upweighted=c.upweighted, counts=list(c.counts), budget=sum(c.counts),
            share_real=c.share_real, feasible=c.feasible, n_classes=n_classes,
            n_common=len(common), fm_seed=seed,
        )
        if c.feasible:
            if c.labeled not in fitted:
                labeled = np.array(c.labeled, dtype=np.int64)
                own = np.setdiff1d(panel, labeled)
                t0 = time.perf_counter()
                # predict on [context rows; own rows] so predict_with_budget's X holds both
                rows_x = np.concatenate([labeled, own])
                p = sel.predict_with_budget(predict, X[rows_x], np.arange(len(labeled)), y[labeled],
                                            n_classes, seed)
                secs = time.perf_counter() - t0
                proba = np.zeros((len(y), n_classes))
                proba[own] = p[len(labeled):]
                fitted[c.labeled] = (_score(y, proba, labeled, own, common), secs)
                rec["fitted_here"] = True
            else:
                rec["fitted_here"] = False
            scores, secs = fitted[c.labeled]
            rec.update(scores, seconds=secs, n_eval=len(panel) - len(np.intersect1d(panel, c.labeled)))
        rows.append(rec)
    return rows


def run_job(fm: str, key: str, did: int, base_seed: int, deadline: float) -> str:
    import torch

    torch.set_num_threads(1)
    from simaudit.data.openml_loader import load
    from simaudit.runner.study7 import predictor
    from simaudit.settings import settings

    out = RAW / f"{fm}_{key.replace(':', '_')}.parquet"
    if out.exists():
        return f"skip {fm} {key}"
    partial = out.with_name(out.name.replace(".parquet", ".partial.parquet"))
    rows: list[dict[str, Any]] = []
    done: set[tuple[int, int]] = set()
    if partial.exists():
        old = pd.read_parquet(partial)
        rows = old.to_dict("records")
        done = set(zip(old["mult"].astype(int), old["rep"].astype(int)))
    data = load(did, settings.data_cache_dir)
    panel = panel_rows(len(data.y), did, base_seed)
    predict = predictor(fm)
    t0 = time.perf_counter()
    for rep, mult in product(range(REPS), MULTS):
        if (mult, rep) in done:
            continue
        if time.time() > deadline:
            return f"STOPPED (deadline) {fm} {key} after {len(done)} groups"
        rows.extend(_group(fm, key, did, data.X, data.y, panel, mult, rep, base_seed, predict))
        done.add((mult, rep))
        _atomic(pd.DataFrame(rows), partial)
    _atomic(pd.DataFrame(rows), out)
    partial.unlink(missing_ok=True)
    return f"done {fm} {key} ({time.perf_counter() - t0:.0f}s)"


def _atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def main() -> None:
    from simaudit.settings import settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fms", default=",".join(FMS))
    ap.add_argument("--deadline-hours", type=float, default=30.0)
    ap.add_argument("--keys", default="", help="comma-separated subset keys (smoke tests)")
    ap.add_argument("--out", default="", help="override output root (smoke tests)")
    args = ap.parse_args()
    if args.workers > 6:
        raise SystemExit("at most 6 workers (shared machine)")
    global OUT, RAW
    if args.out:
        OUT = Path(args.out)
        RAW = OUT / "raw"
    RAW.mkdir(parents=True, exist_ok=True)
    sub = pd.read_csv(HERE / "t1p_subset.csv")
    if args.keys:
        sub = sub[sub.key.isin(args.keys.split(","))]
    fms = args.fms.split(",")
    deadline = time.time() + 3600 * args.deadline_hours
    prov = OUT / "provenance.json"
    if not prov.exists():
        prov.write_text(json.dumps(dict(
            started=datetime.now().isoformat(timespec="seconds"), base_seed=settings.base_seed,
            seed_namespace=design.SEED_NAMESPACE, perm_namespace=PERM_NAMESPACE, panel_max=PANEL_MAX,
            mults=MULTS, reps=REPS, levels=design.LEVELS, fms=fms, tables=sub.key.tolist(),
            deadline_hours=args.deadline_hours,
            prereg="paper/p1/exploratory/b_dose/PREREG_T1P_SUBSET.md",
        ), indent=2))
    jobs = [(fm, r.key, int(r.did)) for fm in fms for r in sub.itertuples()]
    print(f"{datetime.now():%F %T} start {len(jobs)} jobs, workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, fm, key, did, settings.base_seed, deadline): (fm, key)
                   for fm, key, did in jobs}
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
