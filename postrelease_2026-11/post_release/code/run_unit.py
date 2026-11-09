"""One (table, FM) unit of the fresh-table confirmation (PLAN.txt). The Study 15 job (simaudit runner/study15.run_job,
which reuses runner/study13 unchanged: row permutation ns 35, cell seeds ns 34, redraw seeds, base seed 20261020,
strategies stratified 5 reps / random 20 reps / redraw, m in {1,3,5,10}, complement scoring, per-fit metrics, CPU,
one thread) restricted to arm T: arm F is not used by any hypothesis and is skipped. Arm-T seeds do not depend on
arm F, so every arm-T fit is the one Study 15's runner would make.

Usage (from a temp dir): nice -n 15 <simaudit venv python> -I run_unit.py --key K --fm fm1 [--frame F] [--out DIR]
Output: <out>/<key>_<fm>.parquet (atomic), checkpoints <key>_<fm>.partial.parquet every 25 fits (resumed).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SIMAUDIT = Path(os.environ.get("SIMAUDIT_ROOT", str(Path.home() / "ge-paper" / "simaudit")))
sys.path.insert(0, str(SIMAUDIT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simaudit.data.openml_loader import Dataset, content_hash  # noqa: E402
from simaudit.runner import study13 as s13  # noqa: E402

BASE_SEED = 20261020  # Study 13/15 base seed (settings.base_seed)


def frame_row(frame: Path, key: str) -> dict:
    with frame.open() as f:
        for r in csv.DictReader(f):
            if r["key"] == key:
                return r
    raise KeyError(key)


def load(row: dict, npz_dir: Path) -> Dataset:
    with np.load(npz_dir / f"{row['key']}.npz", allow_pickle=False) as z:
        X = np.asarray(z["X"], dtype=np.float64)
        y = np.asarray(z["y"], dtype=np.int64)
    if content_hash(X, y) != row["sha256"]:
        raise ValueError(f"{row['key']}: content hash != frozen frame entry")
    return Dataset(int(row["did"]), row["key"], 1, X, y, row["sha256"])


def run_unit(key: str, fm: str, frame: Path, npz_dir: Path, out_dir: Path) -> Path:
    row = frame_row(frame, key)
    did = int(row["did"])
    out = s13.output_path(out_dir, key, fm)
    if out.exists():
        return out
    import torch

    torch.set_num_threads(s13.TORCH_THREADS)
    data = load(row, npz_dir)
    perm = s13.row_permutation(len(data.y), did, BASE_SEED)
    X, y = data.X[perm], data.y[perm]
    chosen = s13.selections(key, did, X, y, BASE_SEED, out_dir)
    chosen = chosen[chosen.arm == "T"]
    ctx = dict(X=X, y=y, perm=perm, test=np.zeros(0, dtype=np.int64))
    partial = out.with_name(out.name.replace(".parquet", ".partial.parquet"))
    rows = s13._resume(partial)
    done = {(r["learner"], r["_key"]) for r in rows}
    fits = {k: r for r in rows if (k := s13._random_key(r)) is not None}
    predictors = s13._predictors(fm, None)
    todo = [(cell, sel, name, fn) for cell, sel in s13._cells(chosen) for name, fn in predictors.items()
            if (name, cell) not in done]
    try:
        for count, (cell, sel, name, fn) in enumerate(todo, start=1):
            base = fits.get((name, int(sel["mult"]), int(sel["rep"])))
            meta, record, reused = s13._fit_cell(sel, fn, ctx, base)
            r = dict(meta, learner=name, reused=reused, _key=cell, **record)
            rows.append(r)
            if (k := s13._random_key(r)) is not None:
                fits[k] = r
            if count % s13.CHECKPOINT_EVERY == 0:
                s13._atomic_write(pd.DataFrame(rows), partial)
    finally:
        s13._close(predictors)
    df = pd.DataFrame(rows).drop(columns="_key")
    df.insert(0, "dataset", key)
    s13._atomic_write(df, out)
    partial.unlink(missing_ok=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--fm", required=True, choices=("fm1", "fm2", "fm3", "fm5"))
    ap.add_argument("--frame", default=str(BASE / "plan" / "frame.csv"))
    ap.add_argument("--npz", default=str(BASE / "harvest" / "npz"))
    ap.add_argument("--out", default=str(BASE / "results" / "fits"))
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = run_unit(a.key, a.fm, Path(a.frame), Path(a.npz), out_dir)
    print("done", p, flush=True)


if __name__ == "__main__":
    main()
