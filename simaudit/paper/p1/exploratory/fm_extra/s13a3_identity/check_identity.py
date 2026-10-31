"""DEV identity check for Study 13 Amendment 3 (fm8 LimiX-16M through the worker).

Engineering check on DEV only; no HELD-OUT file is read. The learners are the
runner's own: ``runner.study13.external_fm(fm)`` (the subprocess worker in the
FM's pinned venv), called through ``runner.study9.predict_on`` (the runner's
missing-class rule) with query = every row, which is the call that produced the
stored DEV Part (a) fits (``selection.strategies.predict_with_budget``).

Cells: the stored Study 7 Part (a) labelled sets and seeds
(``results/study7/a/{did}_fm2.parquet``, read-only), 4 tables x 2 cells = 8.

- fm8: the new worker vs the stored DEV fm8 Part (a) probabilities
  (``out/s7a/{did}_fm8.parquet``, float32).
- fm6: the new worker vs the stored DEV fm6 Part (a) probabilities
  (``out/s7a``) and vs the worker as committed before Amendment 3 (HEAD copy).
- fm7: the new worker vs the stored seeded DEV fm7 run (``out/s7a_v2``) and vs
  the HEAD-copy worker.

Usage (from simaudit/, project .venv; the worker venvs are started by
``external_fm``):
    nice -n 15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
      uv run --offline python paper/p1/exploratory/fm_extra/s13a3_identity/check_identity.py \
      run FM [--old-worker PATH] OUT.npz
    ... check_identity.py compare OUT_DIR   -> out/identity.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FMX = HERE.parent / "out"
TABLES = (1510, 40496, 337, 46562)
CELLS = (("stratified", 1, 0), ("random", 3, 1))
STORED = dict(fm6="s7a", fm7="s7a_v2", fm8="s7a")


def run(fm: str, out: str, old_worker: str | None) -> None:
    import numpy as np
    import pandas as pd
    import torch

    from simaudit.data.openml_loader import load
    from simaudit.runner import study13
    from simaudit.runner.study9 import predict_on
    from simaudit.settings import settings

    torch.set_num_threads(1)
    predict = study13.external_fm(fm)
    if old_worker:
        predict = study13.ExternalFM(fm, predict.python, worker=Path(old_worker))
    res: dict[str, np.ndarray] = {}
    try:
        for did in TABLES:
            data = load(did, settings.data_cache_dir)
            a = pd.read_parquet(settings.results_dir / "study7" / "a" / f"{did}_fm2.parquet",
                                columns=["strategy", "mult", "rep", "seed", "labeled"])
            everything = np.arange(len(data.y), dtype=np.int64)
            for name, m, rep in CELLS:
                r = a[(a.strategy == name) & (a.mult == m) & (a.rep == rep)].iloc[0]
                lab = np.asarray(r.labeled, dtype=np.int64)
                t = time.perf_counter()
                p = predict_on(predict, data.X, lab, data.y, everything,
                               int(data.y.max()) + 1, int(r.seed))
                res[f"{did}__{name}__{m}__{rep}"] = p
                print(fm, did, name, m, rep, len(np.unique(data.y[lab])), "seen",
                      f"{time.perf_counter() - t:.1f}s", flush=True)
    finally:
        pins = predict.pins
        predict.close()
    np.savez(out, **res, _pins=np.array(json.dumps(pins, sort_keys=True)),
             _worker=np.array(str(predict.worker)))


def compare(folder: str) -> None:
    import numpy as np
    import pandas as pd

    from simaudit.analysis.study7 import complement_metrics
    from simaudit.data.openml_loader import load
    from simaudit.settings import settings

    d = Path(folder)
    report: dict[str, dict] = {}
    for fm in ("fm6", "fm7", "fm8"):
        new = np.load(d / f"{fm}_new.npz")
        old = np.load(d / f"{fm}_old.npz") if (d / f"{fm}_old.npz").exists() else None
        e = dict(cells=0, stored=f"out/{STORED[fm]}", max_abs_vs_stored=0.0,
                 bit_identical_vs_stored_f32=True, max_abs_metric_diff_vs_stored=0.0,
                 worker_pins=json.loads(str(new["_pins"])))
        if old is not None:
            e.update(max_abs_new_vs_old_worker=0.0, bit_identical_new_vs_old_worker=True)
        for k in new.files:
            if k.startswith("_"):
                continue
            did, name, m, rep = k.split("__")
            s = pd.read_parquet(FMX / STORED[fm] / f"{did}_{fm}.parquet",
                                columns=["strategy", "mult", "rep", "labeled", "proba"])
            s = s[(s.strategy == name) & (s.mult == int(m)) & (s.rep == int(rep))].iloc[0]
            p = new[k]
            stored = np.asarray(s.proba, dtype=np.float32)
            e["cells"] += 1
            e["max_abs_vs_stored"] = max(e["max_abs_vs_stored"],
                                         float(np.abs(p.ravel() - stored.astype(np.float64)).max()))
            e["bit_identical_vs_stored_f32"] &= bool(np.array_equal(p.astype(np.float32).ravel(), stored))
            y = load(int(did), settings.data_cache_dir).y
            lab = np.asarray(s.labeled, dtype=np.int64)
            ma = complement_metrics(y, p, lab)
            mb = complement_metrics(y, stored.reshape(p.shape).astype(np.float64), lab)
            dm = max(abs(ma[x] - mb[x]) for x in ma if np.isfinite(ma[x]) and np.isfinite(mb[x]))
            e["max_abs_metric_diff_vs_stored"] = max(e["max_abs_metric_diff_vs_stored"], float(dm))
            if old is not None:
                e["max_abs_new_vs_old_worker"] = max(e["max_abs_new_vs_old_worker"],
                                                     float(np.abs(p - old[k]).max()))
                e["bit_identical_new_vs_old_worker"] &= bool(np.array_equal(p, old[k]))
        e["pass"] = bool(e["max_abs_vs_stored"] <= 1e-6
                         and e.get("bit_identical_new_vs_old_worker", True))
        report[fm] = e
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "identity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        args = sys.argv[2:]
        old = None
        if "--old-worker" in args:
            i = args.index("--old-worker")
            old = args[i + 1]
            del args[i:i + 2]
        run(args[0], args[1], old)
    else:
        compare(sys.argv[2])
