"""EXPLORATORY: does Kumo-Tabular predict a declared but unused class? (plan: PLAN.md)

Claims-audit row 26. Kumo-Tabular (``sdm`` 0.2.1) takes its class list from the
categories of the context-label column. In the paper's protocol (``arms.fm._sdm_predict_proba``)
the label column is a string column, so only the seen classes become categories. A
pandas ``Categorical`` keeps declared categories even when no row uses them, and
``TableTensor.from_pandas`` copies ``ser.array.categories`` verbatim, so a declared
class list reaches the model as ``num_classes = C``. This script refits Kumo on the
stored DEV Study 7 Part (a) random draws that missed a class, in two arms:

- ``std``  : the paper's path (``selection.strategies.predict_with_budget`` with
             ``harness.fm_kumo``), i.e. seen classes only, unseen columns = 0
             (control: reproduces the stored probabilities);
- ``decl`` : same call, same seed, same context and query rows, but the label column
             is ``pd.Categorical(..., categories=["0", ..., str(C-1)])``: all C classes
             declared, the missed ones unused. Kumo is called even for one-class
             contexts (binary tables), which the paper's path does not fit.

Read-only inputs: ``results/study7/a/{did}_fm3.parquet`` (labelled sets, seeds,
stored proba). Writes only ``paper/p1/exploratory/kumo_unused/out/``.

Usage (from simaudit/):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
      PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 nice -n 15 .venv/bin/python \
      paper/p1/exploratory/kumo_unused/run_kumo_unused.py --workers 2 [--probe 2]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from simaudit.analysis.study7 import complement_metrics  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.selection import strategies as sel  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get("KUMO_OUT_NAME", "out")
S7A = settings.results_dir / "study7" / "a"
N_ESTIMATORS = 8  # arms.fm.kumo_predict_proba default (as in Study 7)


def dev_ids() -> list[int]:
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_i2.yaml").read_text())
    return [int(d["did"]) for d in pool["dev"]]


def kumo_declared(X_fit, y_fit, X_query, seed, n_classes):  # type: ignore[no-untyped-def]
    """``arms.fm._sdm_predict_proba`` with the label column declared over all classes.

    Only change: ``target`` is a pandas Categorical with categories "0".."C-1" (the
    original class indices), so absent classes are declared but unused.
    """
    import sdm
    import torch

    from simaudit.arms import fm

    model = fm._kumo_model()
    columns = [f"f{i}" for i in range(X_fit.shape[1])]
    frame = pd.DataFrame(np.vstack([X_fit, X_query]), columns=columns)
    labels = np.r_[y_fit, np.full(len(X_query), y_fit[0])].astype(str)
    frame["target"] = pd.Categorical(labels, categories=[str(c) for c in range(n_classes)])
    table = sdm.TableTensor.from_pandas(
        df=frame, stypes=sdm.infer_stypes(frame, overrides={"target": "categorical"}), device="cpu"
    )
    n = len(X_fit)
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = model(
            x_context=table[:n].drop_columns("target"),
            y_context=table[:n, "target"],
            x_query=table[n:].drop_columns("target"),
            num_estimators=N_ESTIMATORS,
        ).to_pandas()
    returned = [str(c) for c in out.columns]
    # Columns Kumo did not return get probability 0 (same convention as expand_proba).
    proba = np.column_stack([out[str(c)].to_numpy(float) if str(c) in out.columns
                             else np.zeros(len(out)) for c in range(n_classes)])
    return proba, returned


def absent_stats(y, proba, labeled):  # type: ignore[no-untyped-def]
    """Absent-class diagnostics on the complement rows."""
    rest = np.setdiff1d(np.arange(len(y)), labeled)
    seen = np.unique(y[labeled])
    C = proba.shape[1]
    absent = np.setdiff1d(np.arange(C), seen)
    p, yt = proba[rest], y[rest]
    pred = p.argmax(1)
    f1s, base = [], []
    for k in absent:
        tp = float(np.sum((pred == k) & (yt == k)))
        npred, ntrue = float(np.sum(pred == k)), float(np.sum(yt == k))
        f1s.append(2 * tp / (npred + ntrue) if npred + ntrue > 0 else 0.0)
        # expected F1 of assigning the same number of rows to k uniformly at random:
        # E[tp] = npred * ntrue / N
        etp = npred * ntrue / len(yt)
        base.append(2 * etp / (npred + ntrue) if npred + ntrue > 0 else 0.0)
    return dict(
        n_absent=len(absent),
        absent_mass=float(p[:, absent].sum(1).mean()),
        absent_max=float(p[:, absent].max()),
        absent_argmax_share=float(np.isin(pred, absent).mean()),
        absent_true_share=float(np.isin(yt, absent).mean()),
        absent_f1=float(np.mean(f1s)),
        absent_f1_random=float(np.mean(base)),
    )


def run_table(did: int, probe: int) -> str:
    import torch

    torch.set_num_threads(1)
    from simaudit.autoresearch import harness

    out = OUT / "runs" / f"{did}.parquet"
    if out.exists():
        return f"skip {did}"
    stored = pd.read_parquet(S7A / f"{did}_fm3.parquet")
    stored = stored[(stored.strategy == "random") & (stored.coverage < 1)]
    if probe:
        stored = stored.head(probe)
    data = load(did, settings.data_cache_dir)
    X, y = data.X, data.y
    C = int(y.max()) + 1
    rows = []
    for r in stored.to_dict("records"):
        labeled = np.asarray(r["labeled"], dtype=np.int64)
        seed = int(r["seed"])
        meta = dict(dataset=did, mult=int(r["mult"]), rep=int(r["rep"]), seed=seed, n_classes=C,
                    n_seen=len(np.unique(y[labeled])))
        stored_proba = np.asarray(r["proba"], dtype=np.float64).reshape(len(y), C)
        for arm in ("std", "decl"):
            t = time.perf_counter()
            returned, error = None, ""
            try:
                if arm == "std":
                    proba = sel.predict_with_budget(harness.fm_kumo, X, labeled, y[labeled], C, seed)
                else:
                    proba, returned = kumo_declared(X[labeled], y[labeled].astype(np.int64), X, seed, C)
            except Exception as exc:  # noqa: BLE001 - recorded (e.g. one-class context)
                error, proba = repr(exc)[:300], None
            secs = time.perf_counter() - t
            if proba is None:
                rows.append(dict(meta, arm=arm, seconds=secs, error=error))
                continue
            rows.append(dict(meta, arm=arm, seconds=secs, error=error,
                             returned_cols=",".join(returned) if returned is not None else "",
                             n_returned=len(returned) if returned is not None else -1,
                             max_abs_diff_stored=float(np.abs(proba - stored_proba).max()),
                             proba=proba.astype(np.float32).ravel(),
                             **complement_metrics(y, proba, labeled),
                             **absent_stats(y, proba, labeled)))
        rows.append(dict(meta, arm="stored", seconds=float(r["seconds"]), max_abs_diff_stored=0.0,
                         proba=stored_proba.astype(np.float32).ravel(),
                         **complement_metrics(y, stored_proba, labeled),
                         **absent_stats(y, stored_proba, labeled)))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(out)
    return f"done {did} ({len(stored)} draws)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--probe", type=int, default=0, help="draws per table (0 = all; debugging only)")
    ap.add_argument("--datasets", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "DONE").unlink(missing_ok=True)
    dids = [int(x) for x in a.datasets.split(",")] if a.datasets else dev_ids()
    todo = []
    for did in dids:
        s = pd.read_parquet(S7A / f"{did}_fm3.parquet", columns=["strategy", "coverage"])
        if ((s.strategy == "random") & (s.coverage < 1)).any():
            todo.append(did)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(run_table, did, a.probe): did for did in todo}
        for f in as_completed(futs):
            try:
                print(f.result(), time.strftime("%H:%M:%S"), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {futs[f]}: {exc!r}", flush=True)
    (OUT / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
