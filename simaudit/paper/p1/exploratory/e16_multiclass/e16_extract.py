"""E16 step 1 (exploratory): per-draw confusion matrices from stored FM probabilities.

Read-only inputs:
  * results/study9/t1prime/*.parquet (52 tables x 4 FMs; strategies random and stratified);
  * results/study7/a/*.parquet (DEV, 20 tables x 4 FMs; random and stratified);
  * class labels (and, for centroids, features) through the project loaders.
No FM is fitted. sealed_heldout/ is never touched.

For every stored fit we rebuild the predicted labels exactly as the runners' metric_set does
(argmax over all C probability columns, unseen columns are 0) on the rows the run evaluated
(T1': the stored ``query`` rows; DEV: the complement of the labelled set), and store
  * N: context composition (labels per class in the labelled set),
  * cm: C x C confusion matrix on the evaluated rows (rows = true, columns = predicted),
  * soft: mean predicted probability per class over the evaluated rows,
  * macro_f1 (stored) and macro_f1_re (recomputed, must agree).
Per table it also stores class counts of the labelling pool and z-scored class centroid
distances (for H3).

Run from simaudit/: OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python paper/p1/exploratory/e16_multiclass/e16_extract.py
Writes e16_multiclass/out/draws.parquet and out/tables.parquet.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.runner.study9 import dataset_key, layout, row_permutation  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
FMS = ("fm1", "fm2", "fm3", "fm5")
STRATS = ("random", "stratified")
S9 = settings.results_dir / "study9" / "t1prime"
S7 = settings.results_dir / "study7" / "a"
DROP = {694: [1]}  # Study 7 Addendum 1 S4: 694's ID column (only used for centroids here)


def cm_of(y_true: np.ndarray, pred: np.ndarray, C: int) -> np.ndarray:
    return np.bincount(y_true * C + pred, minlength=C * C).reshape(C, C)


def macro_f1(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    return float(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0).mean())


def centroid_dist(X: np.ndarray, y: np.ndarray, C: int) -> np.ndarray:
    sd = X.std(0)
    Z = (X - X.mean(0)) / np.where(sd > 0, sd, 1)
    cen = np.stack([Z[y == k].mean(0) for k in range(C)])
    return np.sqrt(((cen[:, None, :] - cen[None, :, :]) ** 2).sum(-1))


def t1prime():
    rows = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())["datasets"]
    draws, tabs = [], []
    for r in rows:
        did = int(r["did"])
        key = dataset_key(did)
        data = load(did, settings.data_cache_dir)
        y = data.y
        C = int(y.max()) + 1
        perm = row_permutation(len(y), did, settings.base_seed)
        s, _ = layout(len(y))
        cP = np.bincount(y[perm][s:], minlength=C)
        tabs.append(dict(set="t1prime", dataset=key, did=did, name=data.name, C=C, n=len(y),
                         counts_pool=cP.tolist(), counts_all=np.bincount(y, minlength=C).tolist(),
                         cdist=centroid_dist(data.X, y, C).ravel().tolist()))
        for fm in FMS:
            df = pd.read_parquet(S9 / f"{key}_{fm}.parquet",
                                 columns=["strategy", "m", "rep", "labeled", "query", "proba", "macro_f1"])
            df = df[df.strategy.isin(STRATS)]
            for t in df.itertuples(index=False):
                q = np.asarray(t.query, int)
                P = np.asarray(t.proba, float).reshape(len(q), C)
                pred = P.argmax(1)
                cm = cm_of(y[q], pred, C)
                N = np.bincount(y[np.asarray(t.labeled, int)], minlength=C)
                draws.append(dict(set="t1prime", dataset=key, fm=fm, strategy=t.strategy, m=int(t.m), rep=int(t.rep),
                                  C=C, N=N.tolist(), cm=cm.ravel().tolist(), soft=P.mean(0).tolist(),
                                  macro_f1=float(t.macro_f1), macro_f1_re=macro_f1(cm)))
        print(key, C, flush=True)
    return draws, tabs


def dev():
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_i2.yaml").read_text())["dev"]
    draws, tabs = [], []
    for r in pool:
        did = int(r["did"])
        data = load(did, settings.data_cache_dir)
        y = data.y
        C = int(y.max()) + 1
        n = len(y)
        X = np.delete(data.X, DROP[did], axis=1) if did in DROP else data.X
        tabs.append(dict(set="dev", dataset=f"dev_{did}", did=did, name=data.name, C=C, n=n,
                         counts_pool=np.bincount(y, minlength=C).tolist(),
                         counts_all=np.bincount(y, minlength=C).tolist(),
                         cdist=centroid_dist(X, y, C).ravel().tolist()))
        for fm in FMS:
            df = pd.read_parquet(S7 / f"{did}_{fm}.parquet",
                                 columns=["strategy", "mult", "rep", "labeled", "proba", "macro_f1"])
            df = df[df.strategy.isin(STRATS)]
            for t in df.itertuples(index=False):
                lab = np.asarray(t.labeled, int)
                rest = np.setdiff1d(np.arange(n), lab)
                P = np.asarray(t.proba, float).reshape(n, C)[rest]
                cm = cm_of(y[rest], P.argmax(1), C)
                N = np.bincount(y[lab], minlength=C)
                draws.append(dict(set="dev", dataset=f"dev_{did}", fm=fm, strategy=t.strategy, m=int(t.mult),
                                  rep=int(t.rep), C=C, N=N.tolist(), cm=cm.ravel().tolist(), soft=P.mean(0).tolist(),
                                  macro_f1=float(t.macro_f1), macro_f1_re=macro_f1(cm)))
        print(did, C, flush=True)
    return draws, tabs


def main():
    d1, t1 = t1prime()
    d2, t2 = dev()
    D = pd.DataFrame(d1 + d2)
    T = pd.DataFrame(t1 + t2)
    err = (D.macro_f1 - D.macro_f1_re).abs()
    print("max |stored - recomputed macro-F1|:", err.max(), "draws:", len(D))
    D.to_parquet(OUT / "draws.parquet", index=False)
    T.to_parquet(OUT / "tables.parquet", index=False)


if __name__ == "__main__":
    main()
