"""Macro precision gap (S - R) on T1' (checks the 'all class-averaged metrics' wording)."""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))
from v_extract import DIDS, load, CACHE


def job(key):
    did = DIDS[key]
    y = load(did, CACHE).y.astype(np.int64); C = int(y.max()) + 1
    rows = []
    for fm in ("fm1", "fm2", "fm3", "fm5"):
        d = pd.read_parquet(ROOT / "results/study9/t1prime" / f"{key}_{fm}.parquet",
                            columns=["strategy", "m", "rep", "query", "proba"])
        d = d[d.strategy.isin(["random", "stratified"])]
        for r in d.itertuples():
            q = np.asarray(r.query); P = np.asarray(r.proba, float).reshape(len(q), C)
            yq = y[q]; pred = P.argmax(1)
            prec = [np.mean(yq[pred == c] == c) if (pred == c).any() else 0.0 for c in range(C)]
            rows.append(dict(dataset=key, learner=fm, strategy=r.strategy, m=r.m, prec=np.mean(prec)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with Pool(4) as p:
        f = pd.concat(p.map(job, list(DIDS)), ignore_index=True)
    g = f.groupby(["dataset", "learner", "m", "strategy"]).prec.mean().unstack()
    gap = (g.stratified - g.random).dropna().groupby(["dataset", "m"]).mean()
    out = []
    for m, v in gap.groupby(level=1):
        out.append(f"m={m}: macro-precision gap {v.mean():+.4f} p={stats.wilcoxon(v).pvalue:.2g} n={len(v)}")
    txt = "\n".join(out); print(txt)
    (HERE / "out" / "v_e17_precision.txt").write_text(txt + "\n")
