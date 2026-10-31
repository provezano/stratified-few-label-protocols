"""(c) sensitivity: definitions of 'minority recall' at m=1, fm2 (own code)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))
from v_extract import DIDS, t1_perm, load, CACHE  # own module

rows = []
for key, did in DIDS.items():
    d = pd.read_parquet(ROOT / "results/study9/t1prime" / f"{key}_fm2.parquet")
    d = d[d.m == 1]
    y = load(did, CACHE).y.astype(np.int64); C = int(y.max()) + 1; n = len(y)
    perm = t1_perm(n, did); s = min(500, n // 5)
    pc = np.bincount(y[perm[s:]], minlength=C)
    maj = int(np.argmax(pc)); mino = C - 1 - int(np.argmin(pc[::-1]))
    for r in d.itertuples():
        q = np.asarray(r.query); P = np.asarray(r.proba, float).reshape(len(q), C)
        yq = y[q]; pred = P.argmax(1)
        rec = np.array([np.mean(pred[yq == c] == c) if (yq == c).any() else np.nan for c in range(C)])
        nonmaj = yq != maj
        rows.append(dict(dataset=key, strategy=r.strategy, rep=r.rep, C=C,
                         rec_maj=rec[maj], rec_rarest=rec[mino],
                         rec_nonmaj_mean=np.nanmean(np.delete(rec, maj)),
                         rec_nonmaj_pooled=np.mean(pred[nonmaj] == yq[nonmaj])))
f = pd.DataFrame(rows)
g = f.groupby(["dataset", "strategy"]).mean(numeric_only=True).reset_index()
cols = ["rec_maj", "rec_rarest", "rec_nonmaj_mean", "rec_nonmaj_pooled"]
print("MEDIAN of per-table means\n", g.groupby("strategy")[cols].median().round(3))
print("MEAN of per-table means\n", g.groupby("strategy")[cols].mean().round(3))
g.to_csv(HERE / "out" / "e15_c_minority_defs.csv", index=False)
