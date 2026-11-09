"""Power of the frozen family (PLAN.txt) by resampling the 20 newer-source pilot tables (seen) with replacement to N
tables; Holm over the 22-test family at 0.05. Writes results/power.csv. Seed 20261108."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import confirm_lib as L
import family as F
R = HERE.parent / "results"
t = pd.read_csv(R / "pilot_sf.csv")
tr = pd.read_csv("evidence_2026-11-06/results/D_traits.csv", dtype={"dataset": str})
t = t.merge(tr[["dataset", "qmc_bar", "V", "binary", "log10n", "r", "pred_cov_mean", "signal_mean"]], on="dataset")
t["pred_pre"] = F.predict_pre(t)
rng = np.random.default_rng(20261108)
rows = []
for N in (20, 25, 30, 35, 40):
    hits = {}
    for b in range(600):
        s = t.iloc[rng.integers(0, len(t), N)].reset_index(drop=True)
        ps, ok, fam = F.family_pvalues(s)
        adj = F.adjusted(ps, fam)
        for k in ps:
            hits[k] = hits.get(k, 0) + int(adj[k] < 0.05 and ok[k])
    for k, v in hits.items():
        rows.append(dict(N=N, test=k, power=v / 600))
    print(N, {k: v / 600 for k, v in hits.items()}, flush=True)
pd.DataFrame(rows).to_csv(R / "power.csv", index=False)
