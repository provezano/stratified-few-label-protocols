"""Pilot for the frozen plan: per-table quantities on the stored newer-source (Study 15, seen) and PMLB/UCI (Study 12,
seen) fits; no fresh table is read. Writes results/pilot_{sf,pmlb}.csv. Run from tmp: nice -n 15 <venv> -I pilot.py"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import confirm_lib as L
EVD = Path("evidence_2026-11-06")
sys.path.insert(0, str(EVD / "code"))
import diagnose_nonrep as D
OUT = HERE.parent / "results"
for s in ("sf", "pmlb"):
    keys = pd.read_csv(EVD / f"results/B_pertable_gaps_{s}.csv", dtype={"dataset": str}).dataset
    rows = []
    for k in keys:
        y = D.load_y(s, k)
        fk = k if s == "sf" else k
        rows.append(L.table_quantities(D.FOLDER[s], fk, np.bincount(y)))
        print(s, k, flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"pilot_{s}.csv", index=False)
