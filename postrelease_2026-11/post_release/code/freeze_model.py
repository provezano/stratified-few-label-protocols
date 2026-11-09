"""Freeze the transfer models used by PLAN.txt (no fresh table is read).
pre : FM-mean S-R macro-F1 gap ~ 1 + qmc_bar + V + binary + log10n, OLS on the 72 standard tables (pre-data: every
      term is a function of n and the class counts only). PRIMARY prediction (H-pred).
m2  : the paper's model M2 without I_new (the diagnosis specification, sha256 7a263e29...), refitted identically on the same 72
      tables: gap ~ 1 + pred_cov + V + binary + signal + log10n (pred_cov and signal need the fits). SECONDARY.
pre92: the pre-data model refitted on all 92 seen tables (72 + 20 newer-source). Descriptive only.
Writes plan/frozen_model.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
EVD = Path("evidence_2026-11-06/results")
t = pd.read_csv(EVD / "D_traits.csv", dtype={"dataset": str})
t["pred_cov"], t["signal"] = t.pred_cov_mean, t.signal_mean
std = t[t.set != "sf"]
def fit(d, xs):
    X = np.column_stack([np.ones(len(d))] + [d[x] for x in xs]); b = np.linalg.lstsq(X, d.gap_mean, rcond=None)[0]
    return dict(const=float(b[0]), **{x: float(v) for x, v in zip(xs, b[1:])})
out = dict(pre=fit(std, ["qmc_bar", "V", "binary", "log10n"]),
           m2=fit(std, ["pred_cov", "V", "binary", "signal", "log10n"]),
           pre92=fit(t, ["qmc_bar", "V", "binary", "log10n"]),
           source=str(EVD / "D_traits.csv"), n_std=int(len(std)), outcome="gap_mean (FM mean, m in {1,3,5,10})")
p = Path(__file__).resolve().parents[1] / "plan" / "frozen_model.json"
p.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
