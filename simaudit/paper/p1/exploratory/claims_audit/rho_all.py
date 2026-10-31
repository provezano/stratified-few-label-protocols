"""EXPLORATORY (claims audit, 2026-10-28): rho(1) for all four FMs on T1' (Study 9 stored results).

Reuses E11 (e11_study9_sensitivity.per_table / rho_summary) unchanged, for fm1, fm2, fm3, fm5.
In Study 9, TabICLv2 (fm2) is the primary learner and TabPFN v2 (fm1) the pre-specified
replication; fm3 and fm5 are exploratory here. fm1/fm2 must reproduce out/e11_rho.csv.
Writes out/rho_all.csv (summary) and out/rho_all_tables.csv (per table).
Run from repo root:  nice -n 15 .venv/bin/python -I paper/p1/exploratory/claims_audit/rho_all.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.dont_write_bytecode = True

import pandas as pd  # noqa: E402

import e11_study9_sensitivity as e  # noqa: E402

raw = e.a9.load_results(e.cm.settings.results_dir / "study9" / "t1prime")
curve = e.a9.curves(raw)
t = pd.concat([e.per_table(curve, lr)[0] for lr in ("fm1", "fm2", "fm3", "fm5")])
(HERE / "out").mkdir(exist_ok=True)
t.to_csv(HERE / "out" / "rho_all_tables.csv", index=False)
r = pd.DataFrame(e.rho_summary(t))
r.to_csv(HERE / "out" / "rho_all.csv", index=False)
print(r[["learner", "model", "median", "lo", "hi", "n", "share_gt1"]].to_string())
