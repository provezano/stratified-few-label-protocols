"""Pre-data traits and frozen predictions for the frame (PLAN.txt): label-free except the class counts. Writes
plan/predictions.csv (key, n, C, binary, log10n, r, d2, V, qmc_bar [exact miss probability], pred_pre, pred_pre92).
Run before any FM fit on the frame."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
BASE = Path(os.environ.get("CONFIRM_BASE", str(HERE.parent)))
import confirm_lib as L, family as F
fr = pd.read_csv(BASE / "plan" / "frame.csv", dtype={"key": str})
rows = []
for _, r in fr.iterrows():
    c = np.array([int(x) for x in str(r.class_counts).split(";")]); n, C = int(c.sum()), len(c); pi = c / n; u = 1 / C
    d2 = float(((pi - u) ** 2).sum())
    V = float(np.mean([d2 + (1 - (pi ** 2).sum()) / (m * C) * (n - m * C) / (n - 1) for m in (3, 5, 10)]))
    rows.append(dict(key=r.key, n=n, C=C, binary=int(C == 2), log10n=np.log10(n), r=C * pi.min(), d2=d2, V=V,
                     qmc_bar=float(np.mean([L.exact_miss(c, m * C) for m in L.MULTS]))))
t = pd.DataFrame(rows)
coef = F.load_coef()
t["pred_pre"] = F.predict_pre(t, coef["pre"]); t["pred_pre92"] = F.predict_pre(t, coef["pre92"])
t.to_csv(BASE / "plan" / "predictions.csv", index=False); print(t.round(4).to_string())
