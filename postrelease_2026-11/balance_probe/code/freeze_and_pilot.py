"""Freeze the prior-response coefficients from the stored evidence fits (exploration + recent sets, 40 tables) and run the
power pilot for H-response on the same stored fits (already seen; pilot only). Writes plan/frozen_coefs.json, results/pilot.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
B = Path(__file__).resolve().parents[1]; E = B.parent / "evidence_2026-11-06/results"
FMS = ["fm1", "fm2", "fm3", "fm5"]
a = pd.read_csv(E / "A_per_table.csv")
X = np.c_[np.ones(len(a)), a.d_BA, a.d_ACC]
coef, *_ = np.linalg.lstsq(X, a.d_MF, rcond=None)
r2 = 1 - ((a.d_MF - X @ coef) ** 2).sum() / ((a.d_MF - a.d_MF.mean()) ** 2).sum()
eff = {s: {fm: float(g.d_BA.mean() / abs(g.d_ACC.mean())) for fm, g in a[a.set == s].groupby("fm")} for s in ("dev", "sf")}
json.dump(dict(source="evidence_2026-11-06/results/A_per_table.csv (dev+sf, 40 tables x 4 FMs, lambda1-lambda0, m 3/5/10)",
               intercept=coef[0], a_BA=coef[1], b_ACC=coef[2], r2=r2, n=len(a), efficiency_stored=eff,
               predicted_order_efficiency=["fm5", "fm1", "fm3", "fm2"]), open(B / "plan/frozen_coefs.json", "w"), indent=1)
print("coef", coef.round(4), "R2", round(r2, 3), "eff", eff)
# pilot: within-table centred Spearman between predicted balance part (variance + frozen shift) and observed S - random (m>=3)
v = pd.read_csv(E / "V_pertable_MF.csv").rename(columns={"learner": "fm"})
o = pd.read_csv(E / "A_observed_gaps.csv"); o = o[(o.metric == "MF") & (o.mults == "m3plus")]
d = a.merge(v[["set", "dataset", "fm", "variance", "total"]], on=["set", "dataset", "fm"]).merge(o[["set", "dataset", "fm", "obs_gap"]], on=["set", "dataset", "fm"])
d["pred"] = d.variance + coef[0] + coef[1] * d.d_BA + coef[2] * d.d_ACC
rng = np.random.default_rng(1)
from scipy.stats import rankdata
d = d.sort_values(["set", "dataset", "fm"])
d = d[d.groupby(["set", "dataset"]).fm.transform("count") == 4]
O = d.obs_gap.to_numpy().reshape(-1, 4)
def crho(P, O):
    a = (P - P.mean(1, keepdims=True)).ravel(); b = (O - O.mean(1, keepdims=True)).ravel()
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]
def permp(P, O, n):
    r0 = crho(P, O); k = 0
    for _ in range(n):
        idx = np.argsort(rng.random(P.shape), axis=1); k += crho(np.take_along_axis(P, idx, 1), O) >= r0
    return r0, (1 + k) / (1 + n)
out = {}
for col in ("pred", "total"):
    P = d[col].to_numpy().reshape(-1, 4)
    r0, p0 = permp(P, O, 5000)
    hits = 0; R = 500
    for _ in range(R):
        j = rng.integers(0, len(P), 39); hits += permp(P[j], O[j], 400)[1] <= 0.025
    out[col] = dict(rho=float(r0), perm_p=float(p0), power_n39_alpha025=hits / R, n_tables=len(P))
    print(col, out[col], flush=True)
out["pooled_rho_pred"] = float(stats.spearmanr(d.pred, d.obs_gap).statistic)
json.dump(out, open(B / "results/pilot.json", "w"), indent=1)
print(out)
