"""EXPLORATORY (after the frozen analysis): why is the scatter part smaller on the fresh tables? Compares per-table scatter
parts of the stored evidence fits (exploration + recent sets) with the probe, against table traits:
the expected scatter of a random composition, V = mean over m of (phi/B)(1 - sum pi^2), the imbalance r = C*pi_min,
C, and binary. Fits scatter ~ V (per FM) on the stored tables and predicts the fresh ones. Writes results/diag_scatter.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
B = Path(__file__).resolve().parents[1]; E = B.parent / "evidence_2026-11-06/results"
def vterm(counts):
    c = np.asarray(counts, float); n = c.sum(); C = len(c); pi = c / n
    return float(np.mean([((n - m * C) / (n - 1)) / (m * C) * (1 - (pi ** 2).sum()) for m in (3, 5, 10)]))
a = pd.read_csv(E / "A_per_table.csv")
v = pd.read_csv(E / "V_pertable_MF.csv").rename(columns={"learner": "fm"})
old = v.merge(a[["set", "dataset", "fm", "C", "r"]], on=["set", "dataset", "fm"])
raw = pd.concat([pd.read_parquet(p, columns=["dataset", "learner", "pool_counts"]) for p in sorted((E.parent / "raw").glob("*.parquet")) if "partial" not in p.name])
pc = raw.groupby("dataset").pool_counts.first()
old["V"] = old.dataset.map(lambda k: vterm([int(x) for x in pc[k].split(";")]))
old = old.rename(columns={"variance": "scatter"})
new = pd.read_csv(B / "results/pertable.csv").rename(columns={"scatter_MF": "scatter"})
rawn = pd.concat([pd.read_parquet(p, columns=["dataset", "pool_counts"]) for p in sorted((B / "raw").glob("*.parquet"))])
pcn = rawn.groupby("dataset").pool_counts.first()
new["V"] = new.key.map(lambda k: vterm([int(x) for x in pcn[k].split(";")]))
new["C"] = new.key.map(lambda k: len(pcn[k].split(";")))
out = {"set_means": {}, "fm": {}}
for nm, d in (("exploration", old[old.set == "dev"]), ("recent", old[old.set == "sf"]), ("fresh", new)):
    out["set_means"][nm] = dict(scatter=float(d.scatter.mean()), V=float(d.V.mean()), r=float(d.r.mean()),
                                binary=float((d.C == 2).mean()), n=int(d[["C"]].shape[0] / 4))
for fm in ("fm1", "fm2", "fm3", "fm5"):
    o = old[old.fm == fm]; f = new[new.fm == fm]
    sl, ic, rr, p, se = stats.linregress(o.V, o.scatter)
    pred = ic + sl * f.V
    out["fm"][fm] = dict(slope=float(sl), intercept=float(ic), r_old=float(rr), old_mean=float(o.scatter.mean()),
                         fresh_obs=float(f.scatter.mean()), fresh_pred=float(pred.mean()),
                         rho_fresh=float(stats.spearmanr(f.V, f.scatter).statistic))
json.dump(out, open(B / "results/diag_scatter.json", "w"), indent=1)
print(json.dumps(out, indent=1))
# --- signal (balanced accuracy of random draws at m=10, rescaled from chance): stored sets from D_traits, fresh from the probe's V level at m=10
dt = pd.read_csv(E / "D_traits.csv")
sig_old = {(r.set, r.dataset, fm): r[f"signal_{fm}"] for _, r in dt.iterrows() for fm in ("fm1", "fm2", "fm3", "fm5")}
old["signal"] = [sig_old.get((s, d, f), np.nan) for s, d, f in zip(old.set, old.dataset, old.fm)]
rn = pd.concat([pd.read_parquet(p) for p in sorted((B / "raw").glob("*.parquet"))])
rn = rn[(rn.level == "V") & (rn.mult == 10)]
sn = rn.groupby(["dataset", "learner"]).apply(lambda g: (g.balanced_accuracy.mean() - 1 / g.C.iloc[0]) / (1 - 1 / g.C.iloc[0]))
new["signal"] = [sn.get((k, f), np.nan) for k, f in zip(new.key, new.fm)]
out["signal"] = {}
for fm in ("fm1", "fm2", "fm3", "fm5"):
    o = old[(old.fm == fm)].dropna(subset=["signal"]); f = new[new.fm == fm].dropna(subset=["signal"])
    X = np.c_[np.ones(len(o)), o.V, o.signal]; b, *_ = np.linalg.lstsq(X, o.scatter, rcond=None)
    pred = np.c_[np.ones(len(f)), f.V, f.signal] @ b
    out["signal"][fm] = dict(signal_old=float(o.signal.mean()), signal_fresh=float(f.signal.mean()),
                             fresh_pred_V_signal=float(pred.mean()), fresh_obs=float(f.scatter.mean()),
                             rho_signal_fresh=float(stats.spearmanr(f.signal, f.scatter).statistic),
                             rho_signal_old=float(stats.spearmanr(o.signal, o.scatter).statistic))
json.dump(out, open(B / "results/diag_scatter.json", "w"), indent=1)
print(json.dumps(out["signal"], indent=1))
