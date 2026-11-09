"""Frozen analysis of the balance-mechanism probe (plan/PLAN.txt sections 4-6). Runs once at ALL_DONE.
Writes results/RESULTS.json, results/RESULTS.md, results/pertable.csv."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
B = Path(__file__).resolve().parents[1]
FMS = ["fm1", "fm2", "fm3", "fm5"]; TABPFN = ["fm1", "fm5"]; OTHER = ["fm2", "fm3"]
NAME = dict(fm1="TabPFN v2", fm2="TabICLv2", fm3="Kumo", fm5="TabPFN v3.5")
coef = json.load(open(B / "plan/frozen_coefs.json"))
frame = pd.read_csv(B / "plan/frame.csv")
raw = pd.concat([pd.read_parquet(p) for p in sorted((B / "raw").glob("*.parquet")) if ".partial" not in p.name], ignore_index=True)
n_err = int((raw.error != "").sum())
ok = raw[raw.error == ""]
# a (table, FM, m, rep) cell counts only if all four compositions were fitted without error (plan sec. 5)
cnt = ok.groupby(["dataset", "learner", "mult", "rep"])["level"].nunique()
full = cnt[cnt == 4].index
ok = ok.set_index(["dataset", "learner", "mult", "rep"]).loc[full].reset_index()
cm = ok.groupby(["dataset", "learner", "mult", "level"])[["macro_f1", "balanced_accuracy", "accuracy"]].mean()
lv = cm.groupby(["dataset", "learner", "level"]).mean().unstack("level")  # mean over m of per-m rep means
pt = pd.DataFrame(index=lv.index)
for short, col in (("MF", "macro_f1"), ("BA", "balanced_accuracy"), ("ACC", "accuracy")):
    pt[f"scatter_{short}"] = lv[(col, "0.0")] - lv[(col, "V")]
    pt[f"shift_{short}"] = lv[(col, "1.0")] - lv[(col, "0.0")]
    pt[f"bal_{short}"] = lv[(col, "1.0")] - lv[(col, "V")]
lev = lv["macro_f1"][["0.0", "0.5", "1.0"]]
pt["lstar"] = lev.to_numpy().argmax(axis=1) * 0.5
pt["pred_shift"] = coef["intercept"] + coef["a_BA"] * pt.shift_BA + coef["b_ACC"] * pt.shift_ACC
pt["pred_bal"] = pt.scatter_MF + pt.pred_shift
pt = pt.reset_index().rename(columns={"dataset": "key", "learner": "fm"})
pc = ok.groupby("dataset").pool_counts.first().apply(lambda s: np.array([int(v) for v in s.split(";")]))
trait = pd.Series({k: len(c) * c.min() / c.sum() for k, c in pc.items()}, name="r")
pt = pt.merge(trait, left_on="key", right_index=True).merge(frame[["key", "set"]], on="key")
pt.to_csv(B / "results/pertable.csv", index=False)
R = dict(n_error_fits=n_err, n_units=int(raw.groupby(["dataset", "learner"]).ngroups),
         hosts=raw.groupby("host").apply(lambda g: g.groupby(["dataset", "learner"]).ngroups).to_dict())
# ---------------- primary
ps = {}
sc = {}
for fm in FMS:
    x = pt[pt.fm == fm].scatter_MF.dropna().to_numpy()
    sc[fm] = dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)), n_pos=int((x > 0).sum()),
                  p_one=float(stats.wilcoxon(x[x != 0], alternative="greater").pvalue))
ps["H_scatter"] = max(v["p_one"] for v in sc.values())  # intersection-union: every FM must pass
conf = pd.read_csv(B.parent / "confirm_2026-11-08/results/fresh_pertable.csv")
obs = conf.melt(id_vars="key", value_vars=[f"bal_mf_{fm}" for fm in FMS], var_name="fm", value_name="obs_bal")
obs["fm"] = obs.fm.str.replace("bal_mf_", "")
d = pt[pt.set == "confirm"].merge(obs, on=["key", "fm"]).dropna(subset=["pred_bal", "obs_bal"])
d = d[d.groupby("key").fm.transform("count") == 4]
def crho(df, col):
    g = df.groupby("key")
    return stats.spearmanr(df[col] - g[col].transform("mean"), df.obs_bal - g.obs_bal.transform("mean")).statistic
rng = np.random.default_rng(20261108)
r0 = crho(d, "pred_bal")
keys = d.key.to_numpy(); vals = d.pred_bal.to_numpy(); idx = [np.flatnonzero(keys == k) for k in pd.unique(keys)]
perm = []
for _ in range(20000):
    v = vals.copy()
    for ii in idx:
        v[ii] = rng.permutation(v[ii])
    perm.append(crho(d.assign(pred_bal=v), "pred_bal"))
perm = np.array(perm)
ps["H_response"] = float((1 + (perm >= r0).sum()) / (1 + len(perm)))
slope = stats.linregress(d.pred_bal - d.groupby("key").pred_bal.transform("mean"), d.obs_bal - d.groupby("key").obs_bal.transform("mean"))
pooled = stats.spearmanr(d.pred_bal, d.obs_bal)
order = d.groupby("key").fm.count()
R["primary"] = dict(H_scatter=dict(per_fm=sc, p=ps["H_scatter"]),
                    H_response=dict(n_tables=int(order.size), rho_within_table=float(r0), perm_p=ps["H_response"],
                                    slope_within=float(slope.slope), slope_ci=[float(slope.slope - 1.96 * slope.stderr), float(slope.slope + 1.96 * slope.stderr)],
                                    pooled_rho=float(pooled.statistic), pred_mean={fm: float(d[d.fm == fm].pred_bal.mean()) for fm in FMS},
                                    obs_mean={fm: float(d[d.fm == fm].obs_bal.mean()) for fm in FMS}))
def holm(p):
    ks = sorted(p, key=p.get); out = {}; run = 0.0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (len(ks) - i) * p[k])); out[k] = run
    return out
R["primary_holm"] = holm(ps)
# ---------------- secondary
w = pt.pivot(index="key", columns="fm")
ss = {}
dl = (w.lstar[TABPFN].mean(axis=1) - w.lstar[OTHER].mean(axis=1)).dropna()
ss["H_lstar"] = float(stats.wilcoxon(dl[dl != 0], alternative="greater").pvalue) if (dl != 0).sum() > 1 else 1.0
ds = (w.shift_MF[TABPFN].mean(axis=1) - w.shift_MF[OTHER].mean(axis=1)).dropna()
ss["H_shiftsign"] = float(stats.wilcoxon(ds[ds != 0], alternative="greater").pvalue)
im = pd.concat([w.shift_MF[OTHER].mean(axis=1).rename("s"), trait], axis=1).dropna()
rim = stats.spearmanr(im.s, im.r); ss["H_imb"] = float(rim.pvalue / 2 if rim.statistic > 0 else 1 - rim.pvalue / 2)
R["secondary"] = dict(H_lstar=dict(mean_diff=float(dl.mean()), n=len(dl), p=ss["H_lstar"]),
                      H_shiftsign=dict(mean_diff=float(ds.mean()), n=len(ds), p=ss["H_shiftsign"]),
                      H_imb=dict(rho=float(rim.statistic), n=len(im), p=ss["H_imb"]))
R["secondary_holm"] = holm(ss)
# ---------------- descriptive (plan sec. 6)
desc = {}
for fm in FMS:
    g = pt[pt.fm == fm]
    def ci(x):
        x = x.dropna().to_numpy(); b = [rng.choice(x, len(x)).mean() for _ in range(5000)]
        return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    desc[fm] = dict(name=NAME[fm], scatter=ci(g.scatter_MF), shift=ci(g.shift_MF), bal=ci(g.bal_MF),
                    efficiency=float(g.shift_BA.mean() / abs(g.shift_ACC.mean())), lstar_mean=float(g.lstar.mean()),
                    heldaside_scatter=[float(v) for v in g[g.set == "heldaside"].scatter_MF],
                    heldaside_shift=[float(v) for v in g[g.set == "heldaside"].shift_MF])
R["descriptive"] = desc
eff_order = sorted(FMS, key=lambda f: -desc[f]["efficiency"]); bal_order = sorted(FMS, key=lambda f: -desc[f]["bal"][0])
R["order"] = dict(predicted=coef["predicted_order_efficiency"], efficiency=eff_order, balance_part=bal_order,
                  obs_conf_balance=sorted(FMS, key=lambda f: -R["primary"]["H_response"]["obs_mean"][f]))
R["verdict"] = dict(H_scatter=R["primary_holm"]["H_scatter"] < 0.05, H_response=R["primary_holm"]["H_response"] < 0.05)
json.dump(R, open(B / "results/RESULTS.json", "w"), indent=1, default=float)
L = ["# Balance-mechanism probe: results (frozen analysis, run once)", "",
     f"units {R['n_units']}, error fits {n_err}, hosts {R['hosts']}", "", "## Primary (Holm over 2)"]
for k in ps:
    L.append(f"- {k}: p {ps[k]:.4g}, Holm {R['primary_holm'][k]:.4g} -> {'PASS' if R['primary_holm'][k] < 0.05 else 'FAIL'}")
L.append(f"  scatter per FM: " + "; ".join(f"{NAME[f]} {sc[f]['mean']:+.4f} ({sc[f]['n_pos']}/{sc[f]['n']} >0, p {sc[f]['p_one']:.3g})" for f in FMS))
h = R["primary"]["H_response"]
L.append(f"  response: within-table rho {h['rho_within_table']:.3f} over {h['n_tables']} tables, slope {h['slope_within']:.2f} {h['slope_ci']}, pooled rho {h['pooled_rho']:.3f}")
L += ["", "## Secondary (Holm over 3)"] + [f"- {k}: {R['secondary'][k]} Holm {R['secondary_holm'][k]:.4g}" for k in ss]
L += ["", "## Per FM (mean [95% bootstrap CI], 43 tables)"] + [f"- {desc[f]['name']}: scatter {desc[f]['scatter']}, shift {desc[f]['shift']}, balance {desc[f]['bal']}, efficiency {desc[f]['efficiency']:.2f}, mean lambda* {desc[f]['lstar_mean']:.2f}" for f in FMS]
L += ["", f"order: {R['order']}"]
(B / "results/RESULTS.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
