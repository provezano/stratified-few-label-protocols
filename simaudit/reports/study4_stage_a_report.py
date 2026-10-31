import glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from simaudit.analysis import study4 as a4
from simaudit.data.openml_loader import load
from simaudit.settings import settings
raw = pd.concat([pd.read_parquet(f) for f in glob.glob(str(settings.results_dir / "study4_stage_a/*.parquet"))], ignore_index=True)
truth = {int(d): load(int(d), settings.data_cache_dir).y for d in raw.dataset.unique()}
fm = raw[raw.learner == "fm5"]
out = []
def tbl(df, metric, title):
    t = a4.tests_table(df, metric)
    out.append(f"\n### {title}: Δ vs realistic random ({metric}), mean [95% CI], wins/20, p_holm\n")
    rows = []
    for _, r in t.iterrows():
        rows.append(dict(strategy=r.strategy, mult=r.mult, cell=f"{r.mean_delta:+.3f} [{r.ci_low:+.3f},{r.ci_high:+.3f}] {r.wins}/{r.n} p={r.p_holm:.3f}"))
    out.append(pd.DataFrame(rows).pivot(index="strategy", columns="mult", values="cell").to_markdown())
    return t
tbl(fm, "macro_f1", "FM5")
tbl(fm, "accuracy", "FM5")
# random baseline absolute and coverage
base = fm.groupby(["mult", "strategy"])[["macro_f1", "accuracy", "coverage"]].mean().unstack(0).round(3)
out.append("\n### Absolute means over datasets (FM5)\n"); out.append(base.to_markdown())
risk = a4.risk_table(fm, "macro_f1")
out.append("\n### Risk (FM5, macro-F1): mean across-draw std, mean 10% quantile, worst-dataset regret vs random\n")
out.append(risk.drop(columns="learner").round(3).to_markdown(index=False))
orc = pd.concat([a4.split_half_oracle(fm[fm.mult == k], truth, "macro_f1", seed=k).assign(mult=k) for k in sorted(fm.mult.unique())])
o = orc.groupby("mult")[["mean_random", "split_half", "in_sample_best"]].mean()
o["realistic_headroom"] = o.split_half - o.mean_random
o["in_sample_headroom"] = o.in_sample_best - o.mean_random
out.append("\n### RQ-L4 headroom of best-of-10 random (FM5, macro-F1; split-half is honest)\n"); out.append(o.round(3).to_markdown())
sens = a4.rescore_on(fm, truth, "macro_f1")
sm = sens.groupby(["dataset", "mult", "strategy"]).macro_f1_common.mean().unstack("strategy")
sd = sm.sub(sm["random"], axis=0).drop(columns="random").groupby("mult").mean()
out.append("\n### Sensitivity: Δ vs random scored only on rows no strategy selected (same rep), macro-F1\n"); out.append(sd.T.round(3).to_markdown())
ll = raw.pivot_table(index=["dataset", "mult", "strategy", "rep"], columns="learner", values="macro_f1")
ll["d"] = ll.llgc_fm5 - ll.fm5
g = ll.groupby(["mult", "strategy"]).d.agg(["mean", lambda x: x.quantile(0.1)]).round(4)
g.columns = ["mean Δ(LLGC−FM5)", "q10"]
out.append("\n### RQ-L3: LLGC on the FM5 prior minus FM5 (macro-F1), by strategy\n"); out.append(g.unstack(0).to_markdown())
print("\n".join(out))
