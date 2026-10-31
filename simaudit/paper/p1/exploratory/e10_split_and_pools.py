"""E10 (exploratory, R3-W13/R1-W3 and R3-W12): binary vs multiclass, and per-table pool characteristics.

(a) Split of ΔM (macro-F1) and ΔA (accuracy), stratified − random, per FM and m and
averaged over m, into the 14 binary and 6 multiclass DEV tables (stored Study 7
Part (a)): mean with bootstrap 95% CI (10k) and Wilcoxon within each group, and the
difference binary − multiclass with a two-sample bootstrap CI and a Mann–Whitney p.
With 6 multiclass tables every multiclass statement is weak.

(b) Per-table pool characteristics for make_tables.py:
- DEV (out/e10_pools_dev.csv): n, C, π_min, IR, TV, and per m ∈ {1,3,5,10} the share of
  the 20 stored Study 7 random draws that missed a class (results/study7/selections)
  and the exact probability of a miss (multivariate hypergeometric,
  analysis.study9.p_all_at_least_dp, read-only).
- T1′ (out/e10_pools_t1prime.csv): metadata from data/pool_t1prime.yaml (key, name,
  subtier, stratum), n, C, π_min, IR, TV from the loaded table, and per m the share of
  the 10 stored Study 9 random draws (TabICL; nested, drawn from the pool P that
  excludes the stop set) that missed a class, and the exact miss probability for draws
  from P. sealed_heldout/ is not read.

Writes out/e10_split.csv, out/e10_pools_dev.csv, out/e10_pools_t1prime.csv, out/E10_SUMMARY.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

import common as cm  # noqa: E402
from simaudit.analysis.study9 import p_all_at_least_dp  # noqa: E402
from simaudit.runner.study9 import dataset_key, layout, row_permutation  # noqa: E402


def split() -> pd.DataFrame:
    c = pd.read_csv(cm.OUT / "e6_cells.csv")
    avg = c.groupby(["dataset", "fm"]).agg(dM=("dM", "mean"), dA=("dA", "mean"), C=("C", "first")).reset_index()
    avg["mult"] = "all"
    c["mult"] = c.mult.astype(str)
    c = pd.concat([c, avg])
    rows = []
    rng = np.random.default_rng(0)
    for (fm, m), g in c.groupby(["fm", "mult"]):
        for metric in ("dM", "dA"):
            b, mc = g[g.C == 2][metric].to_numpy(), g[g.C > 2][metric].to_numpy()
            diffs = [rng.choice(b, len(b)).mean() - rng.choice(mc, len(mc)).mean() for _ in range(10000)]
            sb, sm = cm.summarize(b), cm.summarize(mc)
            rows.append(dict(fm=fm, mult=m, metric=metric,
                             bin_n=len(b), bin_mean=sb["mean"], bin_lo=sb["ci_lo"], bin_hi=sb["ci_hi"], bin_p=sb["p"],
                             mc_n=len(mc), mc_mean=sm["mean"], mc_lo=sm["ci_lo"], mc_hi=sm["ci_hi"], mc_p=sm["p"],
                             diff=b.mean() - mc.mean(), diff_lo=float(np.quantile(diffs, 0.025)),
                             diff_hi=float(np.quantile(diffs, 0.975)),
                             mw_p=float(stats.mannwhitneyu(b, mc).pvalue)))
    return pd.DataFrame(rows)


def p_miss(counts, b: int) -> float:
    return float(1 - p_all_at_least_dp(list(map(int, counts)), int(b), 1))


def pools_dev() -> pd.DataFrame:
    rows = []
    for did in cm.dev_ids():
        y = cm.dev_data(did).y
        tr = cm.pool_traits(y)
        counts = np.bincount(y)
        sel = pd.read_parquet(cm.S7 / "selections" / f"{did}_random.parquet")
        rec = dict(dataset=did, **tr)
        for m in cm.MULTS:
            s = sel[sel.mult == m]
            miss = [len(np.unique(y[np.asarray(lab, int)])) < tr["C"] for lab in s.labeled]
            rec[f"miss_obs_m{m}"] = float(np.mean(miss))
            rec[f"miss_exact_m{m}"] = p_miss(counts, m * tr["C"])
        rows.append(rec)
    return pd.DataFrame(rows)


def pools_t1prime() -> pd.DataFrame:
    meta = yaml.safe_load((cm.PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())["datasets"]
    rows = []
    for d in meta:
        did = int(d["did"])
        key = dataset_key(did)
        data = cm.load(did, cm.settings.data_cache_dir)
        y = data.y
        tr = cm.pool_traits(y)
        perm = row_permutation(len(y), did, cm.settings.base_seed)
        s, _ = layout(len(y))
        pool_counts = np.bincount(y[perm][s:], minlength=tr["C"])
        rec = dict(dataset=key, did=did, name=d["name"], subtier=d["subtier"], stratum=d["stratum"], **tr)
        res = pd.read_parquet(cm.settings.results_dir / "study9" / "t1prime" / f"{key}_fm2.parquet",
                              columns=["strategy", "m", "rep", "coverage"])
        r = res[res.strategy == "random"]
        for m in cm.MULTS:
            rm = r[r.m == m]
            rec[f"miss_obs_m{m}"] = float((rm.coverage < 1).mean()) if len(rm) else np.nan
            b = m * tr["C"]
            rec[f"miss_exact_m{m}"] = p_miss(pool_counts, b) if b <= pool_counts.sum() // 1 and len(rm) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    sp = split()
    sp.to_csv(cm.OUT / "e10_split.csv", index=False)
    dev = pools_dev()
    dev.to_csv(cm.OUT / "e10_pools_dev.csv", index=False)
    t1 = pools_t1prime()
    t1.to_csv(cm.OUT / "e10_pools_t1prime.csv", index=False)
    L = ["# E10: binary vs multiclass split and per-table pool characteristics (exploratory)", "",
         "## (a) DEV split (14 binary, 6 multiclass tables); stratified − random, mean [bootstrap 95% CI]", "",
         "`*` Wilcoxon p < 0.05 within the group (unadjusted). diff = binary − multiclass [two-sample bootstrap CI], Mann–Whitney p.", ""]
    for metric, name in (("dM", "macro-F1"), ("dA", "accuracy")):
        L += [f"### {name}", "", "| FM | m | binary | multiclass | diff (MW p) |", "|---|---|---|---|---|"]
        for fm in cm.FMS:
            for m in ("1", "3", "5", "10", "all"):
                r = sp[(sp.fm == fm) & (sp.mult == m) & (sp.metric == metric)].iloc[0]
                L.append(f"| {cm.FM_NAMES[fm]} | {m} | {r.bin_mean:+.3f} [{r.bin_lo:+.3f}, {r.bin_hi:+.3f}]{' *' if r.bin_p < 0.05 else ''} | "
                         f"{r.mc_mean:+.3f} [{r.mc_lo:+.3f}, {r.mc_hi:+.3f}]{' *' if r.mc_p < 0.05 else ''} | "
                         f"{r['diff']:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}] ({r.mw_p:.2g}) |")
        L.append("")
    L += ["## (b) DEV pools (observed share of the 20 random draws that missed a class / exact probability)", "",
          dev.round(3).to_markdown(index=False), "",
          f"## (b) T1′ pools ({len(t1)} tables; Study 9 random draws, 10 per m, TabICL file)", "",
          f"Binary: {int((t1.C == 2).sum())}, multiclass: {int((t1.C > 2).sum())}. Median n {t1.n.median():.0f}, "
          f"median π_min {t1.pi_min.median():.3f}. Mean observed miss share at m=1: {t1.miss_obs_m1.mean():.3f} "
          f"(exact {t1.miss_exact_m1.mean():.3f}); at m=3: {t1.miss_obs_m3.mean():.3f}.", "",
          "Full table: out/e10_pools_t1prime.csv"]
    (cm.OUT / "E10_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
