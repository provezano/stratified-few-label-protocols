"""(b) review item 8 and (c) item 10, plus (d1) Holm on DEV m = 10. EXPLORATORY.

Stored outputs only: results/study7/a (DEV, n = 20) and results/study12 (T1', n = 52).
Unit = table; per-table value = mean over reps of each arm, difference of arm means.
(b) S - R and S - redraw, macro-F1 and accuracy, per FM x m and averaged.
(c) S - R accuracy: paired t, centred-bootstrap p, Wilcoxon; Holm within FM over 4 m;
    T1' accuracy gap vs pool TV (E6 approach) and binary sign agreement with B*.
Writes out/bc_*.csv, out/rv_bc.tex.
"""

from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
from scipy import stats

from rv_common import FMS, FMW, MW, OUT, Macros, fmt, holm, pfmt
from simaudit.analysis.report import bootstrap_ci
from simaudit.data.openml_loader import load
from simaudit.runner.study9 import dataset_key
from simaudit.settings import PROJECT_ROOT, settings

MULTS = (1, 3, 5, 10)
METRICS = ("macro_f1", "accuracy")
COLS = ["dataset", "learner", "strategy", "mult", "rep", "coverage", *METRICS]


def load_dev() -> pd.DataFrame:
    fr = []
    for p in sorted((settings.results_dir / "study7" / "a").glob("*_fm*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d = d[d.strategy.isin(["stratified", "random"])]
        cov = d[(d.strategy == "random") & (d.coverage >= 1.0)].copy()
        cov["strategy"] = "redraw"
        fr += [d, cov]
    out = pd.concat(fr, ignore_index=True)
    out["dataset"] = out.dataset.astype(str)
    return out


def load_t1() -> pd.DataFrame:
    fr = [pd.read_parquet(p, columns=COLS) for p in sorted((settings.results_dir / "study12").glob("*_fm*.parquet"))]
    out = pd.concat(fr, ignore_index=True)
    out["dataset"] = out.dataset.astype(str)
    return out


def boot_p(x: np.ndarray, n: int = 10_000, seed: int = 0) -> float:
    """Two-sided centred-bootstrap p for mean = 0 (resampling tables)."""
    rng = np.random.default_rng(seed)
    xc = x - x.mean()
    bm = rng.choice(xc, size=(n, x.size)).mean(axis=1)
    return float((np.sum(np.abs(bm) >= abs(x.mean())) + 1) / (n + 1))


def wil(x):
    return float(stats.wilcoxon(x).pvalue) if np.any(x != 0) else 1.0


def summ(x: np.ndarray) -> dict:
    lo, hi = bootstrap_ci(x)
    return dict(n=len(x), mean=float(x.mean()), ci_lo=lo, ci_hi=hi, median=float(np.median(x)),
                p_wilcoxon=wil(x), p_t=float(stats.ttest_1samp(x, 0).pvalue), p_boot=boot_p(x))


def cell(runs):
    return runs.groupby(["dataset", "learner", "strategy", "mult"])[list(METRICS)].mean()


def diff(c, fm, a, b, mults, metric):
    s = c[metric].unstack("strategy")
    s = s.xs(fm, level="learner")
    s = s[s.index.get_level_values("mult").isin(mults)]
    d = (s[a] - s[b])
    # per-table mean over the stated m, complete cells only
    return d.groupby(level="dataset").mean().dropna(), d.dropna()


PROV = {
    ("t1prime", "random", "macro_f1", "avg"): "pre-registered primary (i), Holm over 6",
    ("t1prime", "random", "macro_f1", "m"): "pre-registered robustness, unadjusted",
    ("t1prime", "random", "accuracy", "avg"): "pre-registered secondary, unadjusted",
    ("t1prime", "random", "accuracy", "m"): "pre-registered secondary (per budget), unadjusted",
    ("t1prime", "redraw", "macro_f1", "avg"): "pre-declared secondary S-1(a), Holm over 4 FMs",
    ("t1prime", "redraw", "macro_f1", "m"): "pre-declared secondary S-1 (descriptive per m)",
    ("t1prime", "redraw", "accuracy", "avg"): "exploratory",
    ("t1prime", "redraw", "accuracy", "m"): "exploratory",
    ("dev", "random", "macro_f1", "avg"): "pre-registered primary (i), Holm over 15",
    ("dev", "random", "macro_f1", "m"): "pre-specified exploratory (Study 7 Sec. 4.3)",
    ("dev", "random", "accuracy", "avg"): "pre-specified exploratory (Study 7 Sec. 4.3)",
    ("dev", "random", "accuracy", "m"): "pre-specified exploratory (Study 7 Sec. 4.3)",
    ("dev", "redraw", "macro_f1", "avg"): "exploratory (not pre-registered)",
    ("dev", "redraw", "macro_f1", "m"): "exploratory (not pre-registered)",
    ("dev", "redraw", "accuracy", "avg"): "exploratory (not pre-registered)",
    ("dev", "redraw", "accuracy", "m"): "exploratory (not pre-registered)",
}


def contrasts(sets) -> pd.DataFrame:
    rows = []
    for set_name, runs in sets.items():
        c = cell(runs)
        for comp in ("random", "redraw"):
            for metric in METRICS:
                for fm in FMS:
                    for m in MULTS:
                        x, _ = diff(c, fm, "stratified", comp, (m,), metric)
                        rows.append(dict(set=set_name, comp=comp, metric=metric, fm=fm, m=str(m),
                                         provenance=PROV[(set_name, comp, metric, "m")], **summ(x.to_numpy())))
                    avg_m = (3, 5, 10) if comp == "redraw" else MULTS
                    x, _ = diff(c, fm, "stratified", comp, avg_m, metric)
                    rows.append(dict(set=set_name, comp=comp, metric=metric, fm=fm,
                                     m="avg" + "".join(map(str, avg_m)),
                                     provenance=PROV[(set_name, comp, metric, "avg")], **summ(x.to_numpy())))
    tab = pd.DataFrame(rows)
    # Holm within FM over the 4 budgets, per set x comp x metric, for each test
    for t in ("p_wilcoxon", "p_t", "p_boot"):
        tab[t + "_holm"] = np.nan
        per = tab[tab.m.isin([str(m) for m in MULTS])]
        for _, g in per.groupby(["set", "comp", "metric", "fm"]):
            tab.loc[g.index, t + "_holm"] = holm(g[t].to_numpy())
    return tab


def pool_tv(dids) -> pd.DataFrame:
    rows = []
    for did in dids:
        y = load(did, settings.data_cache_dir).y.astype(np.int64)
        cnt = np.bincount(y)
        pi = cnt / cnt.sum()
        C = len(cnt)
        rows.append(dict(dataset=dataset_key(did), C=C, tv=float(np.abs(pi - 1 / C).sum() / 2),
                         pi_min=float(pi.min())))
    return pd.DataFrame(rows)


def spearman_ci(x, y, n_boot=5000, seed=0):
    rho = stats.spearmanr(x, y)[0]
    rng = np.random.default_rng(seed)
    b = []
    for idx in rng.integers(0, len(x), size=(n_boot, len(x))):
        if np.ptp(x[idx]) > 0 and np.ptp(y[idx]) > 0:
            b.append(stats.spearmanr(x[idx], y[idx])[0])
    return float(rho), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def imbalance(t1: pd.DataFrame) -> pd.DataFrame:
    ids = [int(d["did"]) for d in yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())["datasets"]]
    tv = pool_tv(ids).set_index("dataset")
    tv.to_csv(OUT / "bc_t1_pool_tv.csv")
    c = cell(t1)
    rows = []
    for fm in FMS:
        for m in MULTS + ("avg",):
            ms = MULTS if m == "avg" else (m,)
            x, _ = diff(c, fm, "stratified", "random", ms, "accuracy")
            j = tv.join(x.rename("dA"), how="inner")
            rho, lo, hi = spearman_ci(j.tv.to_numpy(), j.dA.to_numpy())
            bal = j[j.tv < 0.1].dA
            imb = j[j.tv >= 0.1].dA
            row = dict(fm=fm, m=str(m), n=len(j), rho_tv=rho, rho_lo=lo, rho_hi=hi,
                       n_bal=len(bal), mean_bal=bal.mean(), n_imb=len(imb), mean_imb=imb.mean(),
                       p_mw_bal_vs_imb=float(stats.mannwhitneyu(bal, imb).pvalue))
            if m != "avg":
                b = j[j.C == 2].copy()
                p1 = b.pi_min
                tau = np.log((1 - p1) / p1)
                with np.errstate(divide="ignore"):
                    bstar = 1 / (p1 * (1 - p1) * tau ** 2)
                B = 2 * m
                dec = b.dA.abs() > 0.01
                pred = np.sign(bstar - B)
                row.update(bin_cells=int(dec.sum()),
                           bin_agree=float((np.sign(b.dA[dec]) == pred[dec]).mean()) if dec.any() else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    dev, t1 = load_dev(), load_t1()
    reps = pd.concat([dev.assign(set="dev"), t1.assign(set="t1prime")]).groupby(
        ["set", "strategy"]).apply(lambda g: g.groupby(["dataset", "learner", "mult"]).rep.nunique().agg(["min", "max"]).to_dict())
    print("reps per cell:", reps.to_dict())
    tab = contrasts({"dev": dev, "t1prime": t1})
    tab.to_csv(OUT / "bc_contrasts.csv", index=False)
    imb = imbalance(t1)
    imb.to_csv(OUT / "bc_t1_accuracy_imbalance.csv", index=False)
    # coverage share of random draws per set x m (fm2 rows; selections shared)
    cov = pd.concat([d[(d.strategy == "random") & (d.learner == "fm2")].assign(set=s)
                     for s, d in (("dev", dev), ("t1prime", t1))]).groupby(["set", "mult"]).coverage.apply(
        lambda v: float((v >= 1).mean()))
    cov.to_csv(OUT / "bc_covered_share.csv")
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string())
    print(imb.round(3).to_string())
    print(cov)
    write_macros(tab, imb, cov)


def write_macros(tab, imb, cov) -> None:
    mac = Macros("bc")
    SW = dict(dev="Dev", t1prime="Tp")
    CW = dict(random="SR", redraw="SRd")
    MTW = dict(macro_f1="F", accuracy="Acc")
    for r in tab.itertuples():
        mw = MW.get(int(r.m), None) if r.m.isdigit() else "Avg"
        stem = f"{SW[r.set]}{CW[r.comp]}{MTW[r.metric]}{FMW[r.fm]}{mw}"
        src = f"bc_contrasts.csv ({r.set}, S-{r.comp}, {r.metric}, {r.fm}, m={r.m}; {r.provenance})"
        mac.put(stem, fmt(r.mean, 3, sign=True), src)
        mac.put(stem + "Lo", fmt(r.ci_lo, 3, sign=True), src + ", bootstrap 95% CI")
        mac.put(stem + "Hi", fmt(r.ci_hi, 3, sign=True), src + ", bootstrap 95% CI")
        mac.put(stem + "P", pfmt(r.p_wilcoxon), src + ", Wilcoxon, unadjusted")
        if r.metric == "accuracy" and r.comp == "random":
            mac.put(stem + "TP", pfmt(r.p_t), src + ", paired t, unadjusted")
            mac.put(stem + "BootP", pfmt(r.p_boot), src + ", centred bootstrap, unadjusted")
            if r.m.isdigit():
                mac.put(stem + "THolm", pfmt(r.p_t_holm), src + ", paired t, Holm within FM over 4 m")
                mac.put(stem + "WHolm", pfmt(r.p_wilcoxon_holm), src + ", Wilcoxon, Holm within FM over 4 m")
        if r.set == "dev" and r.comp == "random" and r.metric == "macro_f1" and r.m == "10":
            mac.put(stem + "WHolm", pfmt(r.p_wilcoxon_holm), src + " (Holm here = over 4 m within FM)")
    # (d1) DEV m = 10 macro-F1, Holm over the four FMs
    d10 = tab[(tab.set == "dev") & (tab.comp == "random") & (tab.metric == "macro_f1") & (tab.m == "10")]
    adj = holm(d10.p_wilcoxon.to_numpy())
    for (r, a) in zip(d10.itertuples(), adj):
        mac.put(f"DevTenFHolmFMs{FMW[r.fm]}", pfmt(a), "bc_contrasts.csv (dev, S-R macro-F1, m=10, Wilcoxon, Holm over 4 FMs)")
    pd.DataFrame(dict(fm=d10.fm, mean=d10["mean"], p=d10.p_wilcoxon, p_holm_4fms=adj)).to_csv(OUT / "d1_dev_m10_holm.csv", index=False)
    print(pd.read_csv(OUT / "d1_dev_m10_holm.csv"))
    # ranges for the T1' accuracy text
    t1a = tab[(tab.set == "t1prime") & (tab.comp == "random") & (tab.metric == "accuracy") & tab.m.isin(["3", "5", "10"])]
    mac.put("TpAccSigTHolmN", str(int((t1a.p_t_holm < 0.05).sum())), "bc_contrasts.csv (t1prime accuracy S-R, m 3-10, paired t Holm<0.05, of 12 cells)")
    mac.put("TpAccSigWHolmN", str(int((t1a.p_wilcoxon_holm < 0.05).sum())), "bc_contrasts.csv (t1prime accuracy S-R, m 3-10, Wilcoxon Holm<0.05, of 12 cells)")
    mac.put("TpAccNegCiN", str(int((t1a.ci_hi < 0).sum())), "bc_contrasts.csv (t1prime accuracy S-R, m 3-10, CI below 0, of 12 cells)")
    for r in imb.itertuples():
        mw = "Avg" if r.m == "avg" else MW[int(r.m)]
        src = f"bc_t1_accuracy_imbalance.csv ({r.fm}, m={r.m})"
        stem = f"TpAccTv{FMW[r.fm]}{mw}"
        mac.put(stem + "Rho", fmt(r.rho_tv, 2, sign=True), src + ", Spearman")
        mac.put(stem + "RhoLo", fmt(r.rho_lo, 2, sign=True), src + ", bootstrap 95% CI")
        mac.put(stem + "RhoHi", fmt(r.rho_hi, 2, sign=True), src + ", bootstrap 95% CI")
        if r.m == "avg":
            mac.put(stem + "Bal", fmt(r.mean_bal, 3, sign=True), src + ", mean on TV<0.1")
            mac.put(stem + "Imb", fmt(r.mean_imb, 3, sign=True), src + ", mean on TV>=0.1")
    mac.put("TpAccTvBalN", str(int(imb.n_bal.iloc[0])), "bc_t1_accuracy_imbalance.csv (tables with TV<0.1)")
    mac.put("TpAccTvImbN", str(int(imb.n_imb.iloc[0])), "bc_t1_accuracy_imbalance.csv (tables with TV>=0.1)")
    bm = imb[imb.m != "avg"]
    mac.put("TpAccBinAgreeMin", fmt(bm.bin_agree.min(), 2), "bc_t1_accuracy_imbalance.csv (binary sign agreement with B*, min over FM x m)")
    mac.put("TpAccBinAgreeMax", fmt(bm.bin_agree.max(), 2), "bc_t1_accuracy_imbalance.csv (binary sign agreement with B*, max over FM x m)")
    for (s, m), v in cov.items():
        mac.put(f"{dict(dev='Dev', t1prime='Tp')[s]}Covered{MW[m]}", fmt(v, 2), f"bc_covered_share.csv ({s}, m={m})")
    mac.write()


if __name__ == "__main__":
    main()
