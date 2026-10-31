"""Round-3 CRITICAL #5 and #4 (EXPLORATORY re-analyses of stored pre-registered fits).

(1) T1' accuracy, stratified - random, scoped by pool imbalance. Tables split at
    TV = 0.1 (TV = 1/2 ||pi - u||_1 of the full table; the cut-off is POST HOC, chosen
    after seeing the T1' accuracy results): 30 tables with TV >= 0.1, 22 with TV < 0.1.
    Per FM x m and m-averaged: mean, bootstrap 95% CI, Wilcoxon and paired t (table unit),
    Holm within FM over the 4 budgets. Cut-off-free: Spearman rho(Delta A, TV) over all 52
    tables with a two-sided permutation p (10,000 permutations of TV over tables).
(2) Study 7 Part (c) dose-response with the table as the unit: per (table, m) the OLS
    slope of macro-F1 (raw and after p/pi-hat) on the share s over the rep-level fits,
    averaged over feasible m; mean over tables, bootstrap 95% CI, Wilcoxon over tables,
    Holm over the 8 FM x arm cells. Checked against E12 (out/e12_slopes.csv).
Writes out/acc_imb.csv, out/acc_imb_rho.csv, out/dose_table_slopes.csv, out/dose_table.csv,
out/rv_accdose.tex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from rv_common import FMS, FMW, MW, OUT, Macros, fmt, holm, pfmt
from simaudit.analysis.report import bootstrap_ci
from simaudit.settings import settings
import bc_contrasts as bc

MULTS = (1, 3, 5, 10)


def accuracy_scoped():
    t1 = bc.load_t1()
    c = bc.cell(t1)
    tv = pd.read_csv(OUT / "bc_t1_pool_tv.csv").set_index("dataset")
    rows, rho = [], []
    for fm in FMS:
        for m in MULTS + ("avg",):
            ms = MULTS if m == "avg" else (m,)
            x, _ = bc.diff(c, fm, "stratified", "random", ms, "accuracy")
            j = tv.join(x.rename("dA"), how="inner")
            for grp, sub in (("imb", j[j.tv >= 0.1]), ("bal", j[j.tv < 0.1])):
                v = sub.dA.to_numpy()
                lo, hi = bootstrap_ci(v)
                rows.append(dict(fm=fm, m=str(m), group=grp, n=len(v), mean=v.mean(), ci_lo=lo, ci_hi=hi,
                                 p_wilcoxon=float(stats.wilcoxon(v).pvalue), p_t=float(stats.ttest_1samp(v, 0).pvalue)))
            r = stats.spearmanr(j.tv, j.dA)[0]
            rng = np.random.default_rng(0)
            perm = np.array([stats.spearmanr(rng.permutation(j.tv.to_numpy()), j.dA)[0] for _ in range(10_000)])
            rho.append(dict(fm=fm, m=str(m), n=len(j), rho=r, p_perm=float((np.sum(np.abs(perm) >= abs(r)) + 1) / 10_001)))
    tab = pd.DataFrame(rows)
    for col in ("p_wilcoxon", "p_t"):
        tab[col + "_holm"] = np.nan
        for _, g in tab[tab.m != "avg"].groupby(["fm", "group"]):
            tab.loc[g.index, col + "_holm"] = holm(g[col].to_numpy())
    rho = pd.DataFrame(rho)
    return tab, rho


def dose_table():
    fr = []
    for fm in FMS:
        for p in sorted((settings.results_dir / "study7" / "c").glob(f"*_{fm}.parquet")):
            fr.append(pd.read_parquet(p, columns=["learner", "dataset", "arm", "share", "mult", "rep", "feasible",
                                                  "macro_f1", "pc_macro_f1"]))
    c = pd.concat(fr, ignore_index=True)
    c = c[c.feasible & c.arm.isin(["majority", "minority"])]
    rows = []
    for (fm, arm, did), g in c.groupby(["learner", "arm", "dataset"]):
        rec = dict(fm=fm, arm=arm, dataset=did)
        for met in ("macro_f1", "pc_macro_f1"):
            sl = [float(np.polyfit(gm.share.to_numpy(float), gm[met].to_numpy(float), 1)[0])
                  for _, gm in g.groupby("mult") if gm.share.nunique() >= 2]
            rec[met] = float(np.mean(sl)) if sl else np.nan
        rows.append(rec)
    per = pd.DataFrame(rows)
    out = []
    for (fm, arm), g in per.groupby(["fm", "arm"]):
        for met in ("macro_f1", "pc_macro_f1"):
            x = g[met].dropna().to_numpy()
            lo, hi = bootstrap_ci(x)
            out.append(dict(fm=fm, arm=arm, metric=met, n=len(x), mean=x.mean(), ci_lo=lo, ci_hi=hi,
                            p_wilcoxon=float(stats.wilcoxon(x).pvalue), share_neg=float(np.mean(x < 0))))
    tab = pd.DataFrame(out)
    tab["p_holm8"] = np.nan
    for _, g in tab.groupby("metric"):
        tab.loc[g.index, "p_holm8"] = holm(g.p_wilcoxon.to_numpy())
    # agreement with E12
    e12 = pd.read_csv(OUT.parents[1] / "out" / "e12_slopes.csv")
    e12 = e12[e12.quantity.isin(["slope_macro_f1", "slope_pc_macro_f1"])].assign(
        metric=lambda d: d.quantity.str.replace("slope_", ""))
    mg = tab.merge(e12[["fm", "arm", "metric", "mean"]], on=["fm", "arm", "metric"], suffixes=("", "_e12"))
    return per, tab, float((mg["mean"] - mg["mean_e12"]).abs().max())


def main():
    pd.set_option("display.width", 220)
    acc, rho = accuracy_scoped()
    acc.to_csv(OUT / "acc_imb.csv", index=False)
    rho.to_csv(OUT / "acc_imb_rho.csv", index=False)
    print(acc.round(4).to_string())
    print(rho.round(4).to_string())
    per, dose, dmax = dose_table()
    per.to_csv(OUT / "dose_table_slopes.csv", index=False)
    dose.to_csv(OUT / "dose_table.csv", index=False)
    print(dose.round(5).to_string())
    print("max |mean slope - E12|:", dmax)

    mac = Macros("accdose")
    GW = dict(imb="Imb", bal="Bal")
    for r in acc.itertuples():
        mw = "Avg" if r.m == "avg" else MW[int(r.m)]
        stem = f"TpAccImb{GW[r.group]}{FMW[r.fm]}{mw}"
        src = (f"acc_imb.csv (T1', S-R accuracy, {r.fm}, m={r.m}, {'TV>=0.1' if r.group == 'imb' else 'TV<0.1'} "
               f"(post hoc cut-off), n={r.n}")
        mac.put(stem, fmt(r.mean, 3, sign=True), src + ")")
        mac.put(stem + "Lo", fmt(r.ci_lo, 3, sign=True), src + ", bootstrap 95% CI)")
        mac.put(stem + "Hi", fmt(r.ci_hi, 3, sign=True), src + ", bootstrap 95% CI)")
        mac.put(stem + "WP", pfmt(r.p_wilcoxon), src + ", Wilcoxon, unadjusted)")
        mac.put(stem + "TP", pfmt(r.p_t), src + ", paired t, unadjusted)")
        if r.m != "avg":
            mac.put(stem + "WHolm", pfmt(r.p_wilcoxon_holm), src + ", Wilcoxon, Holm within FM over 4 m)")
            mac.put(stem + "THolm", pfmt(r.p_t_holm), src + ", paired t, Holm within FM over 4 m)")
    for grp in ("imb", "bal"):
        a = acc[(acc.group == grp) & (acc.m == "avg")]
        mac.put(f"TpAccImb{GW[grp]}N", str(int(a.n.iloc[0])), "acc_imb.csv (tables in group; post hoc cut-off TV=0.1)")
        mac.put(f"TpAccImb{GW[grp]}AvgMin", fmt(a["mean"].min(), 3, sign=True), "acc_imb.csv (m-averaged, min over FMs)")
        mac.put(f"TpAccImb{GW[grp]}AvgMax", fmt(a["mean"].max(), 3, sign=True), "acc_imb.csv (m-averaged, max over FMs)")
        mac.put(f"TpAccImb{GW[grp]}AvgWPMax", pfmt(a.p_wilcoxon.max()), "acc_imb.csv (m-averaged, Wilcoxon, max over FMs)")
        mac.put(f"TpAccImb{GW[grp]}AvgWPMin", pfmt(a.p_wilcoxon.min()), "acc_imb.csv (m-averaged, Wilcoxon, min over FMs)")
        mac.put(f"TpAccImb{GW[grp]}AvgTPMax", pfmt(a.p_t.max()), "acc_imb.csv (m-averaged, paired t, max over FMs)")
        b = acc[(acc.group == grp) & (acc.m != "avg")]
        mac.put(f"TpAccImb{GW[grp]}CellsWHolmSig", str(int((b.p_wilcoxon_holm < 0.05).sum())),
                "acc_imb.csv (cells of 16 with Wilcoxon Holm-within-FM p<0.05)")
        mac.put(f"TpAccImb{GW[grp]}CellsTHolmSig", str(int((b.p_t_holm < 0.05).sum())),
                "acc_imb.csv (cells of 16 with paired-t Holm-within-FM p<0.05)")
    for r in rho.itertuples():
        mw = "Avg" if r.m == "avg" else MW[int(r.m)]
        src = f"acc_imb_rho.csv (T1', Spearman rho(Delta A, TV), {r.fm}, m={r.m}, n={r.n}"
        mac.put(f"TpAccImbRho{FMW[r.fm]}{mw}", fmt(r.rho, 2, sign=True), src + ")")
        mac.put(f"TpAccImbRho{FMW[r.fm]}{mw}PermP", pfmt(r.p_perm), src + ", permutation p, 10,000)")
    ra = rho[rho.m == "avg"]
    mac.put("TpAccImbRhoAvgMin", fmt(ra.rho.min(), 2, sign=True), "acc_imb_rho.csv (m-averaged, min over FMs)")
    mac.put("TpAccImbRhoAvgMax", fmt(ra.rho.max(), 2, sign=True), "acc_imb_rho.csv (m-averaged, max over FMs)")
    mac.put("TpAccImbRhoAvgPermPMax", pfmt(ra.p_perm.max()), "acc_imb_rho.csv (m-averaged, permutation p, max over FMs)")
    MT = dict(macro_f1="Raw", pc_macro_f1="Pc")
    AW = dict(majority="Majority", minority="Minority")
    for r in dose.itertuples():
        stem = f"DoseTbl{FMW[r.fm]}{AW[r.arm]}{MT[r.metric]}"
        src = f"dose_table.csv (Study 7 Part (c) DEV, {r.fm}, {r.arm}, {r.metric}, per-table slope on s, n={r.n} tables"
        mac.put(stem, fmt(r.mean, 2, sign=True), src + ")")
        mac.put(stem + "Lo", fmt(r.ci_lo, 2, sign=True), src + ", bootstrap 95% CI)")
        mac.put(stem + "Hi", fmt(r.ci_hi, 2, sign=True), src + ", bootstrap 95% CI)")
        mac.put(stem + "P", pfmt(r.p_wilcoxon), src + ", Wilcoxon over tables, unadjusted)")
        mac.put(stem + "Holm", pfmt(r.p_holm8), src + ", Wilcoxon, Holm over 8 FM x arm cells)")
    for met, w in MT.items():
        d = dose[dose.metric == met]
        mac.put(f"DoseTbl{w}SlopeMin", fmt(d["mean"].min(), 2, sign=True), f"dose_table.csv ({met}, min over 8 cells)")
        mac.put(f"DoseTbl{w}SlopeMax", fmt(d["mean"].max(), 2, sign=True), f"dose_table.csv ({met}, max over 8 cells)")
        mac.put(f"DoseTbl{w}HolmMax", pfmt(d.p_holm8.max()), f"dose_table.csv ({met}, largest Holm p over 8 cells)")
        mac.put(f"DoseTbl{w}NegCells", str(int((d.ci_hi < 0).sum())), f"dose_table.csv ({met}, cells with CI below 0, of 8)")
    mac.put("DoseTblN", str(int(dose.n.max())), "dose_table.csv (tables per cell)")
    mac.write()


if __name__ == "__main__":
    main()
