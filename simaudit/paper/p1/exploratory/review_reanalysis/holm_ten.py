"""R5: Holm adjustment of the T1' per-budget macro-F1 tests (stratified - random). EXPLORATORY.

Test = the paper's: two-sided Wilcoxon signed-rank over the 52 T1' tables of the per-table
gap (mean over reps of stratified minus mean over reps of random), per FM and m.
- Pre-registered FMs (fm1, fm2, fm3, fm5): out/bc_contrasts.csv (Study 12 rows; equals
  analysis.study12.per_budget). Families: (a) 4 FMs at m = 10; (b) 4 FMs x 4 budgets.
- Exploratory FMs (fm6 TabDPT v1.3, fm7 Mitra-v2): ../fm_extra/out/t1p_cells.csv (all 52
  tables, the source of \\ExFmxTpFM{six,seven}GapMTen); fm8 LimiX-16M:
  ../fm_extra/out/t1p_fm8_cells.csv (same draws, source of \\ExLimixTpGapMTen).
  Separate families: (a) 3 FMs at m = 10; (b) 3 FMs x 4 budgets.
Writes out/holm_ten.csv and out/rv_ten.tex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from rv_common import HERE, OUT, Macros, fmt, holm, pfmt

FMW = dict(fm1="FMone", fm2="FMtwo", fm3="FMthree", fm5="FMfive", fm6="FMsix", fm7="FMseven", fm8="FMeight")
MW = {1: "MOne", 3: "MThree", 5: "MFive", 10: "MTen"}


def main() -> None:
    bc = pd.read_csv(OUT / "bc_contrasts.csv")
    pre = bc[(bc.set == "t1prime") & (bc.comp == "random") & (bc.metric == "macro_f1") & bc.m.isin(["1", "3", "5", "10"])]
    pre = pre[["fm", "m", "n", "mean", "p_wilcoxon"]].assign(m=lambda d: d.m.astype(int), family="prereg")
    fmx = HERE.parent / "fm_extra" / "out"
    cells = pd.concat([pd.read_csv(fmx / "t1p_cells.csv"), pd.read_csv(fmx / "t1p_fm8_cells.csv")])
    rows = []
    for (fm, m), g in cells[cells.learner.isin(["fm6", "fm7", "fm8"])].groupby(["learner", "mult"]):
        x = g.gap.to_numpy()
        rows.append(dict(fm=fm, m=int(m), n=len(x), mean=x.mean(), p_wilcoxon=float(stats.wilcoxon(x).pvalue),
                         family="exploratory"))
    tab = pd.concat([pre, pd.DataFrame(rows)], ignore_index=True)
    tab["p_holm_m10"] = np.nan
    tab["p_holm_all_m"] = np.nan
    for fam, g in tab.groupby("family"):
        tab.loc[g.index, "p_holm_all_m"] = holm(g.p_wilcoxon.to_numpy())
        g10 = g[g.m == 10]
        tab.loc[g10.index, "p_holm_m10"] = holm(g10.p_wilcoxon.to_numpy())
    tab = tab.sort_values(["family", "fm", "m"])
    tab.to_csv(OUT / "holm_ten.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab.to_string(index=False))

    mac = Macros("ten")
    src = "holm_ten.csv (T1', S-R macro-F1, Wilcoxon two-sided over 52 tables"
    for r in tab[tab.m == 10].itertuples():
        w = FMW[r.fm]
        fam = "4 pre-registered FMs" if r.family == "prereg" else "3 exploratory FMs"
        mac.put(f"TpTenHolm{w}", fmt(r.mean, 3, sign=True), src + f"; {r.fm}, m=10, gap)")
        mac.put(f"TpTenHolm{w}Raw", pfmt(r.p_wilcoxon), src + f"; {r.fm}, m=10, unadjusted)")
        mac.put(f"TpTenHolm{w}FMs", pfmt(r.p_holm_m10), src + f"; {r.fm}, m=10, Holm over {fam} at m=10)")
        mac.put(f"TpTenHolm{w}All", pfmt(r.p_holm_all_m), src + f"; {r.fm}, m=10, Holm over {fam} x 4 budgets)")
    for fam, word in (("prereg", "Prereg"), ("exploratory", "Expl")):
        g = tab[(tab.family == fam) & (tab.m == 10)]
        mac.put(f"TpTenHolm{word}SurvFMs", str(int((g.p_holm_m10 < 0.05).sum())),
                src + f"; {fam}: FMs with Holm p<0.05, family = FMs at m=10, of {len(g)})")
        mac.put(f"TpTenHolm{word}SurvAll", str(int((g.p_holm_all_m < 0.05).sum())),
                src + f"; {fam}: FMs at m=10 with Holm p<0.05, family = FMs x 4 budgets, of {len(g)})")
        mac.put(f"TpTenHolm{word}N", str(len(g)), src + f"; {fam}: FMs in family)")
        ga = tab[tab.family == fam]
        mac.put(f"TpTenHolm{word}CellsSurv", str(int((ga.p_holm_all_m < 0.05).sum())),
                src + f"; {fam}: cells with Holm p<0.05 over FMs x 4 budgets, of {len(ga)})")
    sig = tab[(tab.family == "prereg") & (tab.m == 10) & (tab.p_holm_m10 < 0.05)]
    mac.put("TpTenHolmPreregSurvMin", fmt(sig["mean"].min(), 3, sign=True), src + "; surviving prereg FMs at m=10, min gap)")
    mac.put("TpTenHolmPreregSurvMax", fmt(sig["mean"].max(), 3, sign=True), src + "; surviving prereg FMs at m=10, max gap)")
    mac.put("TpTenHolmPreregMaxAdjSurv", pfmt(sig.p_holm_m10.max()), src + "; largest Holm p (FMs at m=10) among survivors)")
    mac.write()


if __name__ == "__main__":
    main()
