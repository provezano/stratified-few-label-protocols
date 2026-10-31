"""Round-14 check (EXPLORATORY re-analysis of stored pre-registered fits; no new fits): the inference chain
of the Study 7 mechanism rule with the table as the unit.

Pre-registered rule (prereg/STUDY7, "Competing alternative"), per FM over both arms (majority, minority):
"beyond a prior shift" if (a) the corrected (p / pi-hat) trend is one-sided significant (stratified JT,
p < 0.05, both arms) and (b) the mean corrected drop from s = 0.5 to the largest feasible level is at least
half of the uncorrected drop; "a prior shift" if the corrected drop is TOST-equivalent to 0 (margin 0.01);
otherwise "unresolved".

The JT test uses draws and budgets within a table as independent units. The valid chain here:
(a') per table, the OLS slope of corrected macro-F1 on s (per budget mult on the share means, averaged over
     mult), one-sided Wilcoxon signed-rank over tables (H1: median slope < 0) and a one-sided sign test,
     unadjusted and Holm over the 8 FM x arm cells; bootstrap 95% CI of the mean slope over tables;
(b') the corrected/raw drop ratio (point estimate exactly as the pre-registered report: mean over
     (table, mult) of first-minus-last feasible share, corrected over raw), with a percentile bootstrap 95% CI
     resampling tables (the ratio of means equals sum(pc drops) / sum(raw drops) over the resampled rows);
(c') the prior-shift branch: paired-t TOST (margin 0.01) of the per-table corrected drop (mean over mult).
Verdict per FM: "beyond" if (a') passes (Holm p < 0.05) in both arms and the ratio CI lies above 0.5 in
both arms; "prior shift" if the TOST passes in both arms; "not beyond" if the ratio CI lies below 0.5 in
some arm (and no prior shift); otherwise "inconclusive". Both scorings: the pre-registered complement
(comp) and the rows common to all compositions (com; r11 sensitivity).
Input: out/r11_partc_common.parquet (written by r11_partc_common.py; reproduces the stored metrics).
Writes out/r14_mech_chain.csv, out/r14_verdicts.csv, out/rv_r14.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r14_mech_chain.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from rv_common import FMS, FMW, OUT, Macros, fmt, holm, pfmt
from simaudit.analysis.report import bootstrap_ci
from simaudit.analysis.study7 import tost_p

SEED, NBOOT, THR = 14, 10_000, 0.5
ARMS = ("majority", "minority")
SC = ("comp", "com")


def per_table(g: pd.DataFrame, sc: str) -> pd.DataFrame:
    """Per table: summed raw / corrected drops over mult (and their count), mean corrected drop, slopes."""
    rows = []
    for did, gd in g.groupby("dataset"):
        rd, pd_, sl_r, sl_p = [], [], [], []
        for _, gm in gd.groupby("mult"):
            m = gm.groupby("share")[[f"raw_{sc}", f"pc_{sc}"]].mean().sort_index()
            if len(m) < 2:
                continue
            rd.append(m[f"raw_{sc}"].iloc[0] - m[f"raw_{sc}"].iloc[-1])
            pd_.append(m[f"pc_{sc}"].iloc[0] - m[f"pc_{sc}"].iloc[-1])
            x = m.index.to_numpy(float)
            sl_r.append(np.polyfit(x, m[f"raw_{sc}"].to_numpy(), 1)[0])
            sl_p.append(np.polyfit(x, m[f"pc_{sc}"].to_numpy(), 1)[0])
        if rd:
            rows.append(dict(dataset=did, raw_sum=sum(rd), pc_sum=sum(pd_), k=len(rd), pc_drop=np.mean(pd_),
                             slope_raw=np.mean(sl_r), slope_pc=np.mean(sl_p)))
    return pd.DataFrame(rows)


def ratio_ci(t: pd.DataFrame, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    raw, pc = t.raw_sum.to_numpy(), t.pc_sum.to_numpy()
    idx = rng.integers(0, len(t), size=(NBOOT, len(t)))
    r = pc[idx].sum(1) / raw[idx].sum(1)
    return float(pc.sum() / raw.sum()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def main() -> None:
    d = pd.read_parquet(OUT / "r11_partc_common.parquet")
    d = d[d.arm.isin(ARMS)]
    rows = []
    for (fm, arm), g in d.groupby(["fm", "arm"]):
        for sc in SC:
            t = per_table(g, sc)
            v = t.slope_pc.to_numpy()
            lo, hi = bootstrap_ci(v, seed=SEED)
            ratio, rlo, rhi = ratio_ci(t, SEED)
            rows.append(dict(fm=fm, arm=arm, scoring=sc, n=len(t), slope_pc=v.mean(), slope_lo=lo, slope_hi=hi,
                             p_wil=float(stats.wilcoxon(v, alternative="less").pvalue),
                             n_neg=int((v < 0).sum()),
                             p_sign=float(stats.binomtest(int((v < 0).sum()), int((v != 0).sum()), 0.5,
                                                          alternative="greater").pvalue),
                             ratio=ratio, ratio_lo=rlo, ratio_hi=rhi,
                             p_tost=tost_p(t.pc_drop.to_numpy())))
    o = pd.DataFrame(rows)
    o["p_wil_holm"] = np.nan
    o["p_sign_holm"] = np.nan
    for sc, g in o.groupby("scoring"):
        o.loc[g.index, "p_wil_holm"] = holm(g.p_wil.to_numpy())
        o.loc[g.index, "p_sign_holm"] = holm(g.p_sign.to_numpy())
    # check: point ratios equal r11 (and hence r4 on the complement)
    r11 = pd.read_csv(OUT / "r11_partc.csv").set_index(["fm", "arm"])
    dev = max(abs(r.ratio - r11.loc[(r.fm, r.arm), f"ratio_{r.scoring}"]) for r in o.itertuples())
    assert dev < 1e-12, dev
    o.to_csv(OUT / "r14_mech_chain.csv", index=False)

    ver = []
    for (fm, sc), g in o.groupby(["fm", "scoring"]):
        trend = bool((g.p_wil_holm < 0.05).all())
        if trend and (g.ratio_lo > THR).all():
            v = "beyond"
        elif (g.p_tost < 0.05).all():
            v = "prior shift"
        elif (g.ratio_hi < THR).any():
            v = "not beyond"
        else:
            v = "inconclusive"
        ver.append(dict(fm=fm, scoring=sc, trend_both=trend, verdict=v,
                        point_rule=("beyond" if (g.ratio >= THR).all() else "not beyond")))
    vv = pd.DataFrame(ver)
    vv.to_csv(OUT / "r14_verdicts.csv", index=False)
    pd.set_option("display.width", 250)
    print(o.round(4).to_string(index=False))
    print(vv.to_string(index=False))

    m = Macros("r14")
    src = "r14_mech_chain.py (Study 7 Part (c) DEV, table unit"
    W = dict(comp="Comp", com="Com")
    for r in o.itertuples():
        w = f"{FMW[r.fm]}{r.arm.capitalize()}{W[r.scoring]}"
        s = f"{src}; {r.fm} {r.arm} {r.scoring})"
        m.put(f"Mc{w}Ratio", fmt(r.ratio, 2), s + ": corrected/raw drop")
        m.put(f"Mc{w}RatioLo", fmt(r.ratio_lo, 2), s + ": bootstrap 95% CI lower, tables resampled")
        m.put(f"Mc{w}RatioHi", fmt(r.ratio_hi, 2), s + ": bootstrap 95% CI upper")
        m.put(f"Mc{w}PcSlope", fmt(r.slope_pc, 2, sign=True), s + ": mean per-table corrected slope")
        m.put(f"Mc{w}PWil", pfmt(r.p_wil), s + ": one-sided Wilcoxon over tables, unadjusted")
        m.put(f"Mc{w}PWilHolm", pfmt(r.p_wil_holm), s + ": one-sided Wilcoxon, Holm over 8 cells")
        m.put(f"Mc{w}NNeg", str(r.n_neg), s + ": tables with a negative corrected slope")
    for sc in SC:
        g = o[o.scoring == sc]
        m.put(f"Mc{W[sc]}PWilHolmMax", pfmt(g.p_wil_holm.max()), f"{src}; {sc}): largest Holm Wilcoxon p, 8 cells")
        m.put(f"Mc{W[sc]}PSignHolmMax", pfmt(g.p_sign_holm.max()), f"{src}; {sc}): largest Holm sign-test p, 8 cells")
        m.put(f"Mc{W[sc]}NNegMin", str(int(g.n_neg.min())), f"{src}; {sc}): fewest tables with a negative corrected slope")
        vs = vv[vv.scoring == sc].set_index("fm").verdict
        for k, name in (("beyond", "Beyond"), ("inconclusive", "Inconcl"), ("not beyond", "NotBeyond"),
                        ("prior shift", "Prior")):
            m.put(f"Mc{W[sc]}N{name}", str(int((vs == k).sum())), f"{src}; {sc}): FMs '{k}' under the table-unit chain")
        # per cell (exploratory refinement, as the per-cell reading of the rule): CI above / straddling / below 0.5
        ok = g.p_wil_holm < 0.05
        m.put(f"Mc{W[sc]}CellsAbove", str(int((ok & (g.ratio_lo > THR)).sum())), f"{src}; {sc}): cells with trend and ratio CI above 0.5")
        m.put(f"Mc{W[sc]}CellsStraddle", str(int((ok & (g.ratio_lo <= THR) & (g.ratio_hi >= THR)).sum())), f"{src}; {sc}): cells whose ratio CI contains 0.5")
        m.put(f"Mc{W[sc]}CellsBelow", str(int((g.ratio_hi < THR).sum())), f"{src}; {sc}): cells with ratio CI below 0.5")
        m.put(f"Mc{W[sc]}PTostMin", fmt(np.floor(g.p_tost.min() * 100) / 100, 2), f"{src}; {sc}): smallest TOST p (floored to 2 decimals) of the per-table corrected drop, 8 cells")
    m.put("McN", str(int(o.n.max())), f"{src}): tables per cell")
    m.put("McNBoot", f"{NBOOT:,}".replace(",", "{,}"), f"{src}): bootstrap resamples")
    m.write()


if __name__ == "__main__":
    main()
