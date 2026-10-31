"""Round-4 review checks (EXPLORATORY re-analyses of stored pre-registered fits; read-only inputs).

(1) Prop. 2(iii) initial rise, Study 7 Part (c), majority arm, DEV. Prop. 2(iii) applies only to
    binary tables with pi_min <= 1/3 (and J0 >= pi_1/pi_2). Per table: mean macro-F1 at s = 0.6
    minus at s = 0.5 (mean over reps and feasible mult). J0 is estimated per FM and table as
    2 * balanced accuracy - 1 of the balanced (s = 0.5) context. Reported for the eligible tables
    and, for contrast, for the others.
(2) Pre-registered Study 7 mechanism rule per FM over both arms: ratio of the corrected to the
    uncorrected drop (s = 0.5 to the largest feasible s; mean over dataset x mult), per arm.
(3) T1' accuracy, stratified - random, mean over m in {1,3,5,10}, binary tables only, split at
    TV = 0.1 (POST HOC, same cut as acc_imb_dose.py); multiclass imbalanced tables for contrast.
(4) Redraw TOST (TabICLv2, T1', m = 3-10): paired-t TOST (as in claims_audit/redraw_tost.py) and a
    Wilcoxon-based TOST (max of the two one-sided signed-rank p values for |mean| < 0.01).
Writes out/r4_rise.csv, out/r4_rule.csv, out/r4_accbin.csv, out/r4_tost.csv, out/rv_r4.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r4_checks.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from rv_common import FMS, FMW, OUT, Macros, fmt, pfmt
from simaudit.analysis.study7 import tost_p
from simaudit.settings import settings
import bc_contrasts as bc

MARGIN = 0.01


def part_c() -> pd.DataFrame:
    fr = [pd.read_parquet(p, columns=["learner", "dataset", "arm", "share", "mult", "rep", "feasible",
                                      "macro_f1", "pc_macro_f1", "balanced_accuracy"])
          for fm in FMS for p in sorted((settings.results_dir / "study7" / "c").glob(f"*_{fm}.parquet"))]
    c = pd.concat(fr, ignore_index=True)
    c["dataset"] = c.dataset.astype(int)
    return c[c.feasible]


def rise(c: pd.DataFrame):
    pools = pd.read_csv(OUT.parents[1] / "out" / "e10_pools_dev.csv").set_index("dataset")
    maj = c[c.arm == "majority"]
    t = maj.groupby(["learner", "dataset", "share"]).macro_f1.mean().unstack()
    ba = maj[maj.share == 0.5].groupby(["learner", "dataset"]).balanced_accuracy.mean()
    rows = []
    for (fm, did), r in t.iterrows():
        p = pools.loc[did]
        pi1 = float(p.pi_min)
        j0 = 2 * float(ba.loc[(fm, did)]) - 1
        rows.append(dict(fm=fm, dataset=did, C=int(p.C), pi_min=pi1, j0=j0,
                         elig_pi=bool(p.C == 2 and pi1 <= 1 / 3),
                         elig_full=bool(p.C == 2 and pi1 <= 1 / 3 and j0 >= pi1 / (1 - pi1)),
                         d56=float(r[0.6] - r[0.5])))
    return pd.DataFrame(rows)


def rule(c: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (fm, arm), g in c.groupby(["learner", "arm"]):
        drops = {}
        for met in ("macro_f1", "pc_macro_f1"):
            # as reports/study7_report.py: per (dataset, mult), first minus last feasible share
            per = g.groupby(["dataset", "mult", "share"])[met].mean().unstack()
            d = per.apply(lambda r: r.dropna().iloc[0] - r.dropna().iloc[-1] if r.notna().sum() >= 2 else np.nan, axis=1)
            drops[met] = float(d.dropna().mean())
            if met == "pc_macro_f1":
                pc_tost = tost_p(d.dropna().to_numpy()) if d.notna().sum() > 1 else np.nan
        out.append(dict(fm=fm, arm=arm, drop=drops["macro_f1"], pc_drop=drops["pc_macro_f1"],
                        ratio=drops["pc_macro_f1"] / drops["macro_f1"], pc_tost=pc_tost))
    return pd.DataFrame(out)


def acc_binary():
    c = bc.cell(bc.load_t1())
    tv = pd.read_csv(OUT / "bc_t1_pool_tv.csv").set_index("dataset")
    rows = []
    for fm in FMS:
        x, _ = bc.diff(c, fm, "stratified", "random", (1, 3, 5, 10), "accuracy")
        j = tv.join(x.rename("dA"), how="inner")
        for grp, sub in (("bin_imb", j[(j.C == 2) & (j.tv >= 0.1)]), ("bin_bal", j[(j.C == 2) & (j.tv < 0.1)]),
                         ("mc_imb", j[(j.C > 2) & (j.tv >= 0.1)])):
            v = sub.dA.to_numpy()
            rows.append(dict(fm=fm, group=grp, n=len(v), mean=v.mean(), p_wilcoxon=float(stats.wilcoxon(v).pvalue)))
    return pd.DataFrame(rows)


def tost_wilcoxon(v: np.ndarray, margin: float = MARGIN) -> float:
    p_lo = stats.wilcoxon(v + margin, alternative="greater").pvalue
    p_hi = stats.wilcoxon(v - margin, alternative="less").pvalue
    return float(max(p_lo, p_hi))


def redraw_tost() -> pd.DataFrame:
    d = bc.load_t1()
    m = d.groupby(["learner", "dataset", "strategy", "mult"]).macro_f1.mean().unstack("strategy")
    m["sr"] = m.stratified - m.redraw
    rows = []
    for fm in FMS:
        x = m.loc[fm]
        x = x[x.index.get_level_values("mult") >= 3]
        v = x.sr.groupby("dataset").mean().dropna().to_numpy()
        rows.append(dict(fm=fm, n=len(v), mean=v.mean(), p_tost_t=tost_p(v), p_tost_w=tost_wilcoxon(v)))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 200)
    c = part_c()
    ri = rise(c)
    ri.to_csv(OUT / "r4_rise.csv", index=False)
    ru = rule(c)
    ru.to_csv(OUT / "r4_rule.csv", index=False)
    ab = acc_binary()
    ab.to_csv(OUT / "r4_accbin.csv", index=False)
    tt = redraw_tost()
    tt.to_csv(OUT / "r4_tost.csv", index=False)
    summ = ri.groupby(["fm", "elig_pi"]).agg(n=("d56", "size"), mean=("d56", "mean"), up=("d56", lambda s: int((s > 0).sum())),
                                            n_full=("elig_full", "sum"))
    print(summ.round(4).to_string())
    print(ri[ri.elig_pi].round(3).to_string())
    print(ru.round(4).to_string())
    print(ab.round(4).to_string())
    print(tt.round(4).to_string())

    mac = Macros("r4")
    src = "r4_checks.py"
    e = ri[ri.elig_pi]
    m_el = e.groupby("fm").d56.mean()
    m_ot = ri[~ri.elig_pi].groupby("fm").d56.mean()
    mac.put("RfRiseEligN", str(e.dataset.nunique()), f"{src} (1): binary DEV tables with pi_min <= 1/3")
    mac.put("RfRiseEligMin", fmt(m_el.min(), 3, True), f"{src} (1): min over 4 FMs of mean d(0.5->0.6), majority arm, eligible")
    mac.put("RfRiseEligMax", fmt(m_el.max(), 3, True), f"{src} (1): max over 4 FMs, eligible")
    mac.put("RfRiseOtherN", str(ri[~ri.elig_pi].dataset.nunique()), f"{src} (1): other DEV tables")
    mac.put("RfRiseOtherMin", fmt(m_ot.min(), 3, True), f"{src} (1): min over FMs, other tables")
    mac.put("RfRiseOtherMax", fmt(m_ot.max(), 3, True), f"{src} (1): max over FMs, other tables")
    up = e.groupby("fm").d56.apply(lambda s: int((s > 0).sum()))
    mac.put("RfRiseUpMin", str(int(up.min())), f"{src} (1): min over FMs of eligible tables whose macro-F1 rises 0.5->0.6")
    mac.put("RfRiseUpMax", str(int(up.max())), f"{src} (1): max over FMs, tables rising")
    ef = ri[ri.elig_full].groupby("fm").d56.mean()
    mac.put("RfRiseFullMin", fmt(ef.min(), 3, True), f"{src} (1): min over FMs of mean d(0.5->0.6) on tables also meeting J0 >= pi1/pi2")
    mac.put("RfRiseFullMax", fmt(ef.max(), 3, True), f"{src} (1): max over FMs, J0 condition met")
    full = e.groupby("fm").elig_full.sum()
    mac.put("RfRiseJzeroMin", str(int(full.min())), f"{src} (1): min over FMs of eligible tables also meeting J0 >= pi1/pi2 (J0 = 2 BA - 1 at s=0.5)")
    mac.put("RfRiseJzeroMax", str(int(full.max())), f"{src} (1): max over FMs, J0 condition")
    for fm in FMS:
        mac.put(f"RfRise{FMW[fm]}", fmt(m_el[fm], 3, True), f"{src} (1): {fm} eligible mean d(0.5->0.6)")
    for r in ru[ru.arm.isin(["majority", "minority"])].itertuples():
        mac.put(f"RfRule{FMW[r.fm]}{r.arm.capitalize()}Ratio", fmt(r.ratio, 3), f"{src} (2): {r.fm} {r.arm} corrected/uncorrected drop (prereg rule threshold 0.5)")
    # pre-registered rule per FM over both arms (STUDY7 prereg Sec. 3); the corrected stratified JT is
    # significant in all 8 cells (reports/STUDY7_REPORT.md, pc_jt_p = 0), so "beyond" reduces to the ratio.
    ver = {}
    for fm, g in ru[ru.arm.isin(["majority", "minority"])].groupby("fm"):
        if (g.ratio >= 0.5).all() and (g["drop"] > 0).all():
            ver[fm] = "beyond"
        elif (g.pc_tost < 0.05).all():
            ver[fm] = "prior"
        else:
            ver[fm] = "unresolved"
    print("per-FM verdicts:", ver)
    vs = list(ver.values())
    mac.put("RfRuleFMNBeyond", str(vs.count("beyond")), f"{src} (2): FMs 'beyond a prior shift' by the per-FM rule over both arms")
    mac.put("RfRuleFMNUnresolved", str(vs.count("unresolved")), f"{src} (2): FMs 'unresolved' by the per-FM rule")
    mac.put("RfRuleFMNPrior", str(vs.count("prior")), f"{src} (2): FMs 'a prior shift' by the per-FM rule")
    bi = ab[ab.group == "bin_imb"]
    mac.put("RfAccBinImbN", str(int(bi.n.iloc[0])), f"{src} (3): binary T1' tables with TV >= 0.1 (post hoc)")
    mac.put("RfAccBinImbMin", fmt(bi["mean"].min(), 3, True), f"{src} (3): min over FMs, accuracy S-R, m-averaged")
    mac.put("RfAccBinImbMax", fmt(bi["mean"].max(), 3, True), f"{src} (3): max over FMs")
    mac.put("RfAccBinImbPMax", pfmt(bi.p_wilcoxon.max()), f"{src} (3): max over FMs of Wilcoxon p (unadjusted)")
    mi = ab[ab.group == "mc_imb"]
    mac.put("RfAccMcImbN", str(int(mi.n.iloc[0])), f"{src} (3): multiclass T1' tables with TV >= 0.1")
    mac.put("RfAccMcImbNSig", str(int((mi.p_wilcoxon < 0.05).sum())), f"{src} (3): FMs with Wilcoxon p < 0.05 on them")
    f2 = tt[tt.fm == "fm2"].iloc[0]
    mac.put("RfTostTFMtwo", pfmt(f2.p_tost_t), f"{src} (4): TabICLv2 redraw paired-t TOST p, margin 0.01")
    mac.put("RfTostWFMtwo", pfmt(f2.p_tost_w), f"{src} (4): TabICLv2 redraw Wilcoxon TOST p, margin 0.01")
    mac.write()


if __name__ == "__main__":
    main()
