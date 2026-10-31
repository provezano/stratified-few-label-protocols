"""Turn the E1-E15 outputs (out/*.csv) into LaTeX macros: out/exploratory_numbers.tex.

No hand-typed numbers: every value is read from a CSV written by e1-e15. Macro
names contain letters only (digits spelled out), prefix ``Ex``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
MAC: dict[str, str] = {}
SRC: dict[str, str] = {}


def put(name: str, value: str, src: str) -> None:
    name = re.sub(r"[^A-Za-z]", "", "".join(DIG.get(c, c) for c in name))
    if name in MAC:
        raise KeyError(name)
    MAC[name], SRC[name] = value, src


def f(x: float, nd: int = 3, sign: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$").replace("+", "$+$") if sign else s.replace("-", "$-$")


def pf(p: float) -> str:
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def gap_macros(df: pd.DataFrame, prefix: str, learner: str, metric: str, mult, src: str) -> None:
    r = df[(df.learner == learner) & (df.metric == metric) & (df.mult.astype(str) == str(mult))].iloc[0]
    put(f"{prefix}Mean", f(r["mean"], sign=True), src)
    put(f"{prefix}Lo", f(r.ci_lo, sign=True), src)
    put(f"{prefix}Hi", f(r.ci_hi, sign=True), src)
    put(f"{prefix}Holm", pf(r.p_holm), src)
    put(f"{prefix}Win", f"{r.win:.2f}", src)


def e1() -> None:
    src = "e1_gaps.csv"
    g = pd.read_csv(OUT / src)
    names = dict(lr="LR", knn1="KnnOne", knn3="KnnThree", gnb="GNB", lda="LDA", hgb="HGB")
    for lr, tag in names.items():
        gap_macros(g, f"ExEOne{tag}F", lr, "macro_f1", "all", src)
        for m in (1, 3, 10):
            r = g[(g.learner == lr) & (g.metric == "macro_f1") & (g.mult.astype(str) == str(m))].iloc[0]
            put(f"ExEOne{tag}FM{m}", f(r["mean"], sign=True), src)
            put(f"ExEOne{tag}FM{m}Lo", f(r.ci_lo, sign=True), src)
            put(f"ExEOne{tag}FM{m}Hi", f(r.ci_hi, sign=True), src)
    cls = [lr for lr in names if lr != "knn3"]  # knn3 at m=1 is degenerate (k = B)
    for metric, tag in (("macro_f1", "F"), ("accuracy", "Acc"), ("balanced_accuracy", "BA")):
        for m in (1, 3, 5, 10, "all"):
            s = g[g.learner.isin(cls) & (g.metric == metric) & (g.mult.astype(str) == str(m))]
            mt = "All" if m == "all" else f"M{m}"
            put(f"ExEOneCls{tag}{mt}Min", f(s["mean"].min(), sign=True), src)
            put(f"ExEOneCls{tag}{mt}Max", f(s["mean"].max(), sign=True), src)
    allcls = g[g.learner.isin(names) & (g.mult.astype(str) == "all")]
    put("ExEOneNSigF", str(int((allcls[allcls.metric == "macro_f1"].p_holm < 0.05).sum())), src)
    put("ExEOneNSigAcc", str(int((allcls[allcls.metric == "accuracy"].p_holm < 0.05).sum())), src)
    put("ExEOneNSigBA", str(int((allcls[allcls.metric == "balanced_accuracy"].p_holm < 0.05).sum())), src)
    acc = g[g.learner.isin(names) & (g.metric == "accuracy") & (g.mult.astype(str) != "all")]
    put("ExEOneAccMinP", pf(acc.p.min()), src)
    put("ExEOneAccNSigHolm", str(int((acc.p_holm < 0.05).sum())), src)
    for lr, tag in (("gnb", "GNB"), ("lda", "LDA"), ("lr", "LR")):
        for m in (3, 5):
            r = g[(g.learner == f"{lr}_pc") & (g.metric == "macro_f1") & (g.mult.astype(str) == str(m))].iloc[0]
            put(f"ExEOne{tag}PcFM{m}", f(r["mean"], sign=True), src)
    th = pd.read_csv(OUT / "e1_theory.csv")
    src = "e1_theory.csv"
    m1 = th[(th.mult == 1) & (th.learner != "knn3")]
    rs = [np.corrcoef(g2.macro_f1_miss_obs, g2.f1_cov_pred)[0, 1] for _, g2 in m1.groupby("learner")]
    put("ExEOneCovRMin", f(min(rs), 2), src)
    put("ExEOneCovRMax", f(max(rs), 2), src)
    put("ExEOneCovObs", f(m1.macro_f1_miss_obs.mean()), src)
    put("ExEOneCovPred", f(m1.f1_cov_pred.mean()), src)
    sub = th[th.learner.isin(["gnb", "lda"]) & (th.mult >= 3)]
    put("ExEOnePlugBAPos", f"{(sub.balanced_accuracy_bal_obs > 0).mean():.2f}", src)
    put("ExEOnePlugBANeg", f"{(sub.balanced_accuracy_bal_obs < -0.01).mean():.2f}", src)
    put("ExEOnePlugCells", str(len(sub)), src)
    means = th[th.mult >= 3].groupby(["learner", "mult"])[["accuracy_bal_obs", "macro_f1_bal_obs", "balanced_accuracy_bal_obs"]].mean()
    put("ExEOneAccLeFMeans", f"{(means.accuracy_bal_obs <= means.macro_f1_bal_obs).mean():.2f}", src)
    put("ExEOneOrderMeans", f"{((means.accuracy_bal_obs <= means.macro_f1_bal_obs) & (means.macro_f1_bal_obs <= means.balanced_accuracy_bal_obs)).mean():.2f}", src)
    put("ExEOneOrderMeansN", str(len(means)), src)


def e2() -> None:
    src = "e2_auc_gaps.csv"
    g = pd.read_csv(OUT / src)
    tags = dict(fm1="FMone", fm2="FMtwo", fm3="FMthree", fm5="FMfive")
    for fm, t in tags.items():
        gap_macros(g, f"ExETwo{t}AllMOne", fm, "auc_all", 1, src)
        gap_macros(g, f"ExETwo{t}All", fm, "auc_all", "all", src)
        gap_macros(g, f"ExETwo{t}Seen", fm, "auc_seen", "all", src)
    for m in (1, 3, 5, 10, "all"):
        mt = "All" if m == "all" else f"M{m}"
        for metric, tag in (("auc_all", "AUCall"), ("auc_seen", "AUCseen")):
            s = g[(g.metric == metric) & (g.mult.astype(str) == str(m))]
            put(f"ExETwo{tag}{mt}Min", f(s["mean"].min(), sign=True), src)
            put(f"ExETwo{tag}{mt}Max", f(s["mean"].max(), sign=True), src)
            put(f"ExETwo{tag}{mt}NSig", str(int((s.p_holm < 0.05).sum())), src)
    runs = pd.read_parquet(OUT / "e2_auc_runs.parquet")
    put("ExETwoSeenNaN", str(int(runs.auc_seen.isna().sum())), "e2_auc_runs.parquet")
    put("ExETwoDraws", str(len(runs)), "e2_auc_runs.parquet")


def e3() -> None:
    src = "e3_cells.csv / e3_configs.csv"
    c = pd.read_csv(OUT / "e3_cells.csv")
    k = pd.read_csv(OUT / "e3_configs.csv")
    mc = c[c.C > 2]
    imb = mc[mc.IR > 1]
    put("ExEThreeCells", str(len(mc)), src)
    put("ExEThreeCovR", f(np.corrcoef(mc.cov_mf1, mc.cov_pred_mf1)[0, 1]), src)
    lo = mc[mc.m <= 3]
    put("ExEThreeCovRatio", f(lo.cov_mf1.sum() / lo.cov_pred_mf1.sum(), 2), src)
    put("ExEThreeBAPos", f"{(mc.bal_ba >= -1e-4).mean():.2f}", src)
    order = lambda d: ((d.bal_acc <= d.bal_mf1 + 1e-4) & (d.bal_mf1 <= d.bal_ba + 1e-4)).mean()  # noqa: E731
    put("ExEThreeOrderImb", f"{order(imb):.2f}", src)
    put("ExEThreeOrderAll", f"{order(mc):.2f}", src)
    put("ExEThreeAccLeF", f"{(mc.bal_acc <= mc.bal_mf1 + 1e-4).mean():.2f}", src)
    for met, tag in (("ba", "BA"), ("acc", "Acc"), ("mf1", "F")):
        ok = mc[[f"bal_{met}", f"bal2_{met}"]].dropna()
        ok = ok[ok[f"bal_{met}"].abs() > 1e-3]
        put(f"ExEThreeSecond{tag}Sign", f"{(np.sign(ok[f'bal_{met}']) == np.sign(ok[f'bal2_{met}'])).mean():.2f}", src)
        put(f"ExEThreeSecond{tag}R", f(np.corrcoef(ok[f"bal_{met}"], ok[f"bal2_{met}"])[0, 1], 2), src)
    mk = k[(k.C > 2) & (k.IR > 1)]
    put("ExEThreeNConf", str(len(mk)), src)
    put("ExEThreeSInside", f"{((mk.s_path > 0) & (mk.s_path < 1)).mean():.2f}", src)
    put("ExEThreeSMin", f(mk.s_path.min(), 2), src)
    put("ExEThreeSMax", f(mk.s_path.max(), 2), src)
    put("ExEThreeProjMed", f(mk.proj.median(), 2), src)
    put("ExEThreeBetweenMed", f(mk.between_share.median(), 2), src)
    fin = mk[np.isfinite(mk.B_cross_sim)]
    rq = fin.B_cross_quad / fin.B_cross_sim
    put("ExEThreeCrossRatioMed", f(rq.median(), 2), src)
    put("ExEThreeCrossRatioMin", f(rq.min(), 2), src)
    put("ExEThreeCrossRatioMax", f(rq.max(), 2), src)


def e4() -> None:
    src = "e4_spearman.csv"
    s = pd.read_csv(OUT / src)
    s = s[s.subset == "all"]
    for resp, rt in (("gap_all", "All"), ("gap_m1", "MOne"), ("gap_m3", "MThree"), ("gap_m10", "MTen")):
        for p, pt in (("C", "C"), ("pi_min", "PiMin"), ("IR", "IR"), ("n", "N"), ("d", "D"),
                      ("sep_acc", "SepAcc"), ("sep_ba", "SepBA"), ("prior_dist", "PriorDist")):
            r = s[(s.response == resp) & (s.predictor == p)].iloc[0]
            put(f"ExEFour{pt}{rt}", f(r.rho, 2, sign=True), src)
            put(f"ExEFour{pt}{rt}Lo", f(r.lo, 2, sign=True), src)
            put(f"ExEFour{pt}{rt}Hi", f(r.hi, 2, sign=True), src)
    r = s[(s.response == "gap_m1") & (s.predictor == "covterm_m1")].iloc[0]
    put("ExEFourCovTermMOne", f(r.rho, 2, sign=True), src)
    put("ExEFourCovTermMOneLo", f(r.lo, 2, sign=True), src)
    put("ExEFourCovTermMOneHi", f(r.hi, 2, sign=True), src)


def e5() -> None:
    src = "e5_diff.csv"
    d = pd.read_csv(OUT / src)
    ch = d[d.changed]
    put("ExEFiveNQuant", str(len(d)), src)
    put("ExEFiveNChanged", str(len(ch)), src)
    t = ch[ch.set == "t1prime"]
    put("ExEFiveMaxAbsTOne", f(t["diff"].abs().max()), src)
    put("ExEFiveMaxAbsDev", f(ch[ch.set == "dev"]["diff"].abs().max()), src)
    nonci = ch[~ch.quantity.str.contains(r"\[1\]|\[2\]")]
    put("ExEFiveNonCIChanged", str(len(nonci)), src)
    pr = d[(d.set == "t1prime") & d.quantity.str.contains("dist\\|fm2\\|random")]
    for q, tag in (("P(m*<=5)", "Pfive"), ("P(m*<=10)", "Pten"), ("P(m*>20)", "Ptwenty")):
        for i, b in ((1, "Lo"), (2, "Hi")):
            r = pr[pr.quantity == f"/h1/dist|fm2|random/{q}[{i}]"].iloc[0]
            put(f"ExEFive{tag}{b}", f(r.boot2000, 2), src)
            put(f"ExEFive{tag}{b}Frozen", f(r.boot500, 2), src)


# ----------------------------------------------------------------- E6-E13
FT = dict(fm1="FMone", fm2="FMtwo", fm3="FMthree", fm5="FMfive")
MT = {"1": "MOne", "3": "MThree", "5": "MFive", "10": "MTen", "all": "All"}


def ci3(prefix: str, mean: float, lo: float, hi: float, src: str, nd: int = 3, sign: bool = True) -> None:
    put(prefix, f(mean, nd, sign), src)
    put(prefix + "Lo", f(lo, nd, sign), src)
    put(prefix + "Hi", f(hi, nd, sign), src)


def e6() -> None:
    src = "e6_assoc.csv"
    a = pd.read_csv(OUT / src)
    for fm, t in (*FT.items(), ("mean", "Mean")):
        for m, mt in (*MT.items(), ("all_no_max_tv", "NoMax")):
            r = a[(a.fm == fm) & (a.mult.astype(str) == m) & (a.predictor == "tv")].iloc[0]
            ci3(f"ExESix{t}{mt}Rho", r.rho, r.lo, r.hi, src, 2)
            ci3(f"ExESix{t}{mt}Slope", r.slope, r.slope_lo, r.slope_hi, src, 2)
            put(f"ExESix{t}{mt}RTwo", f(r.r2, 2), src)
            q = a[(a.fm == fm) & (a.mult.astype(str) == m) & (a.predictor == "pi_min")].iloc[0]
            ci3(f"ExESix{t}{mt}RhoPiMin", q.rho, q.lo, q.hi, src, 2)
        r = a[(a.fm == fm) & (a.mult == "all") & (a.predictor == "tv")].iloc[0]
        put(f"ExESix{t}AccMin", f(r.dA_min, sign=True), src)
        put(f"ExESix{t}AccMax", f(r.dA_max, sign=True), src)
        put(f"ExESix{t}NPos", str(int(r.n_pos)), src)
        put(f"ExESix{t}NNeg", str(int(r.n_neg)), src)
    c = pd.read_csv(OUT / "e6_cells.csv")
    put("ExESixMaxTVTable", str(int(c.loc[c.tv.idxmax(), "dataset"])), "e6_cells.csv")
    put("ExESixMaxTV", f(c.tv.max(), 2), "e6_cells.csv")
    src = "e6_threshold.csv"
    th = pd.read_csv(OUT / src)
    for fm, t in (*FT.items(), ("all", "All")):
        r = th[th.fm == fm].iloc[0]
        put(f"ExESixThr{t}Agree", f(r.agree, 2), src)
        put(f"ExESixThr{t}Beyond", f(r.dA_beyond, sign=True), src)
        put(f"ExESixThr{t}Below", f(r.dA_below, sign=True), src)
        put(f"ExESixThr{t}RhoLog", f(r.rho_logratio, 2, sign=True), src)
    r = th[th.fm == "all"].iloc[0]
    put("ExESixThrNCells", str(int(r.n_cells)), src)
    put("ExESixThrNDecisive", str(int(r.n_decisive)), src)
    put("ExESixThrNBeyond", str(int(r.n_beyond)), src)


def e7() -> None:
    src = "e7_gaps.csv"
    g = pd.read_csv(OUT / src)
    g["mult"] = g.mult.astype(str)
    for metric, mt_ in (("macro_f1", "F"), ("accuracy", "Acc"), ("balanced_accuracy", "BA")):
        for fm, t in FT.items():
            for m in ("1", "3", "5", "10", "all"):
                for v, vt in (("pair", "Pair"), ("cur", "Cur"), ("pair_minus_cur", "PairMinusCur"),
                              ("strict", "Strict"), ("strict_minus_cur", "StrictMinusCur")):
                    r = g[(g.variant == v) & (g.metric == metric) & (g.fm == fm) & (g.mult == m)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    ci3(f"ExESeven{mt_}{t}{MT[m]}{vt}", r["mean"], r.ci_lo, r.ci_hi, src)
                    if v in ("pair", "strict"):
                        put(f"ExESeven{mt_}{t}{MT[m]}{vt}Holm", pf(r.p_holm), src)
                        if v == "strict":
                            put(f"ExESeven{mt_}{t}{MT[m]}{vt}N", str(int(r.n)), src)
            d = g[(g.variant == "pair_minus_cur") & (g.metric == metric)]
        d = g[(g.variant == "pair_minus_cur") & (g.metric == metric)]
        put(f"ExESeven{mt_}PairMinusCurMin", f(d["mean"].min(), sign=True), src)
        put(f"ExESeven{mt_}PairMinusCurMax", f(d["mean"].max(), sign=True), src)
        d = g[(g.variant == "strict_minus_cur") & (g.metric == metric)]
        put(f"ExESeven{mt_}StrictMinusCurMin", f(d["mean"].min(), sign=True), src)
        put(f"ExESeven{mt_}StrictMinusCurMax", f(d["mean"].max(), sign=True), src)
        p = g[(g.variant == "pair") & (g.metric == metric) & (g.mult != "all")]
        put(f"ExESeven{mt_}PairNSig", str(int((p.p_holm < 0.05).sum())), src)
        put(f"ExESeven{mt_}PairNCells", str(len(p)), src)
    src = "e7_cells.csv"
    c = pd.read_csv(OUT / src)
    one = c[c.fm == "fm1"]
    for m, gm in one.groupby("mult"):
        mt = MT[str(m)]
        put(f"ExESeven{mt}StrictUsable", str(int(gm.strict_ok.sum())), src)
        put(f"ExESeven{mt}StrictMedRows", str(int(round(gm.strict_rows.median()))), src)
        put(f"ExESeven{mt}PairMeanRows", str(int(round(gm.pair_rows_mean.mean()))), src)
        put(f"ExESeven{mt}PairMinRows", str(int(gm.pair_rows_min.min())), src)
        put(f"ExESeven{mt}PairMinClass", str(int(gm.pair_min_class.min())), src)
    put("ExESevenPairExcluded", str(int(one.pair_missing.sum())), src)
    put("ExESevenPairTotal", str(int((one.pair_kept + one.pair_missing).sum())), src)


def e8() -> None:
    src = "e8_gaps.csv"
    g = pd.read_csv(OUT / src)
    for sub, st in (("all", ""), ("binary", "Bin"), ("multiclass", "Multi")):
        for fm, t in FT.items():
            for m in (1, 3):
                for v, vt in (("l0", "LZero"), ("argmax0.05", "ArgmaxA"), ("argmax0.1", "ArgmaxB"), ("argmax0.2", "ArgmaxC"),
                              ("guess0.05", "GuessA"), ("guess0.1", "GuessB"), ("guess0.2", "GuessC")):
                    r = g[(g.subset == sub) & (g.fm == fm) & (g.mult == m) & (g.variant == v) & (g.part == "cov")].iloc[0]
                    ci3(f"ExEEight{st}{t}M{m}Cov{vt}", r["mean"], r.ci_lo, r.ci_hi, src)
                    if v != "l0":
                        ci3(f"ExEEight{st}{t}M{m}Cov{vt}Change", r.change, r.change_lo, r.change_hi, src, 4)
        for v, vt in (("argmax0.2", "ArgmaxC"), ("guess0.05", "GuessA"), ("guess0.2", "GuessC")):
            r = g[(g.subset == sub) & (g.mult == 1) & (g.variant == v) & (g.part == "cov")]
            put(f"ExEEight{st}MOneCov{vt}ChangeMin", f(r.change.min(), 4, sign=True), src)
            put(f"ExEEight{st}MOneCov{vt}ChangeMax", f(r.change.max(), 4, sign=True), src)
            put(f"ExEEight{st}MOneCov{vt}RelMin", f((r.change / g[(g.subset == sub) & (g.mult == 1) & (g.variant == "l0") & (g.part == "cov")]["mean"].to_numpy()).min(), 2), src)
            put(f"ExEEight{st}MOneCov{vt}RelMax", f((r.change / g[(g.subset == sub) & (g.mult == 1) & (g.variant == "l0") & (g.part == "cov")]["mean"].to_numpy()).max(), 2), src)
    a = g[(g.variant.str.startswith("argmax")) & (g.part == "cov")]
    put("ExEEightArgmaxMaxAbsChange", f(a.change.abs().max(), 4), src)
    c = pd.read_csv(OUT / "e8_cells.csv")
    put("ExEEightShareAbsentPredC", f(c["share_absent_pred0.2"].mean(), 4), "e8_cells.csv")
    put("ExEEightShareAbsentPredCMOneMulti", f(c[(c.mult == 1) & (c.C > 2)]["share_absent_pred0.2"].mean(), 4), "e8_cells.csv")


def e9() -> None:
    src = "e9_meta.csv"
    g = pd.read_csv(OUT / src)
    g["mult"] = g.mult.astype(str)
    for metric, mt_ in (("macro_f1", "F"), ("accuracy", "Acc"), ("balanced_accuracy", "BA")):
        for fm, t in FT.items():
            for m in ("1", "3", "5", "10", "all"):
                r = g[(g.metric == metric) & (g.fm == fm) & (g.mult == m) & (g.estimator == "REML")].iloc[0]
                k = f"ExENine{mt_}{t}{MT[m]}"
                ci3(k + "Mu", r.mu, r.ci_lo, r.ci_hi, src)
                put(k + "Tau", f(r.tau), src)
                put(k + "ITwo", f(r.I2, 2), src)
                put(k + "PILo", f(r.pi_lo, sign=True), src)
                put(k + "PIHi", f(r.pi_hi, sign=True), src)
        r = g[(g.metric == metric) & (g.estimator == "REML")]
        put(f"ExENine{mt_}ITwoMin", f(r.I2.min(), 2), src)
        put(f"ExENine{mt_}ITwoMax", f(r.I2.max(), 2), src)
    src = "e9_varcomp.csv"
    v = pd.read_csv(OUT / src)
    v = v[v.fm == "all"]
    for metric, mt_ in (("macro_f1", "F"), ("accuracy", "Acc"), ("balanced_accuracy", "BA")):
        for m in (1, 3, 5, 10):
            r = v[(v.metric == metric) & (v.mult == m)].iloc[0]
            tot = r.table + r.model + r.table_x_model + r.draw
            for comp, ct in (("table", "Table"), ("model", "Model"), ("table_x_model", "TableModel"), ("draw", "Draw")):
                put(f"ExENine{mt_}{MT[str(m)]}VC{ct}SD", f(np.sqrt(r[comp])), src)
                put(f"ExENine{mt_}{MT[str(m)]}VC{ct}Share", f(r[comp] / tot, 2), src)


def e10() -> None:
    src = "e10_split.csv"
    g = pd.read_csv(OUT / src)
    g["mult"] = g.mult.astype(str)
    for metric, mt_ in (("dM", "F"), ("dA", "Acc")):
        for fm, t in FT.items():
            for m in ("1", "3", "5", "10", "all"):
                r = g[(g.metric == metric) & (g.fm == fm) & (g.mult == m)].iloc[0]
                k = f"ExETen{mt_}{t}{MT[m]}"
                ci3(k + "Bin", r.bin_mean, r.bin_lo, r.bin_hi, src)
                ci3(k + "Multi", r.mc_mean, r.mc_lo, r.mc_hi, src)
                ci3(k + "Diff", r["diff"], r.diff_lo, r.diff_hi, src)
                put(k + "MWp", f(r.mw_p, 2), src)
    r = g.iloc[0]
    put("ExETenNBin", str(int(r.bin_n)), src)
    put("ExETenNMulti", str(int(r.mc_n)), src)
    t1 = pd.read_csv(OUT / "e10_pools_t1prime.csv")
    put("ExETenTOneN", str(len(t1)), "e10_pools_t1prime.csv")
    put("ExETenTOneNBin", str(int((t1.C == 2).sum())), "e10_pools_t1prime.csv")
    put("ExETenTOneNMulti", str(int((t1.C > 2).sum())), "e10_pools_t1prime.csv")
    put("ExETenTOneMissMOne", f(t1.miss_obs_m1.mean(), 2), "e10_pools_t1prime.csv")
    put("ExETenTOneMissMThree", f(t1.miss_obs_m3.mean(), 2), "e10_pools_t1prime.csv")
    d = pd.read_csv(OUT / "e10_pools_dev.csv")
    put("ExETenDevMissMOne", f(d.miss_obs_m1.mean(), 2), "e10_pools_dev.csv")
    put("ExETenDevMissMThree", f(d.miss_obs_m3.mean(), 2), "e10_pools_dev.csv")


def e11() -> None:
    src = "e11_rho.csv"
    g = pd.read_csv(OUT / src)
    for lr, lt in (("fm2", "FMtwo"), ("fm1", "FMone")):
        for model, mt_ in (("frozen", "Frozen"), ("log", "Log"), ("power", "Power")):
            r = g[(g.learner == lr) & (g.model == model)].iloc[0]
            ci3(f"ExEEleven{lt}Rho{mt_}", r["median"], r.lo, r.hi, src, 2, sign=False)
            put(f"ExEEleven{lt}Rho{mt_}ShareGtOne", f(r.share_gt1, 2), src)
        r = g[(g.learner == lr) & (g.model == "grid")].iloc[0]
        put(f"ExEEleven{lt}RhoGridCell", f"$({r.median_cell_lo:g}, {r['median']:g}]$", src)
        put(f"ExEEleven{lt}RhoGridLo", f"{r.lo:g}", src)
        put(f"ExEEleven{lt}RhoGridHi", f"{r.hi:g}", src)
        sub = g[(g.learner == lr) & (g.model.isin(["frozen", "log", "power"]))]
        put(f"ExEEleven{lt}RhoMin", f(sub["median"].min(), 2), src)
        put(f"ExEEleven{lt}RhoMax", f(sub["median"].max(), 2), src)
    src = "e11_fit.csv"
    fit = pd.read_csv(OUT / src)
    for lr, lt in (("fm2", "FMtwo"), ("fm1", "FMone")):
        x = fit[fit.learner == lr]
        for model, mt_ in (("log", "Log"), ("power", "Power")):
            put(f"ExEEleven{lt}{mt_}RTwoMed", f(x[f"r2_{model}"].median(), 2), src)
            put(f"ExEEleven{lt}{mt_}RTwoShareNine", f((x[f"r2_{model}"] >= 0.9).mean(), 2), src)
            put(f"ExEEleven{lt}{mt_}RMSEMed", f(x[f"rmse_{model}"].median(), 3), src)
    src = "e11_mstar.csv"
    ms = pd.read_csv(OUT / src)
    for eps, et in ((0.005, "EpsLow"), (0.01, "EpsMid"), (0.02, "EpsHigh")):
        r = ms[np.isclose(ms.eps, eps)].iloc[0]
        put(f"ExEEleven{et}Median", f"$({r.median_lo:g}, {r.median_hi:g}]$".replace("inf", "\\infty"), src)
        put(f"ExEEleven{et}IQR", f"$({r.q25_lo:g}, {r.q25_hi:g}]$--$({r.q75_lo:g}, {r.q75_hi:g}]$".replace("inf", "\\infty"), src)
        for q, qt in (("P(m*<=5)", "Pfive"), ("P(m*<=10)", "Pten")):
            ci3(f"ExEEleven{et}{qt}", r[q], r[q + "_lo"], r[q + "_hi"], src, 2, sign=False)
        for m in (3, 5, 10, 20):
            ci3(f"ExEEleven{et}GainAfter{m}", r[f"P_gain_after_{m}"], r[f"P_gain_after_{m}_lo"], r[f"P_gain_after_{m}_hi"], src, 2, sign=False)
            ci3(f"ExEEleven{et}NextStep{m}", r[f"next_step_{m}"], r[f"next_step_{m}_lo"], r[f"next_step_{m}_hi"], src, 2, sign=False)
        put(f"ExEEleven{et}RightCens", str(int(r.n_right_cens)), src)


def e12() -> None:
    src = "e12_confirmatory.csv"
    t = pd.read_csv(OUT / src)
    tags = {"(i)-fm1": "IFMone", "(i)-fm2": "IFMtwo", "(i)-fm3": "IFMthree", "(i)-fm5": "IFMfive",
            "(ii-a)": "IIa", "(ii-b)": "IIb", "(ii-c)": "IIc"}
    for cid, tg in tags.items():
        r = t[t.id == cid].iloc[0]
        put(f"ExETwelve{tg}RawP", pf(r.p), src)
        put(f"ExETwelve{tg}HolmMult", str(int(r.holm_multiplier)), src)
        put(f"ExETwelve{tg}HolmP", pf(r.p_holm), src)
    put("ExETwelveFamilySize", str(len(t)), src)
    put("ExETwelveExactFloor", pf(2 / 2 ** 20), "exact Wilcoxon, n = 20: 2/2^20")
    put("ExETwelveExactFloorHolm", pf(len(t) * 2 / 2 ** 20), "15 x 2/2^20")
    src = "e12_binary_p.csv"
    b = pd.read_csv(OUT / src)
    put("ExETwelveBinMissN", str(len(b)), src)
    put("ExETwelveBinMissPGeHalf", f((b.p >= 0.5).mean(), 2), src)
    put("ExETwelveBinMissPGeHalfN", str(int((b.p >= 0.5).sum())), src)
    put("ExETwelveBinMissMOnePGeHalf", f((b[b.mult == 1].p >= 0.5).mean(), 2), src)
    put("ExETwelveBinMissMOneN", str(int((b.mult == 1).sum())), src)
    put("ExETwelveBinMissMinP", f(b.p.min(), 2), src)
    put("ExETwelveBinMissExtraMed", f(b.extra_loss.median()), src)
    put("ExETwelveBinMissExtraMin", f(b.extra_loss.min()), src)
    src = "e12_slopes.csv"
    s = pd.read_csv(OUT / src)
    for fm, tg in FT.items():
        for arm in ("majority", "minority"):
            for q, qt in (("slope_macro_f1", "Slope"), ("slope_pc_macro_f1", "SlopePc"), ("slope_diff", "SlopeDiff"),
                          ("drop_macro_f1", "Drop"), ("drop_pc_macro_f1", "DropPc"), ("drop_diff", "DropDiff")):
                r = s[(s.fm == fm) & (s.arm == arm) & (s.quantity == q)].iloc[0]
                k = f"ExETwelve{tg}{arm.title()}{qt}"
                ci3(k, r["mean"], r.ci_lo, r.ci_hi, src)
                put(k + "P", pf(r.p), src)
            r = s[(s.fm == fm) & (s.arm == arm) & (s.quantity == "slope_macro_f1")].iloc[0]
            put(f"ExETwelve{tg}{arm.title()}ShareNeg", f(r.share_neg, 2), src)
    raw = s[s.quantity == "slope_macro_f1"]
    put("ExETwelveSlopeMaxP", pf(raw.p.max()), src)
    put("ExETwelveSlopeMinShareNeg", f(raw.share_neg.min(), 2), src)
    src = "e12_ratio.csv"
    r = pd.read_csv(OUT / src).iloc[0]
    put("ExETwelveRatio", f(r.ratio, 2), src)
    put("ExETwelveRatioLo", f(r.lo, 2), src)
    put("ExETwelveRatioHi", f(r.hi, 2), src)
    put("ExETwelveRatioPct", f"{100 * r.ratio:.0f}\\%", src)
    put("ExETwelveRatioPctLo", f"{100 * r.lo:.0f}\\%", src)
    put("ExETwelveRatioPctHi", f"{100 * r.hi:.0f}\\%", src)
    put("ExETwelveRatioPBootHalf", f(r.share_boot_gt_half, 2), src)


def e13() -> None:
    src = "e13_diff.csv"
    d = pd.read_csv(OUT / src)
    d["mult"] = d.mult.astype(str)
    for metric, mt_ in (("dM", "F"), ("dA", "Acc"), ("dBA", "BA")):
        for fm, t in FT.items():
            for m in ("1", "3", "5", "10", "all"):
                r = d[(d.classical == "classical_mean") & (d.metric == metric) & (d.fm == fm) & (d.mult == m)].iloc[0]
                k = f"ExEThirteen{mt_}{t}{MT[m]}"
                ci3(k + "D", r["mean"], r.ci_lo, r.ci_hi, src)
                put(k + "Holm", pf(r.p_holm), src)
                put(k + "GapFM", f(r.gap_fm, sign=True), src)
                put(k + "GapCl", f(r.gap_cl, sign=True), src)


def e14() -> None:
    src = "e14_mstar.csv"
    ms = pd.read_csv(OUT / src)
    GT = dict(binary="Bin", multiclass="Multi", diff="Diff")
    for lr, lt in (("fm2", "FMtwo"), ("fm1", "FMone")):
        for eps, et in ((0.005, "EpsLow"), (0.01, "EpsMid"), (0.02, "EpsHigh")):
            for g, gt in GT.items():
                r = ms[(ms.learner == lr) & np.isclose(ms.eps, eps) & (ms.group == g)].iloc[0]
                k = f"ExEFourteen{lt}{et}{gt}"
                sg = g == "diff"
                ci3(k + "Pfive", r.P_le_5, r.P_le_5_lo, r.P_le_5_hi, src, 2, sign=sg)
                ci3(k + "Pten", r.P_le_10, r.P_le_10_lo, r.P_le_10_hi, src, 2, sign=sg)
                ci3(k + "GainAfterTen", r.P_gain_after_10, r.P_gain_after_10_lo, r.P_gain_after_10_hi, src, 2, sign=sg)
                if not sg:
                    put(k + "Median", f"$({r.median_lo:g}, {r.median_hi:g}]$".replace("inf", "\\infty"), src)
                    put(k + "IQR", f"$({r.q25_lo:g}, {r.q25_hi:g}]$--$({r.q75_lo:g}, {r.q75_hi:g}]$".replace("inf", "\\infty"), src)
                    put(k + "RightCens", str(int(r.n_right_cens)), src)
                    put(k + "N", str(int(r.n)), src)
    src = "e14_rho.csv"
    rh = pd.read_csv(OUT / src)
    for lr, lt in (("fm2", "FMtwo"), ("fm1", "FMone")):
        for model, mt_ in (("frozen", "Frozen"), ("log", "Log"), ("power", "Power")):
            for g, gt in GT.items():
                r = rh[(rh.learner == lr) & (rh.model == model) & (rh.group == g)].iloc[0]
                k = f"ExEFourteen{lt}Rho{mt_}{gt}"
                ci3(k, r["median"], r.lo, r.hi, src, 2, sign=g == "diff")
                if g != "diff":
                    put(k + "ShareGtOne", f(r.share_gt1, 2), src)
                    put(k + "NInf", str(int(r.n_inf)), src)


def e15() -> None:
    """E15 macros are computed by e15_budget_metrics.py (out/e15_macros.csv, also out/e15_numbers.tex)."""
    m = pd.read_csv(OUT / "e15_macros.csv", dtype=str, keep_default_na=False)
    for r in m.itertuples(index=False):
        put(r.name, r.value, f"{r.src} (via e15_macros.csv)")


HERE = Path(__file__).resolve().parent


def fmx() -> None:
    """New FMs on DEV, Study 7 Part (a) design (fm_extra): fm6 TabDPT v1.3 and fm8 LimiX-16M from
    out/s7a_summary.csv, fm7 Mitra-v2 from the per-fit-seeded re-run out/s7a_v2_summary.csv."""
    for fm, tag, src in [("fm6", "FMsix", "fm_extra/out/s7a_summary.csv"),
                         ("fm7", "FMseven", "fm_extra/out/s7a_v2_summary.csv"),
                         ("fm8", "FMeight", "fm_extra/out/s7a_summary.csv")]:
        d = pd.read_csv(HERE / src)
        d = d[d.fm == fm]
        for mult, mt in [("mean", "All"), ("1", "MOne"), ("10", "MTen")]:
            r = d[d.mult.astype(str) == mult].iloc[0]
            ci3(f"ExFmx{tag}Gap{mt}", r.gap, r.ci_lo, r.ci_hi, src)
            put(f"ExFmx{tag}Gap{mt}P", pf(r.wilcoxon_p), src + " (Wilcoxon, unadjusted)")
        put(f"ExFmx{tag}N", str(int(d.tables.iloc[0])), src)
    # T1' replication of fm6/fm7 (same Study 12 selections, seeds and panel; EXPLORATORY, not pre-registered)
    src = "fm_extra/out/t1p_summary.csv (EXPLORATORY; T1', Study 12 draws; stratified - random)"
    t = pd.read_csv(HERE / src.split(" ")[0])
    t = t[(t.set == "all") & (t.subset == "all")]
    for fm, tag in [("fm6", "FMsix"), ("fm7", "FMseven")]:
        d = t[t.fm == fm]
        for mult, mt in [("mean", "All"), ("1", "MOne"), ("10", "MTen")]:
            r = d[d.mult.astype(str) == mult].iloc[0]
            ci3(f"ExFmxTp{tag}Gap{mt}", r.gap, r.ci_lo, r.ci_hi, src)
            put(f"ExFmxTp{tag}Gap{mt}P", pf(r.wilcoxon_p), src + " (Wilcoxon, unadjusted)")
        r = d[d.mult.astype(str) == "mean"].iloc[0]
        put(f"ExFmxTp{tag}N", str(int(r.tables)), src)
        put(f"ExFmxTp{tag}Pos", str(int(round(r.win * r.tables))), src + "; tables with positive mean gap")


def bdose() -> None:
    """Counts over the 8 (FM x side) cells of the dose arms F/S/N (b_dose/out/summary.csv);
    a cell declines when the 95% bootstrap CI of the mean table slope is below 0."""
    src = "b_dose/out/summary.csv"
    d = pd.read_csv(HERE / src)
    for metric, arm, tag in [("macro_f1", "F", "F"), ("macro_f1", "S", "S"), ("macro_f1", "N", "N"),
                             ("pc_macro_f1", "F", "FPc"), ("cm_macro_f1", "F", "FCm"),
                             ("cm_pc_macro_f1", "F", "FCmPc")]:
        g = d[(d.metric == metric) & (d.arm == arm)]
        assert len(g) == 8, (metric, arm)
        put(f"ExBSum{tag}Decl", str(int((g.slope_hi < 0).sum())), src + f" ({metric}, arm {arm}, CI < 0)")
        put(f"ExBSum{tag}Rise", str(int((g.slope_lo > 0).sum())), src + f" ({metric}, arm {arm}, CI > 0)")
        put(f"ExBSum{tag}Cells", str(len(g)), src)
        put(f"ExBSum{tag}SlopeMin", f(g.slope_mean.min(), sign=True), src + f" ({metric}, arm {arm})")
        put(f"ExBSum{tag}SlopeMax", f(g.slope_mean.max(), sign=True), src + f" ({metric}, arm {arm})")


def bdose_t1p() -> None:
    """Dose arms F/S/N on the preregistered 16-table T1' subset (EXPLORATORY; supports the appendix only).
    Prereg: b_dose/PREREG_T1P_SUBSET.md. Residual = -0.4 x slope of p/pihat macro-F1 in arm F
    (positive = decline that survives); ROPE +-0.01; 90% CI for TOST, 95% CI otherwise."""
    tag = " (EXPLORATORY; T1' subset, prereg b_dose/PREREG_T1P_SUBSET.md)"
    src = "b_dose/out_t1p/primary.csv" + tag
    if not (HERE / "b_dose/out_t1p/primary.csv").exists():
        return
    p = pd.read_csv(HERE / "b_dose/out_t1p/primary.csv")
    fmt = dict(fm1="TabPFNTwo", fm2="TabICL", fm3="Kumo", fm5="TabPFNThreeFive")
    side_t = dict(majority="Maj", minority="Min", pooled="Pool")
    for label, st in (("T1'", ""), ("DEV", "Dev")):
        for arm in ("F", "N"):
            g = p[(p.set == label) & (p.arm == arm)]
            at = "" if arm == "F" else "N"
            for r in g.itertuples():
                k = f"ExBTp{st}{at}{side_t[r.side]}{fmt[r.learner]}Res"
                ci3(k, r.mean, r.lo95, r.hi95, src)
                put(k + "LoNinety", f(r.lo90, sign=True), src + " (90% CI)")
                put(k + "HiNinety", f(r.hi90, sign=True), src + " (90% CI)")
            c = g[g.side != "pooled"]
            for v, vt in (("equivalent", "Equiv"), ("decline survives", "Fail"), ("over-correction", "Over"),
                          ("inconclusive", "Incon")):
                put(f"ExBTp{st}{at}{vt}", str(int((c.verdict == v).sum())), src + f" (arm {arm}, cells '{v}')")
            put(f"ExBTp{st}{at}Cells", str(len(c)), src)
            put(f"ExBTp{st}{at}ResMin", f(c["mean"].min(), sign=True), src)
            put(f"ExBTp{st}{at}ResMax", f(c["mean"].max(), sign=True), src)
            put(f"ExBTp{st}{at}ResLoMin", f(c.lo95.min(), sign=True), src + " (lowest 95% bound)")
            put(f"ExBTp{st}{at}ResHiMax", f(c.hi95.max(), sign=True), src + " (highest 95% bound)")
    c = p[(p.set == "T1'") & (p.arm == "F") & (p.side != "pooled")]
    decision = ("pass" if (c.verdict == "equivalent").all() else "fail" if (c.verdict == "decline survives").any()
                else "not established")
    put("ExBTpDecision", decision, src + " (prereg Sec. 6)")
    # corrected macro-F1 rise from s = 0.5 to 0.9 at fixed count (= -residual), arm F, 8 FM x side cells
    put("ExBTpRiseMin", f(-c["mean"].max()), src + " (rise = -residual; smallest over 8 cells)")
    put("ExBTpRiseMax", f(-c["mean"].min()), src + " (rise = -residual; largest over 8 cells)")
    src2 = "b_dose/out_t1p/summary.csv" + tag
    d = pd.read_csv(HERE / "b_dose/out_t1p/summary.csv")
    for metric, arm, t in [("macro_f1", "F", "F"), ("macro_f1", "S", "S"), ("macro_f1", "N", "N"),
                           ("pc_macro_f1", "F", "FPc"), ("pc_macro_f1", "S", "SPc"), ("pc_macro_f1", "N", "NPc"),
                           ("cm_macro_f1", "F", "FCm"), ("cm_pc_macro_f1", "F", "FCmPc")]:
        g = d[(d.metric == metric) & (d.arm == arm)]
        put(f"ExBTpSum{t}Decl", str(int((g.slope_hi < 0).sum())), src2 + f" ({metric}, arm {arm}, CI < 0)")
        put(f"ExBTpSum{t}Rise", str(int((g.slope_lo > 0).sum())), src2 + f" ({metric}, arm {arm}, CI > 0)")
        put(f"ExBTpSum{t}Cells", str(len(g)), src2)
        put(f"ExBTpSum{t}SlopeMin", f(g.slope_mean.min(), sign=True), src2 + f" ({metric}, arm {arm})")
        put(f"ExBTpSum{t}SlopeMax", f(g.slope_mean.max(), sign=True), src2 + f" ({metric}, arm {arm})")
    agree = not (c.verdict == "decline survives").any() and int(
        (d[(d.metric == "pc_macro_f1") & (d.arm == "S")].slope_hi < 0).sum()) >= 6
    put("ExBTpAgreeDev", "yes" if agree else "no", src + " (prereg Sec. 6: no surviving decline and S after p/pihat declines in >= 6 of 8)")
    sub = pd.read_csv(HERE / "b_dose/t1p_subset.csv")
    src3 = "b_dose/t1p_subset.csv" + tag
    put("ExBTpTables", str(len(sub)), src3)
    put("ExBTpBin", str(int((sub.C == 2).sum())), src3)
    put("ExBTpMulti", str(int((sub.C > 2).sum())), src3)
    put("ExBTpPanel", "1000", "b_dose/t1p_subset.py PANEL_MAX" + tag)
    cost = json.loads((HERE / "b_dose/out_t1p/cost.json").read_text())
    src4 = "b_dose/out_t1p/cost.json" + tag
    put("ExBTpFits", f"{cost['fits']:,}".replace(",", "{,}"), src4)
    put("ExBTpCpuHours", f"{cost['cpu_hours']:.0f}", src4)
    put("ExBTpNFMs", str(len(cost["hours_by_fm"])), src4)


def e17extra() -> None:
    """E17 caveat (independent verification): macro precision at 15C and 20C on T1' (pooled FMs)."""
    src = "e17_metrics/e17_gap_tests.csv"
    d = pd.read_csv(HERE / src)
    for m, t in [(15, "MOneFive"), (20, "MTwoZero")]:
        r = d[(d.set == "t1prime") & (d.metric == "macro_precision") & (d.learner == "pooled") & (d.m == m)].iloc[0]
        put(f"ExEPrecTp{t}", f(r["mean"], sign=True), src + f" (t1prime, macro_precision, pooled, m={m})")
        put(f"ExEPrecTp{t}N", str(int(r.n)), src + f" (t1prime, macro_precision, pooled, m={m})")
        put(f"ExEPrecTp{t}Ph", pf(r.p_holm), src + f" (Holm within family, m={m})")
    # Holm p-values of e17_numbers.tex re-printed in the paper's p format (same values)
    src2 = "out/e17_numbers.tex"
    tex = (OUT / "e17_numbers.tex").read_text()
    for raw, name in [("HiImbTpCitlLOneMThreePh", "CitlHiMThree"), ("HiImbTpCitlLOneMTenPh", "CitlHiMTen"),
                      ("HiImbTpAccuracyMThreePh", "AccHiMThree"), ("HiImbTpAccuracyMTenPh", "AccHiMTen"),
                      ("HiImbTpBrierMThreePh", "BrierHiMThree"), ("HiImbTpBrierMTenPh", "BrierHiMTen"),
                      ("HiImbTpEceClasswiseMTenPh", "EceHiMTen"),
                      ("TpLogLrPosMTenPh", "LrPosMTen"), ("TpLogLrNegMTenPh", "LrNegMTen"),
                      ("TpAucOvrMacroAllMOnePh", "AucAllMOne"), ("TpMacroFOneMOnePh", "MFMOne")]:
        v = re.search(r"\\newcommand\{\\ExESeventeen" + raw + r"\}\{([^}]*)\}", tex).group(1)
        x = float(v)
        put(f"ExEPSeventeen{name}", pf(x) if x < 0.01 else f"{x:.3f}", src2 + f" (ExESeventeen{raw}, re-printed)")


def main() -> None:
    for fn in (e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, fmx, bdose, bdose_t1p, e17extra):
        fn()
    lines = ["% Generated by paper/p1/exploratory/make_exploratory_numbers.py. Do not edit.",
             "% Exploratory checks E1-E15 (DEV only except E3 synthetic and E10/E11/E14/E15 on T1'). Source CSV in a comment."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}} % {SRC[k]}" for k, v in sorted(MAC.items())]
    (OUT / "exploratory_numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(MAC)} macros", file=sys.stderr)


if __name__ == "__main__":
    main()
