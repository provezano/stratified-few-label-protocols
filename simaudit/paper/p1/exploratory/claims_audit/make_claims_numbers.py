"""EXPLORATORY (claims audit, 2026-10-28): macros for the P1 claims pass. Prefix ExCa.

No hand-typed numbers. Sources (all stored outputs, read-only):
- out/rho_all.csv (rho_all.py): rho(1) for fm1-fm5 on T1' (Study 9 stored results).
- out/redraw_tost.csv (redraw_tost.py): TOST of stratified - redraw, T1'.
- ../e17_metrics/e17_gap_tests.csv: pooled T1' gaps at 15/20/30C and the AUC over all classes.
- ../e17_metrics/e17_moderators.csv: T1' Spearman of the accuracy gap with pool TV, per budget.
- ../e17_metrics/e17_flips_cells.csv: per-table change of the top-ranked FM, protocol vs
  split-half noise (paired Wilcoxon over tables; mean over budgets per table).
- reports/STUDY12_REPORT.md: per-budget (ii-a) and (ii-b) rows -> share of the T1' reversal
  removed by the balance option (m 3-10, as on DEV); coverage-closure table at 3C.
- ../fm_extra/out/t1p_summary.csv: binary vs multiclass gap for six FMs on T1'.
Writes out/claims_audit_numbers.tex.
Run from repo root:  nice -n 15 .venv/bin/python -I paper/p1/exploratory/claims_audit/make_claims_numbers.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[3]
DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
MAC, SRC = {}, {}
FMW = {"fm1": "FMone", "fm2": "FMtwo", "fm3": "FMthree", "fm5": "FMfive", "fm6": "FMsix", "fm7": "FMseven"}


def put(name, value, src):
    name = re.sub(r"[^A-Za-z]", "", "".join(DIG.get(c, c) for c in name))
    if name in MAC:
        raise KeyError(name)
    MAC[name], SRC[name] = str(value), src


def f(x, nd=3, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def pct(x):
    return f"{100 * x:.0f}\\%"


def pf(p):
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


# rho(1), all four FMs (frozen rule = Study 9 primary statistic; power law = E11 sensitivity)
r = pd.read_csv(HERE / "out" / "rho_all.csv")
src = "claims_audit/out/rho_all.csv"
for lr, w in FMW.items():
    g = r[r.learner == lr]
    if g.empty:
        continue
    fr = g[g.model == "frozen"].iloc[0]
    pw = g[g.model == "power"].iloc[0]
    put(f"ExCaRho{w}", f(fr["median"], 2), src + f" ({lr}, frozen)")
    put(f"ExCaRho{w}Lo", f(fr.lo, 2), src + f" ({lr}, frozen)")
    put(f"ExCaRho{w}Hi", f(fr.hi, 2), src + f" ({lr}, frozen)")
    put(f"ExCaRho{w}Pow", f(pw["median"], 2), src + f" ({lr}, power)")
    put(f"ExCaRho{w}Share", pct(fr.share_gt1), src + f" ({lr}, frozen, share rho>1)")
fz = r[r.model == "frozen"]
put("ExCaRhoShareMin", pct(fz.share_gt1.min()), src + " (min over 4 FMs)")
put("ExCaRhoShareMax", pct(fz.share_gt1.max()), src + " (max over 4 FMs)")
oth = fz[fz.learner != "fm2"]
put("ExCaRhoOtherMin", f(oth["median"].min(), 1), src + " (fm1/fm3/fm5, min)")
put("ExCaRhoOtherMax", f(oth["median"].max(), 1), src + " (fm1/fm3/fm5, max)")
opw = r[(r.model == "power") & (r.learner != "fm2")]
put("ExCaRhoOtherPowMin", f(opw["median"].min(), 1), src + " (fm1/fm3/fm5, power law, min)")
put("ExCaRhoOtherPowMax", f(opw["median"].max(), 1), src + " (fm1/fm3/fm5, power law, max)")

# redraw TOST
t = pd.read_csv(HERE / "out" / "redraw_tost.csv").set_index("learner")
put("ExCaRedrawFMtwoTostP", f"{t.loc['fm2', 'p_tost']:.3f}", "claims_audit/out/redraw_tost.csv (fm2)")

# E17 pooled T1' gaps
g = pd.read_csv(EXP / "e17_metrics" / "e17_gap_tests.csv")
g = g[(g.set == "t1prime") & (g.learner == "pooled")]
src = "e17_metrics/e17_gap_tests.csv (t1prime, pooled)"
mf = g[g.metric == "macro_f1"].set_index("m")
big = mf.loc[[15, 20, 30]]
put("ExCaFBigMin", f(big["mean"].min(), 3, True), src + " macro_f1 m=15-30")
put("ExCaFBigMax", f(big["mean"].max(), 3, True), src + " macro_f1 m=15-30")
put("ExCaFBigHolmMax", pf(big.p_holm.max()), src + " macro_f1 m=15-30")
put("ExCaFThirty", f(mf.loc[30, "mean"], 3, True), src + " macro_f1 m=30")
put("ExCaFThirtyN", str(int(mf.loc[30, "n"])), src + " macro_f1 m=30")
au = g[g.metric == "auc_ovr_macro_all"].set_index("m").sort_index()
sig = au[au.p_holm < 0.05].index
assert list(sig) == list(au.index[: len(sig)]), "AUC significance not a prefix of budgets"
put("ExCaAucAllSigMaxM", str(int(sig.max())), src + " auc_ovr_macro_all, largest m with Holm p<0.05")

# T1' Spearman, accuracy gap vs pool TV, per budget
mo = pd.read_csv(EXP / "e17_metrics" / "e17_moderators.csv")
mo = mo[(mo.set == "t1prime") & (mo.metric == "accuracy")]
put("ExCaAccTvRhoMin", f(mo.rho_tv.min(), 2), "e17_moderators.csv (t1prime, accuracy, min over m)")
put("ExCaAccTvRhoMax", f(mo.rho_tv.max(), 2), "e17_moderators.csv (t1prime, accuracy, max over m)")

# per-table change of the top-ranked FM, protocol vs split-half noise
fl = pd.read_csv(EXP / "e17_metrics" / "e17_flips_cells.csv")
src = "e17_metrics/e17_flips_cells.csv (top_flip)"
for st, w in [("t1prime", "Tp"), ("dev", "Dev")]:
    x = fl[(fl.set == st) & (fl.metric == "macro_f1")]
    p = x.pivot_table(index="table", columns="kind", values="top_flip", aggfunc="mean").dropna()
    put(f"ExCaTop{w}Prot", f(p.protocol.mean(), 2), src + f" ({st}, macro_f1, protocol)")
    put(f"ExCaTop{w}Noise", f(p.null_split_half.mean(), 2), src + f" ({st}, macro_f1, noise)")
    put(f"ExCaTop{w}P", pf(wilcoxon(p.protocol, p.null_split_half).pvalue), src + f" ({st}, paired Wilcoxon)")

# Study 12 report: share of the reversal removed by the balance option (m 3-10)
rep = (ROOT / "reports" / "STUDY12_REPORT.md").read_text(encoding="utf-8")
blk = rep.split("## Robustness: per budget")[1].split("###")[0]


def row(label):
    line = [ln for ln in blk.splitlines() if ln.startswith(f"| {label}")][0]
    return [float(v) for v in re.findall(r"\| ([+\-]\d\.\d+) \(", line)]


iia, iib = np.mean(row("(ii-a)")[1:]), np.mean(row("(ii-b)")[1:])
put("ExCaRevShareTp", pct(1 - iib / iia), "STUDY12_REPORT.md per-budget (ii-a), (ii-b), m=3-10")

# coverage closure at 3C (S-1), per FM, and the 3C gap (per-budget (i) rows)
cblk = rep.split("Gap closed by conditioning on coverage")[1].split("\n\n(")[0]
for lr, w in FMW.items():
    ln = [l for l in cblk.splitlines() if l.startswith(f"| {lr} ")]
    if not ln:
        continue
    vals = [float(v) for v in re.findall(r"\|\s*([\d.]+)\s*(?=\|)", ln[0])]
    put(f"ExCaCovClose{w}MThree", f(vals[1], 3), "STUDY12_REPORT.md S-1 coverage-closure table, 3C")
    put(f"ExCaGap{w}MThree", f(row(f"(i)-{lr}")[1], 3), "STUDY12_REPORT.md per-budget (i), 3C")

# binary vs multiclass gap for six FMs on T1'
s = pd.read_csv(EXP / "fm_extra" / "out" / "t1p_summary.csv")
s = s[(s.set == "all") & (s.mult.astype(str) == "mean")]
b = s[s.subset == "binary"].set_index("fm").gap
m = s[s.subset == "multiclass"].set_index("fm").gap
put("ExCaSplitLargerN", str(int((m > b).sum())), "fm_extra/out/t1p_summary.csv (mean over m)")
put("ExCaSplitFMs", str(len(b)), "fm_extra/out/t1p_summary.csv (mean over m)")

lines = ["% AUTO-GENERATED by paper/p1/exploratory/claims_audit/make_claims_numbers.py. EXPLORATORY "
         "(claims audit; not pre-registered). Do not edit."]
lines += [f"\\newcommand{{\\{k}}}{{{v}}} % EXPLORATORY; source: {SRC[k]}" for k, v in sorted(MAC.items())]
(HERE / "out" / "claims_audit_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
for k, v in sorted(MAC.items()):
    print(k, v)
