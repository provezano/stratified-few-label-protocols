"""EXPLORATORY summary of fm8 LimiX-16M on T1' (run_t1p_fm8.py; plan limix_t1p/PLAN.md).

Same analysis as ``summarize_t1p.py`` (imported, unchanged): per (FM, m) and mean over
m in {1, 3, 5, 10}, per-table mean over reps of stratified and random, gap = stratified -
random, mean over tables, 10k bootstrap 95% CI over tables (seed 0), two-sided Wilcoxon
signed-rank p (unadjusted, as for fm6/fm7), share of tables with positive gap, DEV gap
of the same FM. Added: Holm adjustment over fm8's four per-budget p values.
Context rows: fm1-fm5 (stored Study 12, read-only; report frozen) and fm6/fm7 (out/t1p).

Writes out/t1p_fm8_summary.csv (macro-F1), out/t1p_fm8_acc_summary.csv (accuracy, same method;
LimiX DEV accuracy gap from out/s7a via summarize.py), out/T1P_FM8_SUMMARY.md, out/t1p_fm8_numbers.tex
(macro prefix ExLimixTp, every macro EXPLORATORY).
Usage: PYTHONPATH=src .venv_fm_extra/bin/python paper/p1/exploratory/fm_extra/summarize_t1p_fm8.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import summarize_t1p as st  # noqa: E402
from simaudit.analysis.report import holm  # noqa: E402

RUNS8 = HERE / "out" / "t1p_fm8"
OUT = HERE / "out"
st.NAMES["fm8"] = "LimiX-16M"
st.DEV["fm8"] = ("s7a_summary.csv", "fm8")


def load(metric: str = "macro_f1") -> pd.DataFrame:
    """Rows of all FMs; ``metric`` is renamed to ``macro_f1`` so summarize_t1p's code runs on it."""
    files = sorted(RUNS8.glob("*_fm8.parquet"))
    if len(files) != 52:
        raise SystemExit(f"expected 52 fm8 tables, found {len(files)}")
    files += sorted((HERE / "out" / "t1p").glob("*_fm[67].parquet"))
    files += sorted(st.S12.glob("*_fm[1235].parquet"))
    cols = [c for c in st.COLS if c != "macro_f1"] + [metric]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files]).rename(columns={metric: "macro_f1"})
    return df[df.strategy.isin(["stratified", "random"])]


def dev_accuracy_gap() -> dict[str, float]:
    """LimiX DEV accuracy gap per m and mean, by summarize.py's method on out/s7a (as s7a_summary.csv)."""
    import summarize as sd

    df = sd.load(HERE / "out" / "s7a").drop(columns="macro_f1").rename(columns={"accuracy": "macro_f1"})
    _, s = sd.summary(df)
    s = s[s.fm == "fm8"]
    return {str(m): float(g) for m, g in zip(s.mult, s.gap)}


def accuracy_summary() -> pd.DataFrame:
    dev, st.DEV = st.DEV, {}
    try:
        s = st.summary(st.cells(load("accuracy")))
    finally:
        st.DEV = dev
    s["p_holm"] = np.nan
    for _, g in s[(s.fm == "fm8") & (s.mult.astype(str) != "mean")].groupby(["set", "subset"]):
        s.loc[g.index, "p_holm"] = holm(g.wilcoxon_p.tolist())
    devacc = dev_accuracy_gap()
    sel = (s.fm == "fm8") & (s.subset == "all")
    s.loc[sel, "dev_gap"] = s.loc[sel, "mult"].astype(str).map(devacc)
    return s


def f(x: float, sign: bool = True) -> str:
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def pf(p: float) -> str:
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def macros(s: pd.DataFrame) -> dict[str, tuple[str, str]]:
    src = "fm_extra/out/t1p_fm8_summary.csv (EXPLORATORY; LimiX-16M on T1', Study 12 draws; stratified - random)"
    mac: dict[str, tuple[str, str]] = {}
    d = s[(s.set == "all") & (s.subset == "all") & (s.fm == "fm8")]
    for _, r in d.iterrows():
        mt = dict(mean="All", **{"1": "MOne", "3": "MThree", "5": "MFive", "10": "MTen"})[str(r.mult)]
        k = f"ExLimixTpGap{mt}"
        mac[k] = (f(r.gap), src)
        mac[k + "Lo"] = (f(r.ci_lo), src)
        mac[k + "Hi"] = (f(r.ci_hi), src)
        mac[k + "P"] = (pf(r.wilcoxon_p), src + " (Wilcoxon, unadjusted)")
        if str(r.mult) != "mean":
            mac[k + "Holm"] = (pf(r.p_holm), src + " (Holm over the 4 budgets of LimiX-16M)")
        mac[k + "Pos"] = (str(int(round(r.win * r.tables))), src + "; tables with positive mean gap")
        mac[k + "Strat"] = (f(r.mf1_strat, False), src + "; mean macro-F1, stratified")
        mac[k + "Random"] = (f(r.mf1_random, False), src + "; mean macro-F1, random")
    mac["ExLimixTpN"] = (str(int(d.tables.iloc[0])), src)
    pmax = d[d.mult.astype(str) != "mean"].wilcoxon_p.max()
    mac["ExLimixTpGapMaxP"] = (pf(pmax), src + " (largest unadjusted Wilcoxon p over the 4 budgets)")
    for sub, tag in (("binary", "Bin"), ("multiclass", "Multi")):
        r = s[(s.set == "all") & (s.subset == sub) & (s.fm == "fm8") & (s.mult.astype(str) == "mean")].iloc[0]
        mac[f"ExLimixTp{tag}GapAll"] = (f(r.gap), src + f"; {sub} tables")
        mac[f"ExLimixTp{tag}GapAllLo"] = (f(r.ci_lo), src + f"; {sub} tables")
        mac[f"ExLimixTp{tag}GapAllHi"] = (f(r.ci_hi), src + f"; {sub} tables")
    r = s[(s.set == "excl257") & (s.subset == "all") & (s.fm == "fm8") & (s.mult.astype(str) == "mean")].iloc[0]
    mac["ExLimixTpExclUCIGapAll"] = (f(r.gap), src + "; 51 tables without UCI 257")
    return mac


def acc_macros(a: pd.DataFrame) -> dict[str, tuple[str, str]]:
    src = ("fm_extra/out/t1p_fm8_acc_summary.csv (EXPLORATORY; LimiX-16M on T1', Study 12 draws; "
           "stratified - random; ACCURACY on E \\ L)")
    mac: dict[str, tuple[str, str]] = {}
    d = a[(a.set == "all") & (a.subset == "all") & (a.fm == "fm8")]
    for _, r in d.iterrows():
        mt = dict(mean="All", **{"1": "MOne", "3": "MThree", "5": "MFive", "10": "MTen"})[str(r.mult)]
        k = f"ExLimixTpAccGap{mt}"
        mac[k] = (f(r.gap), src)
        mac[k + "Lo"] = (f(r.ci_lo), src)
        mac[k + "Hi"] = (f(r.ci_hi), src)
        mac[k + "P"] = (pf(r.wilcoxon_p), src + " (Wilcoxon, unadjusted)")
        if str(r.mult) != "mean":
            mac[k + "Holm"] = (pf(r.p_holm), src + " (Holm over the 4 budgets of LimiX-16M)")
        mac[k + "Pos"] = (str(int(round(r.win * r.tables))), src + "; tables with positive mean gap")
        mac[f"ExLimixDevAccGap{mt}"] = (f(r.dev_gap), "fm_extra/out/s7a (EXPLORATORY; LimiX-16M on DEV, "
                                        "Study 7 Part (a) sets; accuracy gap, summarize.py method)")
    return mac


def write_tex(mac: dict[str, tuple[str, str]]) -> None:
    lines = ["% Generated by paper/p1/exploratory/fm_extra/summarize_t1p_fm8.py. Do not edit.",
             "% EXPLORATORY (not pre-registered; plan limix_t1p/PLAN.md): LimiX-16M on the 52 T1' tables."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}} % {c}" for k, (v, c) in sorted(mac.items())]
    (OUT / "t1p_fm8_numbers.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = load()
    c = st.cells(df)
    s = st.summary(c)
    s["p_holm"] = np.nan
    for key, g in s[(s.fm == "fm8") & (s.mult.astype(str) != "mean")].groupby(["set", "subset"]):
        s.loc[g.index, "p_holm"] = holm(g.wilcoxon_p.tolist())
    s.to_csv(OUT / "t1p_fm8_summary.csv", index=False)
    c[c.learner == "fm8"].to_csv(OUT / "t1p_fm8_cells.csv", index=False)
    secs = df[df.learner == "fm8"].seconds
    L = ["# EXPLORATORY: fm8 LimiX-16M on T1' (Study 12 design, arm T)", "",
         "Plan: limix_t1p/PLAN.md (committed before the run). Same Study 12 selections, seeds, "
         "permutation and panel as fm6/fm7 (run_t1p.py); stratified and random only. Macro-F1 on E \\ L; "
         "gap = stratified - random; mean over tables [10k bootstrap 95% CI]; Wilcoxon two-sided, "
         "unadjusted (as fm6/fm7), plus Holm over LimiX's 4 budgets. fm1-fm5 = stored Study 12; "
         "fm6/fm7 = out/t1p.", "",
         f"LimiX seconds per fit: mean {secs.mean():.1f}, median {secs.median():.1f}, "
         f"worker-hours {secs.sum() / 3600:.1f}", ""]
    cols = ["fm", "model", "mult", "tables", "mf1_strat", "mf1_random", "gap", "ci_lo", "ci_hi",
            "wilcoxon_p", "p_holm", "win", "dev_gap"]
    for set_name in ("all", "excl257"):
        g = s[(s.set == set_name) & (s.subset == "all")]
        L += [f"## {set_name}", "", g[cols].round(4).to_markdown(index=False), ""]
    g = s[(s.set == "all") & (s.subset != "all") & (s.mult == "mean")]
    L += ["## binary vs multiclass, mean over m", "",
          g[["subset"] + cols[:-2]].round(4).to_markdown(index=False), ""]
    a = accuracy_summary()
    a.to_csv(OUT / "t1p_fm8_acc_summary.csv", index=False)
    acols = [c for c in cols]
    g = a[(a.set == "all") & (a.subset == "all")].rename(
        columns={"mf1_strat": "acc_strat", "mf1_random": "acc_random"})
    L += ["## accuracy on E \\ L, all 52 tables (dev_gap = LimiX DEV accuracy gap)", "",
          g[[c.replace("mf1", "acc") for c in acols]].round(4).to_markdown(index=False), ""]
    (OUT / "T1P_FM8_SUMMARY.md").write_text("\n".join(L) + "\n")
    mac = macros(s)
    mac.update(acc_macros(a))
    write_tex(mac)
    print("\n".join(L))


if __name__ == "__main__":
    main()
