"""EXPLORATORY analysis of the E1/E13 classical-learner control on T1' (PLAN.md).

Reads out/runs/*.parquet (run_e1_t1p.py) and the stored Study 12 FM runs
(results/study12/{key}_fm{1,2,3,5}.parquet, stratified/random rows, read-only).
Gap = stratified - random per table (mean over reps); mean over tables, 10k bootstrap
95% CI, win rate, two-sided Wilcoxon (common.summarize); Holm within metric x m over
the 6 classical + 4 FM learners (FMs context only), as DEV E1. E13: D = gap_FM -
gap_classical (mean over LR, 1-NN, GNB, LDA, HGB), Holm over the 4 FMs. Two table
sets: all 52 and 51 without UCI 257 (pmlb_1000000257).

Writes out/e1t1p_gaps.csv, out/e13t1p_diff.csv, out/SUMMARY.md, out/e1t1p_numbers.tex.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd  # noqa: E402

import common as cm  # noqa: E402

OUT = HERE / "out"
S12 = cm.settings.results_dir / "study12"
UCI257 = "pmlb_1000000257"
LEARNERS = ("lr", "knn1", "knn3", "gnb", "lda", "hgb")
CLASSICAL = ("lr", "knn1", "gnb", "lda", "hgb")
METRICS = ("macro_f1", "accuracy", "balanced_accuracy")
SETS = {"all": "", "excl257": "NoUCI"}
TAG = dict(lr="LR", knn1="KnnOne", knn3="KnnThree", gnb="GNB", lda="LDA", hgb="HGB")
FT = dict(fm1="FMone", fm2="FMtwo", fm3="FMthree", fm5="FMfive")
MT = {"1": "MOne", "3": "MThree", "5": "MFive", "10": "MTen", "all": "All"}
MTAG = dict(macro_f1="F", accuracy="Acc", balanced_accuracy="BA")


def load_runs() -> pd.DataFrame:
    cl = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "runs").glob("*.parquet"))])
    keys = sorted(cl.dataset.unique())
    cols = ["dataset", "learner", "strategy", "mult", "rep", *METRICS]
    fm = pd.concat([pd.read_parquet(S12 / f"{k}_{f}.parquet", columns=cols)
                    for k in keys for f in cm.FMS])
    fm = fm[fm.strategy.isin(["stratified", "random"])]
    return pd.concat([cl[cols], fm], ignore_index=True)


def table_gaps(runs: pd.DataFrame) -> pd.DataFrame:
    """Per (learner, dataset, mult): stratified - random for each metric."""
    w = runs.groupby(["learner", "dataset", "mult", "strategy"])[list(METRICS)].mean().unstack("strategy")
    return pd.DataFrame({m: w[(m, "stratified")] - w[(m, "random")] for m in METRICS}).reset_index()


def gap_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for (lr, m), g in d.groupby(["learner", "mult"]):
            rows.append(dict(learner=lr, metric=metric, mult=str(m), **cm.summarize(g[metric].to_numpy())))
        for lr, g in d.groupby(["learner", "dataset"])[metric].mean().reset_index().groupby("learner"):
            rows.append(dict(learner=lr, metric=metric, mult="all", **cm.summarize(g[metric].to_numpy())))
    out = pd.DataFrame(rows)
    return pd.concat([cm.add_holm(g) for _, g in out.groupby(["metric", "mult"])])


def contrast(d: pd.DataFrame) -> pd.DataFrame:
    cl = d[d.learner.isin(CLASSICAL)]
    ref = cl.groupby(["dataset", "mult"])[list(METRICS)].mean().reset_index().assign(learner="classical_mean")
    cl = pd.concat([cl, ref])
    fm = d[d.learner.isin(cm.FMS)].rename(columns={"learner": "fm"})
    rows = []
    for lr, gl in cl.groupby("learner"):
        j = fm.merge(gl.drop(columns="learner"), on=["dataset", "mult"], suffixes=("_fm", "_cl"))
        for metric in METRICS:
            j["D"] = j[f"{metric}_fm"] - j[f"{metric}_cl"]
            j["a"], j["b"] = j[f"{metric}_fm"], j[f"{metric}_cl"]
            avg = j.groupby(["fm", "dataset"])[["D", "a", "b"]].mean().reset_index().assign(mult="all")
            jj = pd.concat([j.assign(mult=j.mult.astype(str))[["fm", "dataset", "mult", "D", "a", "b"]], avg])
            for (f, m), g in jj.groupby(["fm", "mult"]):
                rows.append(dict(classical=lr, metric=metric, fm=f, mult=m, gap_fm=g.a.mean(),
                                 gap_cl=g.b.mean(), **cm.summarize(g.D.to_numpy())))
    out = pd.DataFrame(rows)
    return pd.concat([cm.add_holm(g) for _, g in out.groupby(["classical", "metric", "mult"], sort=False)])


def ci(r) -> str:
    return f"{r['mean']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"


# ------------------------------------------------------------------ macros
DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
MAC: dict[str, tuple[str, str]] = {}


def put(name: str, value: str, src: str) -> None:
    name = re.sub(r"[^A-Za-z]", "", "".join(DIG.get(c, c) for c in name))
    if name in MAC:
        raise KeyError(name)
    MAC[name] = (value, src)


def fx(x: float, nd: int = 3, sign: bool = True) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def pf(p: float) -> str:
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def ci3(k: str, r, src: str) -> None:
    put(k, fx(r["mean"]), src)
    put(k + "Lo", fx(r.ci_lo), src)
    put(k + "Hi", fx(r.ci_hi), src)


def macros(gaps: pd.DataFrame, diff: pd.DataFrame, ntab: dict[str, int]) -> None:
    for s, st in SETS.items():
        g = gaps[gaps.set == s]
        src = f"e1_t1p/out/e1t1p_gaps.csv (EXPLORATORY; T1' {s}, Study 12 draws; stratified - random)"
        put(f"ExEOneTp{st}NTables", str(ntab[s]), src)
        for lr, t in TAG.items():
            for metric in ("macro_f1", "accuracy"):
                for m in ("all", "1", "10"):
                    r = g[(g.learner == lr) & (g.metric == metric) & (g.mult == m)].iloc[0]
                    k = f"ExEOneTp{st}{t}{MTAG[metric]}{MT[m]}"
                    ci3(k, r, src)
                    if m == "all":
                        put(k + "Holm", pf(r.p_holm), src)
                        put(k + "Win", fx(r.win, 2, False), src)
        for metric in METRICS:
            for m in MT:
                c = g[g.learner.isin(CLASSICAL) & (g.metric == metric) & (g.mult == m)]
                put(f"ExEOneTp{st}Cls{MTAG[metric]}{MT[m]}Min", fx(c["mean"].min()), src)
                put(f"ExEOneTp{st}Cls{MTAG[metric]}{MT[m]}Max", fx(c["mean"].max()), src)
            a = g[g.learner.isin(LEARNERS) & (g.metric == metric) & (g.mult == "all")]
            put(f"ExEOneTp{st}NSig{MTAG[metric]}", str(int((a.p_holm < 0.05).sum())), src)
        dd = diff[(diff.set == s) & (diff.classical == "classical_mean")]
        src = f"e1_t1p/out/e13t1p_diff.csv (EXPLORATORY; T1' {s}; D = gap_FM - gap_classical)"
        for metric in ("macro_f1", "accuracy"):
            for f, t in FT.items():
                for m in MT:
                    r = dd[(dd.metric == metric) & (dd.fm == f) & (dd.mult == m)].iloc[0]
                    k = f"ExEThirteenTp{st}{MTAG[metric]}{t}{MT[m]}"
                    ci3(k + "D", r, src)
                    put(k + "Holm", pf(r.p_holm), src)
                    put(k + "GapFM", fx(r.gap_fm), src)
                    put(k + "GapCl", fx(r.gap_cl), src)
    lines = ["% Generated by paper/p1/exploratory/e1_t1p/analyze_e1_t1p.py. Do not edit.",
             "% EXPLORATORY (not pre-registered; plan e1_t1p/PLAN.md): E1/E13 classical-learner control on T1'."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}} % {src}" for k, (v, src) in sorted(MAC.items())]
    (OUT / "e1t1p_numbers.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ summary
def summary(gaps, diff, ntab, secs) -> None:
    dev = pd.read_csv(cm.OUT / "e1_gaps.csv")
    dev["mult"] = dev.mult.astype(str)
    dev13 = pd.read_csv(cm.OUT / "e13_diff.csv")
    dev13["mult"] = dev13.mult.astype(str)
    L = ["# E1/E13 on T1′: classical-learner control (EXPLORATORY)", "",
         "Plan: `PLAN.md` (committed before any fit). Study 12 draws, seeds, permutation and panel "
         "(E \\ L, E = first min(n, 2000) permuted rows); stratified (5) and random (20) reps; DEV E1 learners "
         "unchanged. Gap = stratified − random, per-table mean over reps; mean over tables [10k bootstrap 95% CI]; "
         "`*` Holm < 0.05 within metric × m over 6 classical + 4 FM learners (FMs = stored Study 12 runs, context). "
         f"Tables: {ntab['all']} (all), {ntab['excl257']} (without UCI 257). Fits: {secs[0]}; "
         f"fit time {secs[1]:.2f} CPU-min in total.", ""]
    for s in SETS:
        g = gaps[gaps.set == s]
        for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
            L += [f"## {metric}, {s} ({ntab[s]} tables): stratified − random", "",
                  "| learner | m=1 | m=3 | m=5 | m=10 | mean over m (p_holm, win) | DEV mean over m |",
                  "|---|---|---|---|---|---|---|"]
            for lr in list(LEARNERS) + list(cm.FMS):
                cells = []
                for m in ("1", "3", "5", "10", "all"):
                    r = g[(g.learner == lr) & (g.metric == metric) & (g.mult == m)].iloc[0]
                    c = ci(r)
                    c += f" (p={r.p_holm:.2g}, win={r.win:.2f})" if m == "all" else (" *" if r.p_holm < 0.05 else "")
                    cells.append(c)
                dv = dev[(dev.learner == lr) & (dev.metric == metric) & (dev.mult == "all")]
                cells.append(ci(dv.iloc[0]) if len(dv) else "")
                L.append(f"| {cm.FM_NAMES.get(lr, lr)} | " + " | ".join(cells) + " |")
            L.append("")
        for metric in ("macro_f1", "accuracy"):
            d = diff[(diff.set == s) & (diff.classical == "classical_mean") & (diff.metric == metric)]
            L += [f"## E13 on T1′, {metric}, {s}: D = Δ_FM − Δ_classical (mean of LR, 1-NN, GNB, LDA, HGB)", "",
                  "| FM | m | Δ_FM | Δ_classical | D [95% CI] | p_holm | win | DEV D |", "|---|---|---|---|---|---|---|---|"]
            dmet = dict(macro_f1="dM", accuracy="dA")[metric]
            for f in cm.FMS:
                for m in MT:
                    r = d[(d.fm == f) & (d.mult == m)].iloc[0]
                    dv = dev13[(dev13.classical == "classical_mean") & (dev13.metric == dmet) & (dev13.fm == f) & (dev13.mult == m)]
                    L.append(f"| {cm.FM_NAMES[f]} | {m} | {r.gap_fm:+.3f} | {r.gap_cl:+.3f} | {ci(r)}"
                             f"{' *' if r.p_holm < 0.05 else ''} | {r.p_holm:.2g} | {r.win:.2f} | "
                             f"{ci(dv.iloc[0]) if len(dv) else ''} |")
            L.append("")
        d = diff[(diff.set == s) & (diff.metric == "macro_f1") & (diff.mult == "all")]
        L += [f"## E13 per classical learner, macro-F1, mean over m, {s}: D [CI]", "",
              "| FM | " + " | ".join(CLASSICAL) + " |", "|---|" + "---|" * len(CLASSICAL)]
        for f in cm.FMS:
            L.append(f"| {cm.FM_NAMES[f]} | " + " | ".join(
                ci(d[(d.classical == lr) & (d.fm == f)].iloc[0]) for lr in CLASSICAL) + " |")
        L.append("")
    L += ["Note: 3-NN at m = 1 uses k = B = C (a vote over the whole context), degenerate as on DEV; "
          "it is excluded from the classical range and from the E13 reference."]
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    runs = load_runs()
    tg = table_gaps(runs)
    sets = {"all": tg, "excl257": tg[tg.dataset != UCI257]}
    ntab = {s: d.dataset.nunique() for s, d in sets.items()}
    gaps = pd.concat([gap_table(d).assign(set=s) for s, d in sets.items()])
    diff = pd.concat([contrast(d).assign(set=s) for s, d in sets.items()])
    gaps.to_csv(OUT / "e1t1p_gaps.csv", index=False)
    diff.to_csv(OUT / "e13t1p_diff.csv", index=False)
    cl = pd.concat([pd.read_parquet(p, columns=["seconds"]) for p in sorted((OUT / "runs").glob("*.parquet"))])
    summary(gaps, diff, ntab, (len(cl), cl.seconds.sum() / 60))
    macros(gaps, diff, ntab)


if __name__ == "__main__":
    main()
