"""Analysis of the independent-learner-seed sensitivity (EXPLORATORY; see PLAN.md).

Reads out/dev/*.parquet (and, if present, results/seed_indep/heldout/*.parquet), writes
out/summary_gaps.csv, out/summary_per_m.csv, out/summary_decomp.csv, out/SUMMARY.md and
seed_indep_numbers.tex. HELD-OUT enters only as aggregates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_seed_indep import CLASSICAL, DEV_OUT, FMS, HO_OUT, METRICS, OUT  # noqa: E402
from simaudit.analysis import study12 as a12  # noqa: E402
from simaudit.analysis.report import bootstrap_ci  # noqa: E402
from simaudit.analysis.study7 import wilcoxon_p  # noqa: E402

MULTS = (1, 3, 5, 10)
MARGIN = 0.01
CHANGE = 0.01
NAMES = dict(fm1="TabPFN v2", fm2="TabICL", fm3="Kumo", fm5="TabPFN v3.5")
TEX_NAME = dict(fm1="FmOne", fm2="FmTwo", fm3="FmThree", fm5="FmFive", lr="Lr",
                knn1="KnnOne", knn3="KnnThree", gnb="Gnb", lda="Lda", hgb="Hgb")


def load(folder: Path) -> pd.DataFrame:
    files = [p for p in sorted(folder.glob("*.parquet")) if ".tmp" not in p.name]
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True) if files else pd.DataFrame()


def stats_row(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    lo, hi = bootstrap_ci(x)
    return dict(n=len(x), mean=float(x.mean()), lo=lo, hi=hi, p=wilcoxon_p(x))


def gap(runs: pd.DataFrame, learner: str, metric: str, mults=MULTS) -> pd.Series:
    means = a12.cell_means(runs[runs.learner == learner], metric)
    return a12.delta(means, (learner, "stratified"), (learner, "random"), tuple(mults), metric)


def decomp(runs: pd.DataFrame, learner: str, metric: str) -> pd.DataFrame:
    """Observed coverage (miss) and balance parts per table (E1 theory_table), mean over m."""
    rows = []
    sub = runs[runs.learner == learner]
    for (did, m), g in sub.groupby(["dataset", "mult"]):
        ms = g[g.strategy == "stratified"][metric].mean()
        r = g[g.strategy == "random"]
        miss = (r.coverage < 1).to_numpy()
        diff = (ms - r[metric]).to_numpy()
        rows.append(dict(dataset=did, mult=m, miss=float((diff * miss).mean()),
                         bal=float((diff * ~miss).mean())))
    return pd.DataFrame(rows).groupby("dataset")[["miss", "bal"]].mean()


def verdict(lo: float, hi: float, mean: float) -> str:
    if -MARGIN <= lo and hi <= MARGIN:
        return "materially unchanged"
    if (lo > 0 or hi < 0) and abs(mean) > MARGIN:
        return "changed"
    return "inconclusive"


def analyse(runs: pd.DataFrame, setname: str):
    orig = runs.copy()
    for k in METRICS:
        orig[k] = orig[f"orig_{k}"]
    gaps, per_m, dec = [], [], []
    for lr in [x for x in (*FMS, *CLASSICAL) if x in set(runs.learner)]:
        for metric in METRICS:
            go, gn = gap(orig, lr, metric), gap(runs, lr, metric)
            d = (gn - go).dropna()
            so, sn, sd = stats_row(go), stats_row(gn), stats_row(d)
            sub = runs[runs.learner == lr]
            delta = (sub[metric] - sub[f"orig_{metric}"]).abs()
            gaps.append(dict(
                set=setname, learner=lr, metric=metric, n_tables=sd["n"],
                gap_orig=so["mean"], gap_orig_lo=so["lo"], gap_orig_hi=so["hi"], gap_orig_p=so["p"],
                gap_indep=sn["mean"], gap_indep_lo=sn["lo"], gap_indep_hi=sn["hi"], gap_indep_p=sn["p"],
                diff=sd["mean"], diff_lo=sd["lo"], diff_hi=sd["hi"], diff_p=sd["p"],
                diff_max_abs_table=float(d.abs().max()),
                share_cells_changed=float((delta > CHANGE).mean()),
                mean_abs_cell_change=float(delta.mean()), n_cells=len(sub),
                verdict=verdict(sd["lo"], sd["hi"], sd["mean"]),
            ))
            if metric == "macro_f1":
                for m in MULTS:
                    dm = (gap(runs, lr, metric, (m,)) - gap(orig, lr, metric, (m,))).dropna()
                    s = stats_row(dm)
                    per_m.append(dict(set=setname, learner=lr, mult=m, diff=s["mean"],
                                      diff_lo=s["lo"], diff_hi=s["hi"], diff_p=s["p"]))
                po, pn = decomp(orig, lr, metric), decomp(runs, lr, metric)
                for part in ("miss", "bal"):
                    a, b = stats_row(po[part]), stats_row(pn[part])
                    s = stats_row((pn[part] - po[part]).to_numpy())
                    dec.append(dict(set=setname, learner=lr, part=part,
                                    orig=a["mean"], orig_lo=a["lo"], orig_hi=a["hi"],
                                    indep=b["mean"], indep_lo=b["lo"], indep_hi=b["hi"],
                                    diff=s["mean"], diff_lo=s["lo"], diff_hi=s["hi"], diff_p=s["p"],
                                    verdict=verdict(s["lo"], s["hi"], s["mean"])))
    return pd.DataFrame(gaps), pd.DataFrame(per_m), pd.DataFrame(dec)


def f(x: float) -> str:
    return f"{x:+.4f}"


def ci(m, lo, hi) -> str:
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def summary_md(gaps: pd.DataFrame, per_m: pd.DataFrame, dec: pd.DataFrame, repro: pd.DataFrame) -> str:
    L = ["# Independent learner seed: stratified − random gaps (EXPLORATORY)", "",
         "Same stored labelled sets; only the learner seed changes (namespace 0xF00D, spawned "
         "child; checked ≠ and disjoint from every selection seed). Gap = mean over m of "
         "(mean S − mean R) per table; mean over tables, 95% table-bootstrap CI (10k), "
         "Wilcoxon p. Reading fixed in PLAN.md: materially unchanged if the 95% CI of the "
         "difference of gaps lies within ±0.01.", ""]
    if len(repro):
        r = repro.groupby("learner")[["d_macro_f1", "d_proba"]].max()
        L += ["Reproduction control (original seed refitted, 20 cells per FM, same host): "
              + "; ".join(f"{k} max|Δ macro-F1| = {v.d_macro_f1:.2g}, max|Δ p| = {v.d_proba:.2g}"
                          for k, v in r.iterrows()), ""]
    for setname, g in gaps.groupby("set", sort=False):
        L += [f"## {setname}", "", "| learner | metric | gap orig | gap indep | indep − orig | "
              "Wilcoxon p (diff) | max |diff| table | cells |Δ|>0.01 | mean |Δ| cell | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            L.append(f"| {r.learner} | {r.metric} | {ci(r.gap_orig, r.gap_orig_lo, r.gap_orig_hi)} (p={r.gap_orig_p:.2g}) | "
                     f"{ci(r.gap_indep, r.gap_indep_lo, r.gap_indep_hi)} (p={r.gap_indep_p:.2g}) | "
                     f"{ci(r.diff, r.diff_lo, r.diff_hi)} | {r.diff_p:.2g} | {r.diff_max_abs_table:.4f} | "
                     f"{r.share_cells_changed:.3f} | {r.mean_abs_cell_change:.4f} | {r.verdict} |")
        L += ["", "Difference of macro-F1 gaps per m (indep − orig):", "",
              "| learner | m=1 | m=3 | m=5 | m=10 |", "|---|---|---|---|---|"]
        pm = per_m[per_m.set == setname]
        for lr, h in pm.groupby("learner", sort=False):
            L.append(f"| {lr} | " + " | ".join(ci(x.diff, x.diff_lo, x.diff_hi) for x in h.itertuples()) + " |")
        L += ["", "Coverage (miss) and balance parts of the macro-F1 gap:", "",
              "| learner | part | orig | indep | indep − orig | verdict |", "|---|---|---|---|---|---|"]
        for x in dec[dec.set == setname].itertuples():
            L.append(f"| {x.learner} | {x.part} | {ci(x.orig, x.orig_lo, x.orig_hi)} | "
                     f"{ci(x.indep, x.indep_lo, x.indep_hi)} | {ci(x.diff, x.diff_lo, x.diff_hi)} | {x.verdict} |")
        L.append("")
    return "\n".join(L) + "\n"


def tex_num(x: float, digits: int = 3) -> str:
    s = f"{x:+.{digits}f}"
    return ("$-$" if s[0] == "-" else "$+$") + s[1:]


def macros(gaps: pd.DataFrame, dec: pd.DataFrame) -> str:
    L = ["% AUTO-GENERATED by paper/p1/exploratory/seed_indep/analyze_seed_indep.py. "
         "EXPLORATORY (independent learner seed; not pre-registered). Do not edit."]

    def add(name: str, value: str, src: str) -> None:
        L.append(f"\\newcommand{{\\ExSeedInd{name}}}{{{value}}} % EXPLORATORY; source: {src}")

    for r in gaps[gaps.metric == "macro_f1"].itertuples():
        s = "Dev" if r.set == "DEV" else "Ho"
        p = f"{s}{TEX_NAME[r.learner]}"
        src = f"out/summary_gaps.csv ({r.set}, {r.learner}, macro_f1)"
        for k in ("gap_orig", "gap_indep", "diff", "diff_lo", "diff_hi"):
            name = "".join(w.capitalize() for w in k.split("_"))
            add(p + name, tex_num(getattr(r, k), 4 if k.startswith("diff") else 3), src)
        add(p + "DiffP", f"{r.diff_p:.2g}", src)
        add(p + "ShareChanged", f"{r.share_cells_changed:.3f}", src)
        add(p + "Verdict", r.verdict, src)
    sub = gaps[gaps.metric == "macro_f1"]
    for s, g in sub.groupby("set"):
        tag = "Dev" if s == "DEV" else "Ho"
        add(f"{tag}MaxAbsDiffCI", f"{max(g.diff_lo.abs().max(), g.diff_hi.abs().max()):.4f}",
            f"out/summary_gaps.csv ({s}, macro_f1, all learners)")
        add(f"{tag}AllUnchanged", "yes" if (g.verdict == "materially unchanged").all() else "no",
            f"out/summary_gaps.csv ({s}, macro_f1, all learners)")
    for x in dec.itertuples():
        s = "Dev" if x.set == "DEV" else "Ho"
        part = "Miss" if x.part == "miss" else "Bal"
        src = f"out/summary_decomp.csv ({x.set}, {x.learner}, {x.part})"
        add(f"{s}{TEX_NAME[x.learner]}{part}Diff", tex_num(x.diff, 4), src)
    return "\n".join(L) + "\n"


def main() -> None:
    parts = []
    dev = load(DEV_OUT)
    if len(dev):
        parts.append(analyse(dev, "DEV"))
    ho = load(HO_OUT)
    if len(ho):
        parts.append(analyse(ho, "HELD-OUT"))
    gaps = pd.concat([p[0] for p in parts], ignore_index=True)
    per_m = pd.concat([p[1] for p in parts], ignore_index=True)
    dec = pd.concat([p[2] for p in parts], ignore_index=True)
    gaps.to_csv(OUT / "summary_gaps.csv", index=False)
    per_m.to_csv(OUT / "summary_per_m.csv", index=False)
    dec.to_csv(OUT / "summary_decomp.csv", index=False)
    repro_path = OUT / "repro.csv"
    repro = pd.read_csv(repro_path) if repro_path.exists() else pd.DataFrame()
    text = summary_md(gaps, per_m, dec, repro)
    (OUT / "SUMMARY.md").write_text(text)
    (HERE / "seed_indep_numbers.tex").write_text(macros(gaps, dec))
    print(text)


if __name__ == "__main__":
    main()
