"""Appendix tables for the P1 draft (generated, never typed by hand).

Run:  .venv/bin/python paper/p1/make_tables.py   (or: make -C paper/p1 tables)

Reads only results/study7/a/{did}_{fm}.parquet (Study 7 Part (a), DEV, read-only)
and reports/STUDY7_REPORT.md (read-only), and writes:

paper/p1/tables_p1.tex with three tables:
  1. per DEV dataset: number of classes, and the stratified - random gap in macro-F1
     and in accuracy per FM (mean over reps, then mean over mult in {1, 3, 5, 10}),
     which is the per-dataset value of the confirmatory contrast (i) of Study 7;
  2. the stratified - random gap under six metric conventions (parsed from the
     per-budget contrast tables of STUDY7_REPORT.md);
  3. EXPLORATORY, not pre-registered: a "redraw" protocol, i.e. the random draws
     that cover every class (equal in distribution to drawing at random and
     redrawing until every class appears), computed from the stored random runs.

  4. per DEV dataset: pool size n, number of classes C, smallest class share,
     total-variation distance from the uniform prior, and the share of the stored
     random draws of Study 7 that missed a class at each budget, next to its exact
     hypergeometric probability (read from exploratory/out/e10_pools_dev.csv,
     written by exploratory/e10_split_and_pools.py; exploratory);
  5. the same pool table for the T1' tables of Study 9 (exploratory/out/
     e10_pools_t1prime.csv; random draws of Study 9, TabICLv2 file).

Tables 2 and 3 go to their own files, paper/p1/tables_metrics_p1.tex and
paper/p1/tables_redraw_p1.tex, so that the paper can input them in the appendix
that cites them; tables_p1.tex keeps tables 1, 4 and 5. An optional first argument
names another output directory (used to check a regeneration without overwriting).

paper/p1/tables_numbers_p1.tex with the few macros that the main text quotes
from table 3, and ranges over the four FMs of the binary/multiclass split E10
(exploratory/out/e10_split.csv, read-only).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "study7" / "a"
S7 = ROOT / "reports" / "STUDY7_REPORT.md"
HERE = Path(__file__).resolve().parent
OUT = HERE / "tables_p1.tex"
OUT_NUM = HERE / "tables_numbers_p1.tex"
OUT_METRICS = HERE / "tables_metrics_p1.tex"  # table 2 (tab:metrics)
OUT_REDRAW = HERE / "tables_redraw_p1.tex"  # table 3 (tab:redraw)
EXPL = HERE / "exploratory" / "out"  # written by the exploratory scripts; read-only here
FMS = ["fm1", "fm2", "fm3", "fm5"]  # TabPFN v2, TabICL, Kumo-Tabular, TabPFN v3.5
MULTS = [1, 3, 5, 10]
METRICS = ["macro_f1", "accuracy"]
METRIC_ROWS = [  # (metric key in STUDY7_REPORT.md, label in the table)
    ("macro_f1", "macro-F1 (all classes)"),
    ("accuracy", "accuracy"),
    ("balanced_accuracy", "balanced accuracy"),
    ("auc", "AUC"),
    ("log_loss", "log-loss (lower better)"),
    ("seen_macro_f1", "macro-F1 (seen classes)"),
]
NUM = r"([+\-−]?\d+\.\d+)"


def load_runs(columns: list[str]) -> pd.DataFrame:
    """Stratified and random runs of Study 7, Part (a), at the four budgets."""
    frames = []
    for path in sorted(SRC.glob("*_fm*.parquet")):
        df = pd.read_parquet(path, columns=["dataset", "learner", "strategy", "mult", *columns])
        frames.append(df[df["strategy"].isin(["stratified", "random"]) & df["mult"].isin(MULTS)])
    return pd.concat(frames, ignore_index=True)


def per_dataset_gaps() -> pd.DataFrame:
    """Return one row per dataset: C and the strat - random gap per FM and metric."""
    runs = load_runs(["n_classes", *METRICS])
    cell = runs.groupby(["dataset", "learner", "strategy", "mult"])[METRICS].mean().reset_index()
    wide = cell.pivot_table(index=["dataset", "learner", "mult"], columns="strategy", values=METRICS)
    rows = []
    for metric in METRICS:
        gap = (wide[(metric, "stratified")] - wide[(metric, "random")]).rename("gap").reset_index()
        gap = gap.groupby(["dataset", "learner"])["gap"].mean().unstack("learner")
        gap.columns = [f"{metric}_{c}" for c in gap.columns]
        rows.append(gap)
    table = pd.concat(rows, axis=1)
    table["C"] = runs.groupby("dataset")["n_classes"].first()
    return table.sort_values(["C"]).reset_index()


def metric_conventions() -> dict[str, list[tuple[float, float, int]]]:
    """Per metric and budget: (min gap, max gap, #FMs with Wilcoxon p < 0.05) over the four FMs."""
    text = S7.read_text(encoding="utf-8")
    out = {}
    for metric, _ in METRIC_ROWS:
        start = text.index(f"## Per-budget contrasts, {metric} (")
        block = text[start:text.index("\n## ", start + 5)]
        per_budget: list[list[tuple[float, float]]] = [[] for _ in MULTS]
        for line in block.splitlines():
            m = re.match(r"\|\s*strat − random (fm[1235])\s*\|(.*)", line)
            if not m:
                continue
            cells = [c for c in m.group(2).split("|") if c.strip()]
            for j, c in enumerate(cells[: len(MULTS)]):
                g = re.match(NUM + r" \(p=([\d.]+)", c.strip())
                per_budget[j].append((float(g.group(1).replace("−", "-")), float(g.group(2))))
        assert all(len(b) == len(FMS) for b in per_budget), metric
        out[metric] = [(min(v for v, _ in b), max(v for v, _ in b), sum(p < 0.05 for _, p in b))
                       for b in per_budget]
    return out


def redraw_protocol() -> tuple[pd.DataFrame, dict[str, float]]:
    """Exploratory: stratified versus the random draws that cover every class."""
    runs = load_runs(["coverage", "macro_f1"])
    rand = runs[runs["strategy"] == "random"]
    arms = {
        "strat": runs[runs["strategy"] == "stratified"],
        "random": rand,
        "redraw": rand[rand["coverage"] >= 1.0],
    }
    cell = {k: v.groupby(["dataset", "learner", "mult"])["macro_f1"].mean() for k, v in arms.items()}
    gap = (cell["strat"] - cell["redraw"]).dropna()
    rows = []
    for m in MULTS:
        row: dict[str, float] = {"m": m, "covered_share": float((rand.loc[rand["mult"] == m, "coverage"] >= 1.0).mean())}
        for fm in FMS:
            x = gap.xs((fm, m), level=("learner", "mult"))
            row[f"gap_{fm}"] = float(x.mean())
            row[f"p_{fm}"] = float(wilcoxon(x).pvalue) if (x != 0).any() else 1.0
            row["n_datasets"] = len(x)
        for arm in ["strat", "random", "redraw"]:
            t = cell[arm].xs(m, level="mult").unstack("learner")
            d = (t["fm2"] - t["fm5"]).dropna()
            row[f"rank_{arm}"] = float(d.mean())
            row[f"rankp_{arm}"] = float(wilcoxon(d).pvalue)
        rows.append(row)
    tab = pd.DataFrame(rows)
    later = tab[tab["m"] > 1]
    gaps = later[[f"gap_{fm}" for fm in FMS]].to_numpy()
    nums = {
        "RedrawCoveredOne": float(tab.loc[tab["m"] == 1, "covered_share"].item()),
        "RedrawGapMin": float(gaps.min()),
        "RedrawGapMax": float(gaps.max()),
        "RedrawRankMin": float(later["rank_redraw"].min()),
        "RedrawRankMax": float(later["rank_redraw"].max()),
        "RedrawRankPMax": float(later["rankp_redraw"].max()),
    }
    return tab, nums


def signed(x: float) -> str:
    """Format a gap with an explicit sign, as in numbers_p1.tex."""
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.3f}"


def pcell(x: float, p: float) -> str:
    """Signed gap, with a dagger when the unadjusted Wilcoxon p < 0.05."""
    return signed(x) + ("$^\\dagger$" if p < 0.05 else "")


def table_per_dataset() -> tuple[list[str], int]:
    t = per_dataset_gaps()
    lines = [
        "\\begin{table}[ht]",
        "\\centering\\small",
        "\\caption{Per-dataset gap between the stratified and the random protocol on DEV (the protocol comparison of the DEV study). "
        "Each entry is the mean over repetitions, then over $m\\in\\{1,3,5,10\\}$, of stratified minus random. "
        "Positive values mean that the stratified protocol reports a higher score.}",
        "\\label{tab:perdataset}",
        "\\begin{tabular}{rr" + "r" * 8 + "}",
        "\\toprule",
        " & & \\multicolumn{4}{c}{macro-F1} & \\multicolumn{4}{c}{accuracy}\\\\",
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}",
        "Dataset & $C$ & v2 & TabICLv2 & Kumo & v3.5 & v2 & TabICLv2 & Kumo & v3.5\\\\",
        "\\midrule",
    ]
    for _, r in t.iterrows():
        cells = [str(int(r["dataset"])), str(int(r["C"]))]
        cells += [signed(r[f"macro_f1_{fm}"]) for fm in FMS]
        cells += [signed(r[f"accuracy_{fm}"]) for fm in FMS]
        lines.append(" & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return lines, len(t)


def table_metrics() -> list[str]:
    mc = metric_conventions()
    lines = [
        "\\begin{table}[ht]",
        "\\centering\\small",
        "\\caption{Stratified minus random on DEV (the protocol comparison of the DEV study) under six metric conventions. "
        "Each cell gives the smallest and the largest gap over the four FMs at $m$ labels per class and, "
        "in brackets, the number of FMs (out of four) with an unadjusted Wilcoxon $p<0.05$ over the "
        "\\DevN{} tables. For log-loss, a negative value means that the stratified protocol reports a better score. "
        "The seen-class metric scores macro-F1 only over the classes that appear in the context; at one label "
        "per class, a binary context that contains one class then scores perfectly on that class. "
        "AUC is the one-vs-rest mean over the classes in the context and is undefined for a context with one "
        "class, so it leaves out the draws that miss a class.}",
        "\\label{tab:metrics}",
        "\\setlength{\\tabcolsep}{3pt}\\footnotesize",
        "\\begin{tabular}{l" + "c" * len(MULTS) + "}",
        "\\toprule",
        "Metric & " + " & ".join(f"$m={m}$" for m in MULTS) + "\\\\",
        "\\midrule",
    ]
    for metric, label in METRIC_ROWS:
        cells = [f"{signed(lo)} to {signed(hi)} [{k}]" for lo, hi, k in mc[metric]]
        lines.append(label + " & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return lines


def table_redraw(rd: pd.DataFrame) -> list[str]:
    lines = [
        "\\begin{table}[ht]",
        "\\centering\\small",
        "\\caption{A redraw protocol on DEV (exploratory; not pre-registered). We keep only the random draws "
        "of the DEV study that contain every class; these draws have the same distribution as drawing at random and "
        "redrawing until every class appears. ``Covered'' is the share of random draws that contain every class, "
        "and ``tables'' is the number of DEV tables with at least one covered draw. The middle columns give "
        "stratified minus redraw in macro-F1; the last columns give TabICLv2 minus TabPFN v3.5 in macro-F1 under "
        "the three protocols. $^\\dagger$: unadjusted Wilcoxon $p<0.05$ over tables. At $m=1$, a covered draw "
        "has exactly one label per class, so the redraw and the stratified protocol coincide in distribution, "
        "and the gap at $m=1$ is noise.}",
        "\\label{tab:redraw}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{rcr" + "r" * len(FMS) + "rrr}",
        "\\toprule",
        " & & & \\multicolumn{4}{c}{stratified $-$ redraw} & \\multicolumn{3}{c}{TabICLv2 $-$ TabPFN v3.5}\\\\",
        "\\cmidrule(lr){4-7}\\cmidrule(lr){8-10}",
        "$m$ & covered & tables & v2 & TabICLv2 & Kumo & v3.5 & stratified & random & redraw\\\\",
        "\\midrule",
    ]
    for _, r in rd.iterrows():
        cells = [str(int(r["m"])), f"{r['covered_share']:.2f}", str(int(r["n_datasets"]))]
        cells += [pcell(r[f"gap_{fm}"], r[f"p_{fm}"]) for fm in FMS]
        cells += [pcell(r[f"rank_{a}"], r[f"rankp_{a}"]) for a in ["strat", "random", "redraw"]]
        lines.append(" & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return lines


def tex_escape(text: str, width: int = 26) -> str:
    """Shorten a table name and escape the characters that TeX treats specially."""
    text = str(text)
    if len(text) > width:
        text = text[: width - 1].rstrip() + "."
    for a, b in [("\\", ""), ("_", "\\_"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("$", "\\$")]:
        text = text.replace(a, b)
    return text


def miss_cell(obs: float, exact: float) -> str:
    """Observed share of missed draws, with the exact probability in brackets."""
    return f"{obs:.2f} ({exact:.2f})"


def table_pools_dev() -> list[str]:
    t = pd.read_csv(EXPL / "e10_pools_dev.csv").sort_values(["C", "tv"])
    lines = [
        "\\begin{table}[ht]",
        "\\centering\\small",
        "\\caption{DEV pools (exploratory). $n$: pool size; $\\pi_{\\min}$: smallest class share; "
        "TV: total-variation distance $\\frac12\\lVert\\pi-u\\rVert_1$ of the class prior from the uniform vector. "
        "The last columns give, at $m$ labels per class, the share of the stored random draws of the DEV study "
        "(\\SsevRandomReps{} per table) that missed at least one class and, in brackets, the exact "
        "probability of a miss under the multivariate hypergeometric law.}",
        "\\label{tab:poolsdev}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{rrrrr" + "r" * len(MULTS) + "}",
        "\\toprule",
        " & & & & & \\multicolumn{" + str(len(MULTS)) + "}{c}{share of random draws that miss a class}\\\\",
        "\\cmidrule(lr){6-" + str(5 + len(MULTS)) + "}",
        "Dataset & $n$ & $C$ & $\\pi_{\\min}$ & TV & " + " & ".join(f"$m={m}$" for m in MULTS) + "\\\\",
        "\\midrule",
    ]
    for _, r in t.iterrows():
        cells = [str(int(r["dataset"])), str(int(r["n"])), str(int(r["C"])), f"{r['pi_min']:.3f}", f"{r['tv']:.3f}"]
        cells += [miss_cell(r[f"miss_obs_m{m}"], r[f"miss_exact_m{m}"]) for m in MULTS]
        lines.append(" & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return lines


def table_pools_t1prime() -> list[str]:
    t = pd.read_csv(EXPL / "e10_pools_t1prime.csv").sort_values(["C", "pi_min", "name"], ascending=[True, False, True])
    budgets = [1, 3]
    lines = [
        "\\begin{table}[p]",
        "\\centering\\footnotesize",
        "\\caption{PMLB/UCI pools of the learning-curve study (exploratory). Columns as in Table~\\ref{tab:poolsdev}; the miss "
        "shares come from the random draws of that study (10 per budget, TabICLv2 file), with the exact probability "
        "in brackets. Long table names are shortened.}",
        "\\label{tab:poolst1}",
        "\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{0.92}",
        "\\begin{tabular}{lrrrr" + "r" * len(budgets) + "}",
        "\\toprule",
        "Table & $n$ & $C$ & $\\pi_{\\min}$ & TV & " + " & ".join(f"miss, $m={m}$" for m in budgets) + "\\\\",
        "\\midrule",
    ]
    for _, r in t.iterrows():
        cells = [tex_escape(r["name"]), str(int(r["n"])), str(int(r["C"])), f"{r['pi_min']:.3f}", f"{r['tv']:.3f}"]
        cells += [miss_cell(r[f"miss_obs_m{m}"], r[f"miss_exact_m{m}"]) for m in budgets]
        lines.append(" & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return lines


def split_ranges() -> dict[str, str]:
    """E10 (exploratory): ranges over the four FMs of the binary and multiclass gaps, mean over m."""
    sp = pd.read_csv(EXPL / "e10_split.csv")
    sp = sp[sp["mult"].astype(str) == "all"]
    out: dict[str, str] = {}
    for metric, key in [("dM", "F"), ("dA", "Acc")]:
        x = sp[sp["metric"] == metric]
        assert len(x) == len(FMS), metric
        for col, name in [("bin_mean", "Bin"), ("mc_mean", "Multi")]:
            out[f"Split{key}{name}Min"] = signed(x[col].min())
            out[f"Split{key}{name}Max"] = signed(x[col].max())
        out[f"Split{key}MWpMin"] = f"{x['mw_p'].min():.2f}"
    return out


def main(out_dir: Path = HERE) -> None:
    head = ["% AUTO-GENERATED by paper/p1/make_tables.py from results/study7/a and "
            "reports/STUDY7_REPORT.md (read-only). Do not edit."]
    per_ds, n = table_per_dataset()
    rd, nums = redraw_protocol()
    (out_dir / OUT.name).write_text("\n".join(head + per_ds + table_pools_dev()
                                              + table_pools_t1prime()))
    (out_dir / OUT_METRICS.name).write_text("\n".join(head + table_metrics()))
    (out_dir / OUT_REDRAW.name).write_text("\n".join(head + table_redraw(rd)))

    fmt = {
        "RedrawCoveredOne": f"{nums['RedrawCoveredOne']:.2f}",
        "RedrawGapMin": signed(nums["RedrawGapMin"]),
        "RedrawGapMax": signed(nums["RedrawGapMax"]),
        "RedrawRankMin": signed(nums["RedrawRankMin"]),
        "RedrawRankMax": signed(nums["RedrawRankMax"]),
        "RedrawRankPMax": f"{nums['RedrawRankPMax']:.3f}",
    }
    fmt.update(split_ranges())
    num_lines = ["% AUTO-GENERATED by paper/p1/make_tables.py (results/study7/a, read-only; "
                 "exploratory redraw protocol; Split*: exploratory/out/e10_split.csv). Do not edit."]
    num_lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in fmt.items()]
    (out_dir / OUT_NUM.name).write_text("\n".join(num_lines) + "\n")
    print(f"wrote {OUT.name}, {OUT_METRICS.name}, {OUT_REDRAW.name} ({n} datasets, "
          f"5 tables) and {OUT_NUM.name} in {out_dir}")


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
