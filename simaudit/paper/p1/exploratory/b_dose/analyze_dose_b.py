"""Analysis of the exploratory dose-response arms F, S and N (plan Sec. 6).

EXPLORATORY: DEV only, not pre-registered. Reads out/raw/*.parquet (this run)
and results/study7/c (the original Part (c), read-only, reference O). Writes
out/slopes.csv, out/summary.csv, out/contrasts.csv, out/jt.csv, out/SUMMARY.md
and out/b_numbers.tex (macros \\ExB...). No hand-typed numbers.

Usage (from simaudit/): .venv/bin/python paper/p1/exploratory/b_dose/analyze_dose_b.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import FM_NAMES, add_holm, dev_ids, summarize  # noqa: E402

from simaudit.analysis.study7 import jt_stratified, slope  # noqa: E402
from simaudit.settings import settings  # noqa: E402

FMS = ("fm2", "fm1", "fm5", "fm3")
MULTS = (3, 5, 10)
SIDES = ("majority", "minority")
ARMS = ("O", "F", "S", "N")
BASE = ("macro_f1", "accuracy", "balanced_accuracy")
METRICS = [p + m for p in ("", "pc_", "cm_", "cm_pc_") for m in BASE]
FM_TAG = dict(fm1="TabPFNTwo", fm2="TabICL", fm3="Kumo", fm5="TabPFNThreeFive")
MET_TAG = {
    "macro_f1": "F", "pc_macro_f1": "PcF", "cm_macro_f1": "CmF", "cm_pc_macro_f1": "CmPcF",
    "accuracy": "Acc", "pc_accuracy": "PcAcc", "balanced_accuracy": "BA", "pc_balanced_accuracy": "PcBA",
}
CONTRASTS = (("F", "O"), ("S", "O"), ("N", "F"))


# ----------------------------------------------------------------------------- load
def load_new(raw: Path, allow_partial: bool) -> tuple[pd.DataFrame, list[str], dict]:
    ids = dev_ids()
    frames, complete = [], []
    for fm in FMS:
        files = [raw / f"{fm}_{d}.parquet" for d in ids]
        have = [p for p in files if p.exists()]
        if allow_partial:
            have += [p.with_name(p.name.replace(".parquet", ".partial.parquet")) for p in files if not p.exists()]
            have = [p for p in have if p.exists()]
        if len([p for p in files if p.exists()]) == len(ids):
            complete.append(fm)
        elif not allow_partial:
            continue  # plan Sec. 4: an unfinished FM is excluded entirely
        frames += [pd.read_parquet(p) for p in have]
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    fms = sorted(set(new.learner)) if allow_partial and len(new) else complete
    new = new[new.learner.isin(fms)]
    fitted = new[new.feasible & new.fitted_here.fillna(False).astype(bool)]
    cost = dict(
        fits=int(len(fitted)),
        cpu_hours=float(fitted.seconds.sum() / 3600),
        cells=int(len(new)),
        feasible_cells=int(new.feasible.sum()),
        fits_by_fm={k: int(v) for k, v in fitted.groupby("learner").size().items()},
        hours_by_fm={k: float(v / 3600) for k, v in fitted.groupby("learner").seconds.sum().items()},
    )
    return new, fms, cost


def load_original(fms: list[str]) -> pd.DataFrame:
    cols = ["dataset", "arm", "share", "mult", "rep", "feasible", "n_classes", "macro_f1",
            "accuracy", "balanced_accuracy", "pc_macro_f1", "pc_accuracy", "pc_balanced_accuracy"]
    frames = []
    for fm in fms:
        for d in dev_ids():
            f = pd.read_parquet(settings.results_dir / "study7" / "c" / f"{d}_{fm}.parquet", columns=cols)
            frames.append(f.assign(learner=fm))
    o = pd.concat(frames, ignore_index=True)
    o = o[o.arm.isin(SIDES)].rename(columns=dict(arm="side", share="level")).assign(arm="O")
    return o


# ----------------------------------------------------------------------------- slopes
def table_slopes(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per (learner, arm, side, dataset): mean over mults of the OLS slope of rep-means on the level."""
    f = df[df.feasible & df[metric].notna()]
    means = f.groupby(["learner", "arm", "side", "dataset", "mult", "level"])[metric].mean().reset_index()
    rows = []
    for key, g in means.groupby(["learner", "arm", "side", "dataset", "mult"]):
        if g.level.nunique() < 2:
            continue
        g = g.sort_values("level")
        lo, hi = g.iloc[0], g.iloc[-1]
        drop = float(lo[metric] - hi[metric]) if lo.level == 0.5 else np.nan
        rows.append((*key, slope(g.level.to_numpy(), g[metric].to_numpy()), drop, g.level.nunique()))
    per_mult = pd.DataFrame(rows, columns=["learner", "arm", "side", "dataset", "mult", "slope", "drop", "levels"])
    out = per_mult.groupby(["learner", "arm", "side", "dataset"]).agg(
        slope=("slope", "mean"), drop=("drop", "mean"), mults=("mult", "nunique")
    ).reset_index()
    return out.assign(metric=metric)


def rep_slope_neg(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Share of (dataset, mult, rep) with a negative within-rep slope (for N; also F, S)."""
    f = df[df.feasible & df[metric].notna() & (df.arm != "O")]
    rows = []
    for key, g in f.groupby(["learner", "arm", "side", "dataset", "mult", "rep"]):
        if g.level.nunique() < 2:
            continue
        rows.append((*key[:3], slope(g.level.to_numpy(float), g[metric].to_numpy(float))))
    r = pd.DataFrame(rows, columns=["learner", "arm", "side", "s"])
    return r.groupby(["learner", "arm", "side"]).s.apply(lambda s: float(np.mean(s < 0))).rename("rep_neg_share").reset_index()


def jt(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for (fm, arm, side), g in df[df.feasible & df[metric].notna()].groupby(["learner", "arm", "side"]):
        strata = []
        for _, s in g.groupby(["dataset", "mult"]):
            groups = [lv[metric].to_numpy(float) for _, lv in s.groupby("level", sort=True)]
            strata.append(groups)
        res = jt_stratified(strata)
        rows.append(dict(learner=fm, arm=arm, side=side, metric=metric, jt_z=res["z"], jt_p=res["p"], strata=res["strata"]))
    return pd.DataFrame(rows)


def summarise(slopes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric, fm, arm, side), g in slopes.groupby(["metric", "learner", "arm", "side"]):
        s = summarize(g.slope.to_numpy())
        d = g["drop"].dropna().to_numpy()
        ds = summarize(d) if len(d) >= 2 else dict(mean=np.nan, ci_lo=np.nan, ci_hi=np.nan)
        rows.append(dict(metric=metric, learner=fm, arm=arm, side=side, n_tables=s["n"], slope_mean=s["mean"],
                         slope_lo=s["ci_lo"], slope_hi=s["ci_hi"], slope_median=s["median"],
                         neg_share=1 - s["win"] - float(np.mean(g.slope.to_numpy() == 0)), p=s["p"],
                         drop_mean=ds["mean"], drop_lo=ds["ci_lo"], drop_hi=ds["ci_hi"]))
    return pd.DataFrame(rows)


def contrasts(slopes: pd.DataFrame) -> pd.DataFrame:
    wide = slopes.pivot_table(index=["metric", "learner", "side", "dataset"], columns="arm", values="slope")
    rows = []
    pairs = list(CONTRASTS) + [("O", "F+S")]
    for (metric, fm, side), g in wide.groupby(level=[0, 1, 2]):
        for a, b in pairs:
            if b == "F+S":
                if not {"O", "F", "S"} <= set(g.columns):
                    continue
                v = (g["O"] - (g["F"] + g["S"])).dropna()
            else:
                if not {a, b} <= set(g.columns):
                    continue
                v = (g[a] - g[b]).dropna()
            if len(v) < 3:
                continue
            s = summarize(v.to_numpy())
            rows.append(dict(metric=metric, learner=fm, side=side, contrast=f"{a}-{b}", n_tables=s["n"],
                             mean=s["mean"], lo=s["ci_lo"], hi=s["ci_hi"], median=s["median"], p=s["p"]))
    cols = ["metric", "learner", "side", "contrast", "n_tables", "mean", "lo", "hi", "median", "p"]
    return pd.DataFrame(rows, columns=cols)


def reading(row: pd.Series) -> str:
    """Plan Sec. 6 wording: a trend only when the bootstrap CI excludes 0."""
    if row.slope_hi < 0:
        return "decline (CI < 0)"
    if row.slope_lo > 0:
        return "rise (CI > 0)"
    return "CI includes 0"


# ----------------------------------------------------------------------------- macros
DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
MAC: dict[str, str] = {}


def put(name: str, value: str) -> None:
    name = "ExB" + re.sub(r"[^A-Za-z]", "", "".join(DIG.get(c, c) for c in name))
    if name in MAC:
        raise KeyError(name)
    MAC[name] = value


def fx(x: float, nd: int = 3, sign: bool = True) -> str:
    if not np.isfinite(x):
        return "n/a"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def pf(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def build_macros(summ: pd.DataFrame, con: pd.DataFrame, cost: dict, fms: list[str], wall_h: float) -> str:
    put("Fits", f"{cost['fits']:,}".replace(",", "{,}"))
    put("CpuHours", f"{cost['cpu_hours']:.0f}")
    put("WallHours", f"{wall_h:.0f}" if np.isfinite(wall_h) else "n/a")
    put("NFMs", str(len(fms)))
    put("Reps", "5")
    for _, r in summ.iterrows():
        if r.metric not in MET_TAG:
            continue
        base = f"{r.arm}{r.side[:3].title()}{FM_TAG[r.learner]}{MET_TAG[r.metric]}"
        put(base + "Slope", fx(r.slope_mean))
        put(base + "Lo", fx(r.slope_lo))
        put(base + "Hi", fx(r.slope_hi))
        put(base + "Neg", f"{r.neg_share:.2f}")
        put(base + "P", pf(r.p))
        put(base + "Drop", fx(r.drop_mean))
        put(base + "N", str(int(r.n_tables)))
    for _, r in con.iterrows():
        if r.metric not in MET_TAG:
            continue
        a, b = r.contrast.split("-")
        b = b.replace("+", "plus")
        base = f"{a}vs{b}{r.side[:3].title()}{FM_TAG[r.learner]}{MET_TAG[r.metric]}"
        put(base + "Mean", fx(r["mean"]))
        put(base + "Lo", fx(r.lo))
        put(base + "Hi", fx(r.hi))
        put(base + "P", pf(r.p))
    lines = ["% Generated by paper/p1/exploratory/b_dose/analyze_dose_b.py; do not edit.",
             "% EXPLORATORY (DEV only, not pre-registered). Plan: paper/p1/exploratory/b_dose_response_plan.md",
             "% Naming: \\ExB<Arm><Side><FM><Metric><Stat>; Arm in O (original Part (c)), F, S, N;",
             "% Side Maj/Min; Metric F (macro-F1), PcF (after p/pihat), CmF (common eval set), CmPcF, Acc, PcAcc, BA, PcBA;",
             "% Stat Slope/Lo/Hi (mean table slope per unit share, 95% bootstrap CI), Neg, P (Wilcoxon), Drop, N.",
             "% Contrasts: \\ExB<A>vs<B><Side><FM><Metric><Mean|Lo|Hi|P>, A-B of table slopes (FvsO, SvsO, NvsF, OvsFplusS)."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in MAC.items()]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- report
def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true", help="smoke test only")
    ap.add_argument("--out", default=str(HERE / "out"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    new, fms, cost = load_new(HERE / "out" / "raw", args.allow_partial)
    if not fms:
        raise SystemExit("no FM finished")
    orig = load_original(fms)
    if args.allow_partial:
        orig = orig[orig.dataset.isin(new.dataset.unique())]
    cols = ["learner", "arm", "side", "dataset", "mult", "rep", "level", "feasible", "n_classes"]
    allrows = pd.concat([new[cols + METRICS], orig[cols + [m for m in METRICS if m in orig]]], ignore_index=True)

    slopes = pd.concat([table_slopes(allrows, m) for m in METRICS if allrows[m].notna().any()], ignore_index=True)
    summ = summarise(slopes)
    f1 = summ[summ.metric == "macro_f1"].copy()
    f1 = add_holm(f1[f1.arm != "O"])
    summ = summ.merge(f1[["metric", "learner", "arm", "side", "p_holm"]], how="left")
    con = contrasts(slopes)
    jts = pd.concat([jt(allrows, m) for m in ("macro_f1", "pc_macro_f1", "cm_macro_f1")], ignore_index=True)
    neg = rep_slope_neg(new, "macro_f1")
    summ = summ.merge(neg.assign(metric="macro_f1"), how="left")

    # sensitivities (macro-F1 only)
    sens = []
    for label, keep in (("without 694", allrows.dataset != 694), ("binary only", allrows.n_classes == 2),
                        ("multiclass only", allrows.n_classes > 2)):
        s = summarise(table_slopes(allrows[keep], "macro_f1")).assign(subset=label)
        sens.append(s)
    sens_df = pd.concat(sens, ignore_index=True)

    # feasibility
    feas = new.groupby(["arm", "side", "mult", "level"]).apply(
        lambda g: g[g.feasible].dataset.nunique(), include_groups=False).rename("tables").reset_index()

    slopes.to_csv(out / "slopes.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    con.to_csv(out / "contrasts.csv", index=False)
    jts.to_csv(out / "jt.csv", index=False)
    sens_df.to_csv(out / "sensitivity.csv", index=False)

    prov = json.loads((HERE / "out" / "provenance.json").read_text()) if (HERE / "out" / "provenance.json").exists() else {}
    wall_h = float("nan")
    log = HERE / "run.log"
    if log.exists():
        stamps = re.findall(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ", log.read_text(), flags=re.M)
        if stamps:
            ts = pd.to_datetime(stamps)
            wall_h = float((ts.max() - ts.min()).total_seconds() / 3600)
    (out / "b_numbers.tex").write_text(build_macros(summ, con, cost, fms, wall_h))
    write_summary(out, summ, con, jts, sens_df, feas, cost, fms, wall_h, prov, args.allow_partial)


def write_summary(out, summ, con, jts, sens, feas, cost, fms, wall_h, prov, partial) -> None:
    L = ["# Exploratory B: dose-response arms F, S and N (DEV)", ""]
    L += ["**EXPLORATORY: DEV only, not pre-registered.** Plan committed before any fit: "
          "`paper/p1/exploratory/b_dose_response_plan.md`. Generated by `b_dose/analyze_dose_b.py`; "
          "every number below comes from `b_dose/out/raw/` and `results/study7/c` (read-only).", ""]
    if partial:
        L += ["**SMOKE TEST ON PARTIAL DATA. NOT A RESULT.**", ""]
    not_run = [fm for fm in FMS if fm not in fms]
    L += ["## Run", "",
          f"- FMs analysed (all 20 tables finished): {', '.join(FM_NAMES[f] for f in fms)}."
          + (f" Not run / excluded by the plan's rule: {', '.join(FM_NAMES[f] for f in not_run)}." if not_run else ""),
          f"- Fits: {cost['fits']:,} distinct contexts ({cost['feasible_cells']:,} feasible cells of {cost['cells']:,}); "
          f"fit CPU time {cost['cpu_hours']:.1f} h; wall-clock {wall_h:.1f} h with 3 workers under `nice -n 10`.",
          f"- Per FM: " + "; ".join(f"{FM_NAMES[k]} {v:,} fits, {cost['hours_by_fm'][k]:.1f} CPU-h" for k, v in cost["fits_by_fm"].items()) + ".",
          f"- Seeds: namespace {prov.get('seed_namespace', 41)}, base {prov.get('base_seed', '')}; mult ∈ {{3, 5, 10}}; 5 reps.", "",
          "## Estimand and reading", "",
          "Slope = per table, the OLS slope of rep-mean metric on the dose (share s for O, F, N; matched original share s′ for S), "
          "averaged over feasible mults; units = metric per unit share (a slope of −0.25 ≈ −0.10 from s = 0.5 to 0.9). "
          "Mean over tables with a 95% percentile bootstrap CI (10,000 resamples of tables); Neg = share of tables with a negative slope; "
          "p = two-sided Wilcoxon; Holm over the 24 F/S/N macro-F1 cells (descriptive).",
          "- **O**: original Part (c), fixed B (results/study7/c, 10 reps). Share and minority count move together.",
          "- **F**: minority (non-upweighted) count fixed at k₀, u gains rows; share ↑, B ↑.",
          "- **S**: share fixed at 0.5, count follows the original path k(s′).",
          "- **N**: counts of F, nested contexts (sample identity fixed); FM seed fixed within rep in F, S and N.", ""]
    def verdict(metric: str, fm: str, side: str, arm: str) -> str:
        r = summ[(summ.metric == metric) & (summ.learner == fm) & (summ.side == side) & (summ.arm == arm)]
        return reading(r.iloc[0]) if len(r) else "n/a"

    vt = []
    for fm in fms:
        for side in SIDES:
            vt.append(dict(fm=FM_NAMES[fm], side=side,
                           **{"F (count fixed)": verdict("macro_f1", fm, side, "F"),
                              "S (share fixed)": verdict("macro_f1", fm, side, "S"),
                              "N (nested)": verdict("macro_f1", fm, side, "N"),
                              "F, common set": verdict("cm_macro_f1", fm, side, "F"),
                              "F after p/π̂": verdict("pc_macro_f1", fm, side, "F"),
                              "F after p/π̂, common set": verdict("cm_pc_macro_f1", fm, side, "F")}))
    L += ["## Verdicts by the plan's reading rules (macro-F1 slope; a trend only when the 95% CI excludes 0)", "",
          md(pd.DataFrame(vt)), ""]
    for metric, title in (("macro_f1", "macro-F1 (primary), own complement"),
                          ("pc_macro_f1", "macro-F1 after the p/π̂ prior correction"),
                          ("cm_macro_f1", "macro-F1 on the common evaluation set"),
                          ("cm_pc_macro_f1", "macro-F1, common set, after p/π̂"),
                          ("accuracy", "accuracy"), ("balanced_accuracy", "balanced accuracy")):
        s = summ[summ.metric == metric].copy()
        if s.empty:
            continue
        s["fm"] = s.learner.map(FM_NAMES)
        s["arm"] = pd.Categorical(s.arm, ARMS)
        s = s.sort_values(["learner", "side", "arm"])
        s["slope [95% CI]"] = s.apply(lambda r: f"{r.slope_mean:+.3f} [{r.slope_lo:+.3f}, {r.slope_hi:+.3f}]", axis=1)
        s["drop 0.5→max [CI]"] = s.apply(lambda r: f"{r.drop_mean:+.3f} [{r.drop_lo:+.3f}, {r.drop_hi:+.3f}]", axis=1)
        s["reading"] = s.apply(reading, axis=1)
        cols = ["fm", "side", "arm", "n_tables", "slope [95% CI]", "neg_share", "p", "drop 0.5→max [CI]", "reading"]
        if metric == "macro_f1":
            cols.insert(7, "p_holm")
            cols.insert(8, "rep_neg_share")
        L += [f"## Table-level slopes: {title}", "", md(s[cols]), ""]
    c = con[con.metric.isin(["macro_f1", "pc_macro_f1", "cm_macro_f1"])].copy()
    c["fm"] = c.learner.map(FM_NAMES)
    c["Δ slope [95% CI]"] = c.apply(lambda r: f"{r['mean']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", axis=1)
    L += ["## Paired contrasts of table slopes", "",
          "F−O > 0: holding the minority count fixed flattens the decline (part of O is the count). "
          "S−O: count-only path vs original. N−F: nesting vs independent draws. O−(F+S): additivity residual (descriptive).", "",
          md(c.sort_values(["metric", "learner", "side", "contrast"])[["metric", "fm", "side", "contrast", "n_tables", "Δ slope [95% CI]", "median", "p"]]), ""]
    j = jts.copy()
    j["fm"] = j.learner.map(FM_NAMES)
    L += ["## Secondary: stratified Jonckheere–Terpstra (decline; strata = table × mult; obs = reps)", "",
          "For N the reps are nested across levels, so the JT's independence assumption fails; use rep_neg_share above.", "",
          md(j.sort_values(["metric", "learner", "side", "arm"])[["metric", "fm", "side", "arm", "jt_z", "jt_p", "strata"]]), ""]
    s = sens.copy()
    s["fm"] = s.learner.map(FM_NAMES)
    s["slope [95% CI]"] = s.apply(lambda r: f"{r.slope_mean:+.3f} [{r.slope_lo:+.3f}, {r.slope_hi:+.3f}]", axis=1)
    L += ["## Sensitivities (macro-F1 slopes)", "",
          md(s.sort_values(["subset", "learner", "side", "arm"])[["subset", "fm", "side", "arm", "n_tables", "slope [95% CI]", "p"]]), ""]
    fw = feas.pivot_table(index=["arm", "side", "mult"], columns="level", values="tables").reset_index()
    L += ["## Feasibility (tables with a feasible cell, of 20)", "", md(fw), ""]
    (out / "SUMMARY.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
