"""Analysis of the dose arms F, S and N on the T1' subset (EXPLORATORY, preregistered).

Preregistration: paper/p1/exploratory/b_dose/PREREG_T1P_SUBSET.md (Sec. 5-6).
Reads out_t1p/raw/*.parquet (and, for the DEV reference under the same rule,
out/slopes.csv). Writes out_t1p/{slopes,summary,primary,contrasts,sensitivity}.csv
and out_t1p/SUMMARY.md. Macros go to exploratory/out/exploratory_numbers.tex via
make_exploratory_numbers.py (function bdose_t1p). No hand-typed numbers.

Usage (from simaudit/): .venv/bin/python paper/p1/exploratory/b_dose/analyze_dose_t1p.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from analyze_dose_b import contrasts, reading, rep_slope_neg, summarise, table_slopes  # noqa: E402
from common import FM_NAMES  # noqa: E402

FMS = ("fm2", "fm1", "fm5", "fm3")
SIDES = ("majority", "minority")
BASE = ("macro_f1", "accuracy", "balanced_accuracy")
METRICS = [p + m for p in ("", "pc_", "cm_", "cm_pc_") for m in BASE]
ROPE = 0.01
SPAN = 0.4  # residual = -SPAN * slope: fitted macro-F1 change from s = 0.5 to s = 0.9, sign flipped
N_BOOT = 10_000
OUT = HERE / "out_t1p"


def boot(x: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Percentile bootstrap of the mean over tables (as analysis.report.bootstrap_ci; 90% and 95%)."""
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(N_BOOT, x.size)).mean(axis=1)
    q = np.quantile(means, [0.025, 0.05, 0.95, 0.975])
    return dict(n=len(x), mean=float(x.mean()), lo95=q[0], lo90=q[1], hi90=q[2], hi95=q[3])


def verdict(r: dict[str, float]) -> str:
    """Prereg Sec. 6: equivalent / decline survives (failure) / over-correction / inconclusive."""
    if r["lo90"] >= -ROPE and r["hi90"] <= ROPE:
        return "equivalent"
    if r["lo95"] > ROPE:
        return "decline survives"
    if r["hi95"] < -ROPE:
        return "over-correction"
    return "inconclusive"


def primary(slopes: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for arm in ("F", "N"):
        s = slopes[(slopes.metric == "pc_macro_f1") & (slopes.arm == arm)]
        for (fm, side), g in s.groupby(["learner", "side"]):
            r = boot(-SPAN * g.slope.to_numpy())
            rows.append(dict(set=label, arm=arm, learner=fm, side=side, **r, verdict=verdict(r)))
        pooled = s.groupby(["learner", "dataset"]).slope.mean().reset_index()
        for fm, g in pooled.groupby("learner"):
            r = boot(-SPAN * g.slope.to_numpy())
            rows.append(dict(set=label, arm=arm, learner=fm, side="pooled", **r, verdict=verdict(r)))
    return pd.DataFrame(rows)


def load(raw: Path) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    keys = pd.read_csv(HERE / "t1p_subset.csv").key.tolist()
    frames, fms = [], []
    for fm in FMS:
        files = [raw / f"{fm}_{k.replace(':', '_')}.parquet" for k in keys]
        if all(p.exists() for p in files):  # prereg Sec. 4: an unfinished FM is excluded entirely
            fms.append(fm)
            frames += [pd.read_parquet(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    fit = df[df.feasible & df.fitted_here.fillna(False).astype(bool)]
    cost = dict(fits=int(len(fit)), cpu_hours=float(fit.seconds.sum() / 3600),
                hours_by_fm={k: float(v / 3600) for k, v in fit.groupby("learner").seconds.sum().items()},
                cells=int(len(df)), feasible_cells=int(df.feasible.sum()))
    return df, fms, keys, cost


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def main() -> None:
    df, fms, keys, cost = load(OUT / "raw")
    slopes = pd.concat([table_slopes(df, m) for m in METRICS if df[m].notna().any()], ignore_index=True)
    summ = summarise(slopes)
    neg = rep_slope_neg(df, "macro_f1")
    summ = summ.merge(neg.assign(metric="macro_f1"), how="left")
    con = contrasts(slopes)
    prim = primary(slopes, "T1'")
    dev_slopes = pd.read_csv(HERE / "out" / "slopes.csv")
    dev_slopes = dev_slopes[dev_slopes.learner.isin(fms)]
    prim_dev = primary(dev_slopes, "DEV")
    n_cls = df.groupby("dataset").n_classes.first()
    sens = []
    for label, ids in (("binary only", n_cls[n_cls == 2].index), ("multiclass only", n_cls[n_cls > 2].index)):
        sens.append(primary(slopes[slopes.dataset.isin(ids)], f"T1' {label}"))
    sens_df = pd.concat(sens, ignore_index=True)

    slopes.to_csv(OUT / "slopes.csv", index=False)
    summ.to_csv(OUT / "summary.csv", index=False)
    con.to_csv(OUT / "contrasts.csv", index=False)
    pd.concat([prim, prim_dev], ignore_index=True).to_csv(OUT / "primary.csv", index=False)
    sens_df.to_csv(OUT / "sensitivity.csv", index=False)
    log = HERE / "run_t1p.log"
    wall_h = float("nan")
    if log.exists():
        stamps = re.findall(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ", log.read_text(), flags=re.M)
        if stamps:
            ts = pd.to_datetime(stamps)
            wall_h = float((ts.max() - ts.min()).total_seconds() / 3600)
    cost["wall_hours"] = wall_h
    (OUT / "cost.json").write_text(json.dumps(cost, indent=2))
    write_summary(summ, con, prim, prim_dev, sens_df, cost, fms, keys, n_cls)


def write_summary(summ, con, prim, prim_dev, sens, cost, fms, keys, n_cls) -> None:
    L = ["# Dose arms F, S and N on a T1' subset (EXPLORATORY, preregistered)", "",
         "**EXPLORATORY; supports the appendix only.** Preregistration committed before any T1' fit: "
         "`b_dose/PREREG_T1P_SUBSET.md`. Generated by `b_dose/analyze_dose_t1p.py` from `b_dose/out_t1p/raw/`.", ""]
    not_run = [FM_NAMES[f] for f in FMS if f not in fms]
    L += ["## Run", "",
          f"- Tables ({len(keys)}; {int((n_cls == 2).sum())} binary, {int((n_cls > 2).sum())} multiclass): "
          + ", ".join(f"`{k}`" for k in keys) + ".",
          f"- FMs analysed (all tables finished): {', '.join(FM_NAMES[f] for f in fms)}."
          + (f" Excluded (unfinished): {', '.join(not_run)}." if not_run else ""),
          f"- Fits: {cost['fits']:,} distinct contexts ({cost['feasible_cells']:,} feasible cells of {cost['cells']:,}); "
          f"fit CPU time {cost['cpu_hours']:.1f} h; wall-clock {cost['wall_hours']:.1f} h, 6 workers, `nice -n 15`, CPU only.",
          "- Per FM: " + "; ".join(f"{FM_NAMES[k]} {v:.1f} CPU-h" for k, v in cost["hours_by_fm"].items()) + ".", ""]
    L += ["## Primary: residual after p/π̂ with the count fixed (arm F)", "",
          f"Residual = −{SPAN} × table slope of p/π̂-corrected macro-F1 on the share (fitted drop from s = 0.5 to 0.9; "
          "positive = a decline that survives the correction). Mean over tables, percentile bootstrap (10,000). "
          f"ROPE ±{ROPE}. Equivalent: 90% CI inside the ROPE (TOST, α = 0.05). Decline survives (failure): 95% CI above "
          f"+{ROPE}. Over-correction: 95% CI below −{ROPE}. Otherwise inconclusive. DEV rows apply the same rule to the "
          "DEV slopes (out/slopes.csv).", ""]
    p = pd.concat([prim, prim_dev], ignore_index=True)
    p["fm"] = p.learner.map(FM_NAMES)
    p["residual"] = p["mean"]
    p["95% CI"] = p.apply(lambda r: f"[{r.lo95:+.3f}, {r.hi95:+.3f}]", axis=1)
    p["90% CI"] = p.apply(lambda r: f"[{r.lo90:+.3f}, {r.hi90:+.3f}]", axis=1)
    for arm in ("F", "N"):
        q = p[p.arm == arm].sort_values(["learner", "side", "set"], ascending=[True, True, False])
        L += [f"### Arm {arm}" + (" (primary)" if arm == "F" else " (secondary)"), "",
              md(q[["fm", "side", "set", "n", "residual", "90% CI", "95% CI", "verdict"]]), ""]
    cells = prim[(prim.arm == "F") & (prim.side != "pooled")]
    counts = cells.verdict.value_counts().to_dict()
    overall = ("PASS (all cells equivalent)" if counts.get("equivalent", 0) == len(cells)
               else "FAIL (a decline survives)" if counts.get("decline survives", 0) > 0
               else "NOT ESTABLISHED (no failure; equivalence not shown in every cell)")
    L += [f"**Prereg decision (arm F, 8 FM × side cells): {overall}.** Counts: "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ".", ""]
    s_pc = summ[(summ.metric == "pc_macro_f1") & (summ.arm == "S")]
    n_s = int((s_pc.slope_hi < 0).sum())
    agree = counts.get("decline survives", 0) == 0 and n_s >= 6
    dev_counts = prim_dev[(prim_dev.arm == "F") & (prim_dev.side != "pooled")].verdict.value_counts().to_dict()
    L += [f"**Agreement with DEV (prereg Sec. 6): {'YES' if agree else 'NO'}.** (a) arm-F cells with a surviving "
          f"decline: {counts.get('decline survives', 0)}; (b) arm S after p/π̂ declines in {n_s} of {len(s_pc)} cells "
          "(needed ≥ 6). DEV under the same rule: "
          + ", ".join(f"{k} {v}" for k, v in sorted(dev_counts.items())) + ".", "",
          "Deviations from the preregistration: none. Estimated cost 51 CPU-h; actual "
          f"{cost['cpu_hours']:.1f} CPU-h.", ""]
    vt = []
    for fm in fms:
        for side in SIDES:
            def rd(metric, arm):
                r = summ[(summ.metric == metric) & (summ.learner == fm) & (summ.side == side) & (summ.arm == arm)]
                return reading(r.iloc[0]) if len(r) else "n/a"
            vt.append({"fm": FM_NAMES[fm], "side": side, "F": rd("macro_f1", "F"), "S": rd("macro_f1", "S"),
                       "N": rd("macro_f1", "N"), "F after p/π̂": rd("pc_macro_f1", "F"),
                       "S after p/π̂": rd("pc_macro_f1", "S"), "F common set": rd("cm_macro_f1", "F"),
                       "F common, p/π̂": rd("cm_pc_macro_f1", "F")})
    L += ["## Secondary: DEV reading rules on T1' (macro-F1 slope; trend only when the 95% CI excludes 0)", "",
          md(pd.DataFrame(vt)), ""]
    for metric, title in (("macro_f1", "macro-F1"), ("pc_macro_f1", "macro-F1 after p/π̂"),
                          ("cm_macro_f1", "macro-F1, common set"), ("cm_pc_macro_f1", "macro-F1, common set, after p/π̂"),
                          ("accuracy", "accuracy"), ("balanced_accuracy", "balanced accuracy")):
        s = summ[summ.metric == metric].copy()
        s["fm"] = s.learner.map(FM_NAMES)
        s["slope [95% CI]"] = s.apply(lambda r: f"{r.slope_mean:+.3f} [{r.slope_lo:+.3f}, {r.slope_hi:+.3f}]", axis=1)
        s["drop 0.5→max [CI]"] = s.apply(lambda r: f"{r.drop_mean:+.3f} [{r.drop_lo:+.3f}, {r.drop_hi:+.3f}]", axis=1)
        s["reading"] = s.apply(reading, axis=1)
        L += [f"## Table-level slopes: {title}", "",
              md(s.sort_values(["learner", "side", "arm"])[["fm", "side", "arm", "n_tables", "slope [95% CI]",
                                                            "neg_share", "p", "drop 0.5→max [CI]", "reading"]]), ""]
    c = con[con.metric.isin(["macro_f1", "pc_macro_f1"])].copy()
    c["fm"] = c.learner.map(FM_NAMES)
    c["Δ slope [95% CI]"] = c.apply(lambda r: f"{r['mean']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", axis=1)
    L += ["## N − F contrasts of table slopes", "",
          md(c.sort_values(["metric", "learner", "side"])[["metric", "fm", "side", "contrast", "n_tables", "Δ slope [95% CI]", "p"]]), ""]
    s = sens[sens.side != "pooled"].copy()
    s["fm"] = s.learner.map(FM_NAMES)
    s["95% CI"] = s.apply(lambda r: f"[{r.lo95:+.3f}, {r.hi95:+.3f}]", axis=1)
    L += ["## Sensitivity: residual by binary / multiclass tables (descriptive)", "",
          md(s.sort_values(["set", "arm", "learner", "side"])[["set", "arm", "fm", "side", "n", "mean", "95% CI", "verdict"]]), ""]
    (OUT / "SUMMARY.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
