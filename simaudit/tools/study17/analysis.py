"""Study 17 analysis (prereg Secs. 3-6). Runs once, after every FM job has finished or failed.

  OMP_NUM_THREADS=1 nice -n 15 .venv/bin/python tools/study17/analysis.py results/study17 \
      data/pool_study17.yaml results/study17/study17_results.json

Unit = table. Per (table, FM, arm, m): mean over reps (infeasible draws dropped and counted).
Contrasts per table (macro-F1 unless stated), M3 = mean over m in {3, 5, 10}:
- P   (primary, Holm over 4 FMs): S - RC at M3, Wilcoxon signed-rank two-sided;
- E   (equivalence, Holm over 4): TOST |S - RC| < 0.02 at M3 (two one-sided Wilcoxon);
- Q   (acquisition, Holm over 4): S - A at M3, Wilcoxon two-sided; label overhead and rho_A;
- L1  (one label per class, Holm over 4): TOST |S - RC| < 0.02 at m = 1;
- secondary: balanced accuracy and accuracy S - RC at M3 (Holm over 4 each), coverage part
  RC - R, decomposition per m, strata, subsets, leave-one-table-out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
MARGIN = 0.02
M3 = (3, 5, 10)
FMS = ("fm1", "fm2", "fm3", "fm5")
METRICS = ("macro_f1", "balanced_accuracy", "accuracy")
LOSS_LIMIT = 0.10  # > 10% of tables lost for an FM -> contrast "not evaluable"
BOOT = 10_000
SEED = 20261101


# ---------------------------------------------------------------- tests
def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (monotone)."""
    keys = sorted(p, key=lambda k: p[k])
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out


def wilcoxon_p(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) == 0 or np.all(x == 0):
        return 1.0
    return float(stats.wilcoxon(x, zero_method="wilcox").pvalue)


def tost_p(x: np.ndarray, margin: float = MARGIN) -> float:
    """p of H0: |location| >= margin (max of two one-sided signed-rank tests)."""
    x = x[~np.isnan(x)]
    lo = stats.wilcoxon(x + margin, alternative="greater").pvalue
    hi = stats.wilcoxon(x - margin, alternative="less").pvalue
    return float(max(lo, hi))


def boot_ci(x: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(BOOT, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summary(x: np.ndarray) -> dict[str, Any]:
    x = x[~np.isnan(x)]
    lo, hi = boot_ci(x)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)),
                win=float((x > 0).mean()), ci=[lo, hi])


# ---------------------------------------------------------------- data
def cell_means(results: Path) -> pd.DataFrame:
    """Per (dataset, learner, arm, mult): mean metrics, labels used, infeasible count."""
    cols = ["dataset", "learner", "arm", "mult", "rep", "infeasible", "labels_used", *METRICS]
    files = [p for p in sorted(results.glob("*.parquet")) if ".partial" not in p.name]
    d = pd.concat([pd.read_parquet(p, columns=cols) for p in files], ignore_index=True)
    g = d.groupby(["dataset", "learner", "arm", "mult"])
    out = g[[*METRICS, "labels_used"]].mean()
    out["n_infeasible"] = g["infeasible"].sum()
    return out.reset_index()


def per_table(cm: pd.DataFrame, metric: str, a: str, b: str, ms: tuple[int, ...]) -> pd.DataFrame:
    """Per (dataset, learner): mean over ms of (arm a - arm b)."""
    w = cm.pivot_table(index=["dataset", "learner", "mult"], columns="arm", values=metric)
    diff = (w[a] - w[b]).reset_index(name="d")
    diff = diff[diff.mult.isin(ms)]
    full = diff.groupby(["dataset", "learner"]).d.agg(["mean", "count"])
    full.loc[full["count"] < len(ms), "mean"] = np.nan  # a missing m -> table lost for that FM
    return full["mean"].unstack("learner")


# ---------------------------------------------------------------- label equivalence
def rho_a(cm: pd.DataFrame, m: int, n_classes: dict[str, int]) -> pd.DataFrame:
    """Per (table, FM): labels the A arm needs to reach S's macro-F1 at m, over mC.

    The A curve is (mean labels used, mean macro-F1) at m in {1, 3, 5, 10, 20}, made
    monotone (running max), interpolated linearly in log(labels). Censored ('>') if S(m)
    exceeds the curve's maximum, ('<') if below its minimum.
    """
    rows = []
    for (ds, fm), g in cm[cm.arm == "A"].groupby(["dataset", "learner"]):
        s = cm[(cm.dataset == ds) & (cm.learner == fm) & (cm.arm == "S") & (cm.mult == m)]
        if s.empty or ds not in n_classes:
            continue
        target, budget = float(s.macro_f1.iloc[0]), m * n_classes[ds]
        g = g.dropna(subset=["macro_f1", "labels_used"]).sort_values("labels_used")
        x, y = np.log(g.labels_used.to_numpy()), np.maximum.accumulate(g.macro_f1.to_numpy())
        if len(x) == 0:
            continue
        if target > y.max():
            rows.append(dict(dataset=ds, learner=fm, m=m, rho=np.exp(x.max()) / budget, censor=">"))
        elif target < y.min():
            rows.append(dict(dataset=ds, learner=fm, m=m, rho=np.exp(x.min()) / budget, censor="<"))
        else:
            j = int(np.argmax(y >= target))
            if j == 0:
                xs = x[0]
            else:
                f = (target - y[j - 1]) / max(y[j] - y[j - 1], 1e-12)
                xs = x[j - 1] + f * (x[j] - x[j - 1])
            rows.append(dict(dataset=ds, learner=fm, m=m, rho=float(np.exp(xs)) / budget, censor=""))
    return pd.DataFrame(rows)


def censored_median(r: pd.DataFrame) -> dict[str, Any]:
    """Median of rho with '>' values treated as +inf and '<' as 0 (valid if < 50% censored)."""
    v = r.rho.to_numpy(dtype=float).copy()
    v[r.censor.to_numpy() == ">"] = np.inf
    v[r.censor.to_numpy() == "<"] = 0.0
    return dict(n=int(len(v)), median=float(np.median(v)) if len(v) else np.nan,
                share_censored_high=float((r.censor == ">").mean()) if len(v) else np.nan)


# ---------------------------------------------------------------- decisions
def verdict_sup(p_holm: float, mean: float, lost: bool) -> str:
    if lost:
        return "not evaluable"
    if p_holm < ALPHA:
        return "replicated" if mean > 0 else "reversed"
    return "not established"


def family(tab: pd.DataFrame, n_tables: int, kind: str) -> dict[str, Any]:
    """One Holm family over the four FMs. kind: 'sup' (two-sided) or 'tost'."""
    ps, res = {}, {}
    for fm in FMS:
        x = tab[fm].to_numpy(dtype=float) if fm in tab else np.array([])
        lost = len(x[~np.isnan(x)]) < (1 - LOSS_LIMIT) * n_tables
        res[fm] = dict(summary(x) if len(x[~np.isnan(x)]) else dict(n=0), lost=lost)
        ps[fm] = 1.0 if lost else (wilcoxon_p(x) if kind == "sup" else tost_p(x))
        res[fm]["p"] = ps[fm]
    adj = holm(ps)
    for fm in FMS:
        res[fm]["p_holm"] = adj[fm]
        if kind == "sup":
            res[fm]["verdict"] = verdict_sup(adj[fm], res[fm].get("mean", 0.0), res[fm]["lost"])
        else:
            res[fm]["verdict"] = ("not evaluable" if res[fm]["lost"] else
                                  "equivalent" if adj[fm] < ALPHA else "not established")
    return res


def reading(p: dict[str, Any], e: dict[str, Any]) -> dict[str, str]:
    """Per-FM reading of P and E together (prereg Sec. 4.1)."""
    out = {}
    for fm in FMS:
        pv, ev = p[fm]["verdict"], e[fm]["verdict"]
        if pv == "replicated":
            out[fm] = "balance gap beyond coverage"
        elif pv == "reversed":
            out[fm] = "covered random beats stratified"
        elif ev == "equivalent":
            out[fm] = "coverage explains the gap (within +-0.02)"
        elif pv == "not evaluable":
            out[fm] = "not evaluable"
        else:
            out[fm] = "inconclusive"
    return out


def claim(p: dict[str, Any], e: dict[str, Any]) -> str:
    """Frame-level statement (prereg Sec. 4.2)."""
    rep = [fm for fm in FMS if p[fm]["verdict"] == "replicated"]
    rev = [fm for fm in FMS if p[fm]["verdict"] == "reversed"]
    eq = [fm for fm in FMS if e[fm]["verdict"] == "equivalent"]
    if rev:
        return "C-RC reversed for " + ",".join(rev)
    if len(rep) >= 3:
        return "C-RC holds (>= 3 of 4 FMs): not only coverage"
    if len(eq) >= 3 and not rep:
        return "C-COV holds (>= 3 of 4 FMs equivalent, none replicated): at >= 3C the gap is coverage"
    return f"mixed: replicated {rep}, equivalent {eq}"


def analyse(results: Path, pool: dict[str, Any]) -> dict[str, Any]:
    cm = cell_means(results)
    meta = {d["key"]: d for d in pool["datasets"]}
    n = len(meta)
    nC = {k: int(v["C"]) for k, v in meta.items()}
    out: dict[str, Any] = dict(n_tables=n)
    p = family(per_table(cm, "macro_f1", "S", "RC", M3), n, "sup")
    e = family(per_table(cm, "macro_f1", "S", "RC", M3), n, "tost")
    out.update(P=p, E=e, reading=reading(p, e), claim=claim(p, e))
    out["Q"] = family(per_table(cm, "macro_f1", "S", "A", M3), n, "sup")
    out["L1"] = family(per_table(cm, "macro_f1", "S", "RC", (1,)), n, "tost")
    out["L1_coverage_gap_S_minus_R"] = family(per_table(cm, "macro_f1", "S", "R", (1,)), n, "sup")
    out["secondary"] = {m: family(per_table(cm, m, "S", "RC", M3), n, "sup") for m in METRICS[1:]}
    out["coverage_part_RC_minus_R"] = family(per_table(cm, "macro_f1", "RC", "R", M3), n, "sup")
    out["unconditional_S_minus_R"] = family(per_table(cm, "macro_f1", "S", "R", M3), n, "sup")
    # label use of A and rho_A
    a = cm[cm.arm == "A"].copy()
    a["overhead"] = a.labels_used / (a.mult * a.dataset.map(nC))
    out["A_label_overhead"] = {f"{fm}_m{m}": float(g.overhead.mean())
                               for (fm, m), g in a.groupby(["learner", "mult"])}
    rho = pd.concat([rho_a(cm, m, nC) for m in (1, *M3)], ignore_index=True)
    out["rho_A"] = {f"{fm}_m{m}": censored_median(g) for (fm, m), g in rho.groupby(["learner", "m"])} if len(rho) else {}
    # per-m decomposition (descriptive)
    dec = {}
    for metric in METRICS:
        for m in (1, *M3):
            for fm in FMS:
                sub = cm[(cm.learner == fm) & (cm.mult == m)].pivot_table(index="dataset", columns="arm", values=metric)
                if not {"S", "R", "RC"} <= set(sub.columns):
                    continue
                dec[f"{metric}_{fm}_m{m}"] = dict(S_R=float((sub.S - sub.R).mean()),
                                                  S_RC=float((sub.S - sub.RC).mean()),
                                                  RC_R=float((sub.RC - sub.R).mean()))
    out["decomposition"] = dec
    # strata and subsets (descriptive; P contrast)
    tab = per_table(cm, "macro_f1", "S", "RC", M3)
    groups = {
        "multiclass": [k for k, v in meta.items() if int(v["C"]) > 2],
        "binary": [k for k, v in meta.items() if int(v["C"]) == 2],
        "imbalanced_binary": [k for k, v in meta.items() if int(v["C"]) == 2 and float(v["pi_min"]) < 0.25],
        "post_fm5": [k for k, v in meta.items() if v.get("fm5_contam") == "post"],
        "sens_flag": [k for k, v in meta.items() if not v.get("flag")],
    }
    out["subsets"] = {g: {fm: dict(summary(tab.loc[tab.index.isin(ks), fm].to_numpy(dtype=float)),
                                   p_unadj=wilcoxon_p(tab.loc[tab.index.isin(ks), fm].to_numpy(dtype=float)))
                          for fm in FMS if fm in tab} for g, ks in groups.items() if len(ks) >= 5}
    out["loo_P"] = {fm: [float(np.nanmin([wilcoxon_p(tab[fm].drop(i).to_numpy(dtype=float)) for i in tab.index])),
                         float(np.nanmax([wilcoxon_p(tab[fm].drop(i).to_numpy(dtype=float)) for i in tab.index]))]
                    for fm in FMS if fm in tab}
    out["infeasible_draws"] = int(cm.n_infeasible.sum())
    return out


def main() -> None:  # pragma: no cover
    import yaml

    results, pool_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    res = analyse(results, yaml.safe_load(pool_path.read_text()))
    out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(dict(claim=res["claim"], reading=res["reading"]), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
