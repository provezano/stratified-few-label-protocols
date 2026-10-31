"""E15 (EXPLORATORY, not pre-registered): how many labels, by metric and strategy (Study 9, T1′).

Stored results only (results/study9/t1prime, 52 tables × 4 FMs). No model is fitted:
the only computations on raw data are (i) reading y to score the stored per-query
probabilities (Brier score; recall per class) and (ii) the frozen label-free k-means
cluster sizes of Study 9 H3 (`analysis.study9.kmeans_pool_sizes`, K = C, the same call
as the m̂95 estimate), used as a label-free imbalance proxy. Nothing under results/,
reports/, prereg/ or sealed_heldout/ is written.

Conventions (reused from E11 and the frozen analysis):
- curve = mean over repetitions per (table, FM, strategy, m); B = m·C total labels;
- the per-table power law c − a·m^(−β) of E11 (scipy curve_fit, bounded) for the asymptote;
- the frozen `equivalence_interval` (monotone random envelope, log-m interpolation,
  censored outside the grid) and `turnbull_median_ratio` for the label-equivalence ρ.

Metrics: accuracy, balanced accuracy, macro-F1, macro one-vs-rest AUC (NaN when one class
is seen → imputed 0.5, the AUC of a constant score), log-loss (stored, clipped) and the
multiclass Brier score Σ_k (p_k − 1[y = k])² (range [0, 2]; for C = 2 it is twice the
binary Brier). Losses are oriented by sign for ε criteria (raw units) and turned into
skill scores against the label-free uniform predictor for shares:
LL-skill = 1 − LL / ln C, Brier-skill = 1 − Brier / (1 − 1/C).

Plateau per (table, FM, metric):
- PRIMARY "best": the best mean over all four strategies and all feasible budgets
  (model-free, no extrapolation; upward-biased by the winner's curse of a max over noisy
  means, more so for the 3-rep selectors);
- SECONDARY "asym": max(best, ĉ) with ĉ the power-law asymptote of the random curve
  (10 reps) fitted on the per-class grid, c ≤ 1.
- ROBUSTNESS "own": the best mean of the same strategy's own curve (where that strategy
  tops out on the grid; free of the cross-strategy winner's curse, closest to m*).
Required budget: the first feasible grid budget whose mean reaches plateau − ε
(ε ∈ {0.01, 0.02, 0.05}, raw units) or q·plateau (q ∈ {0.95, 0.99}, share scale);
right-censored (∞) if never reached on the grid (cap 30 per class or B ≤ n/2).

Run from simaudit/:  OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python paper/p1/exploratory/e15_budget_metrics.py
Writes out/e15_*.csv (+ e15_required_tables.csv.gz), out/e15_runs.parquet (per-run cache; delete to rebuild from results/), out/e15_share_*.{pdf,png},
out/e15_numbers.tex, out/e15_macros.csv, out/E15_SUMMARY.md.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from math import inf  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

import common as cm  # noqa: E402
from simaudit.analysis import study9 as a9  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.runner.study9 import dataset_key, layout, row_permutation  # noqa: E402

SEED = 20261027
N_BOOT = 2000
GRID = (1, 2, 3, 5, 7, 10, 15, 20, 30)
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "auc", "log_loss", "brier")
LOSSES = ("log_loss", "brier")
STRATS = ("random", "kmedoids_nested", "lucos_nested", "stratified")
SELS = ("kmedoids_nested", "lucos_nested")
FMS = cm.FMS
EPSS = (0.01, 0.02, 0.05)
QS = (0.95, 0.99)
CRITS = tuple(f"eps{e:g}" for e in EPSS) + tuple(f"q{q:g}" for q in QS)
RES = cm.settings.results_dir / "study9" / "t1prime"
OUT = cm.OUT
RUNS = OUT / "e15_runs.parquet"
COMMON_B = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300)


# ----------------------------------------------------------------- stage 1: per-run table
def members() -> pd.DataFrame:
    rows = yaml.safe_load((cm.PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())["datasets"]
    fr = pd.DataFrame(rows)
    fr["dataset"] = [dataset_key(int(d)) for d in fr.did]
    return fr.set_index("dataset")


def build_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-run scores (stored + Brier + class-level diagnostics) and per-table traits."""
    mem = members()
    runs, traits = [], []
    n_mismatch = 0
    for key, did in mem.did.items():
        data = load(int(did), cm.settings.data_cache_dir)
        y = np.asarray(data.y, np.int64)
        C = int(y.max()) + 1
        perm = row_permutation(len(y), int(did), cm.settings.base_seed)
        s, _ = layout(len(y))
        pool_orig = perm[s:]
        pool_counts = np.bincount(y[pool_orig], minlength=C)
        maj = int(pool_counts.argmax())
        pi_pool = pool_counts / pool_counts.sum()
        km = np.asarray(a9.kmeans_pool_sizes(np.asarray(data.X, float)[perm], C,
                                             np.arange(s, len(y), dtype=np.int64), cm.settings.base_seed))
        full = np.bincount(y, minlength=C)
        traits.append(dict(dataset=key, name=mem.loc[key, "name"], n=len(y), d=int(np.asarray(data.X).shape[1]), C=C,
                           IR=float(full.max() / full.min()), pi_maj=float(pi_pool[maj]),
                           tv_pool=float(0.5 * np.abs(pi_pool - 1 / C).sum()),
                           km_IR=float(km.max() / max(km.min(), 1)),
                           km_tv=float(0.5 * np.abs(km / km.sum() - 1 / C).sum())))
        for fm in FMS:
            df = pd.read_parquet(RES / f"{key}_{fm}.parquet",
                                 columns=["dataset", "learner", "strategy", "m", "budget", "rep", "labeled", "query",
                                          "proba", "coverage", "macro_f1", "accuracy", "balanced_accuracy",
                                          "log_loss", "auc"])
            for r in df.itertuples(index=False):
                q = np.asarray(r.query, np.int64)
                P = np.asarray(r.proba, float).reshape(len(q), C)
                yt = y[q]
                onehot = np.zeros_like(P)
                onehot[np.arange(len(q)), yt] = 1.0
                pred = P.argmax(1)
                if abs(float(np.mean(pred == yt)) - r.accuracy) > 1e-6:
                    n_mismatch += 1
                lab = y[np.asarray(r.labeled, np.int64)]
                pi_lab = np.bincount(lab, minlength=C) / len(lab)
                rec = [float(np.mean(pred[yt == k] == k)) for k in range(C) if (yt == k).any()]
                minor = [float(np.mean(pred[yt == k] == k)) for k in range(C) if k != maj and (yt == k).any()]
                runs.append(dict(dataset=key, learner=fm, strategy=r.strategy, m=int(r.m), budget=int(r.budget),
                                 rep=int(r.rep), C=C, coverage=r.coverage, macro_f1=r.macro_f1, accuracy=r.accuracy,
                                 balanced_accuracy=r.balanced_accuracy, log_loss=r.log_loss,
                                 auc_raw=r.auc, auc=0.5 if not np.isfinite(r.auc) else r.auc,
                                 brier=float(((P - onehot) ** 2).sum(1).mean()),
                                 lab_maj_share=float(pi_lab[maj]), lab_tv=float(0.5 * np.abs(pi_lab - 1 / C).sum()),
                                 recall_maj=float(np.mean(pred[yt == maj] == maj)) if (yt == maj).any() else np.nan,
                                 recall_minor=float(np.mean(minor)) if minor else np.nan,
                                 recall_mean=float(np.mean(rec)), pred_maj_rate=float(np.mean(pred == maj)),
                                 true_maj_rate=float(np.mean(yt == maj))))
        print(f"{key}: done", file=sys.stderr)
    if n_mismatch:
        raise SystemExit(f"{n_mismatch} runs: accuracy from stored proba != stored accuracy (y alignment)")
    return pd.DataFrame(runs), pd.DataFrame(traits).set_index("dataset")


# ----------------------------------------------------------------- stage 2: curves and plateaus
def skill(metric: str, v, C):
    """Higher-is-better score used for shares (skill vs uniform predictor for losses)."""
    if metric == "log_loss":
        return 1 - v / np.log(C)
    if metric == "brier":
        return 1 - v / (1 - 1 / C)
    return v


def oriented(metric: str, v):
    return -v if metric in LOSSES else v


def power(m, c, a, b):
    return c - a * np.power(m, -b)


def fit_c(ms, f):
    """E11 power law on the per-class grid; returns the asymptote c (≤ 1) or NaN."""
    if len(ms) < 4:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(power, ms, f, p0=(min(1.0, f.max() + 0.01), max(f.max() - f.min(), 0.01), 0.5),
                             bounds=([min(f.min(), 0) - 1, 1e-6, 1e-3], [1.0, 1e3, 5.0]), maxfev=20000)
        return float(p[0])
    except (RuntimeError, ValueError):
        return np.nan


def build_curves(runs: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (table, FM, metric, strategy, m) with mean value, score, share."""
    g = runs.groupby(["dataset", "learner", "strategy", "m"])
    cur = g[list(METRICS)].mean().reset_index()
    cur["C"] = g.C.first().to_numpy()
    long = cur.melt(id_vars=["dataset", "learner", "strategy", "m", "C"], value_vars=list(METRICS),
                    var_name="metric", value_name="value")
    long["B"] = long.m * long.C
    long["orient"] = [oriented(mt, v) for mt, v in zip(long.metric, long.value)]
    long["score"] = [skill(mt, v, c) for mt, v, c in zip(long.metric, long.value, long.C)]
    rows = []
    for (ds, fm, mt), grp in long.groupby(["dataset", "learner", "metric"]):
        ib = grp.orient.idxmax()
        best_o, best_s = grp.orient.max(), grp.score.max()
        rnd = grp[grp.strategy == "random"].sort_values("m")
        c = fit_c(rnd.m.to_numpy(float), rnd.score.to_numpy(float))
        asym_s = max(best_s, c) if np.isfinite(c) else best_s
        # back to oriented raw units (score is affine-increasing in the oriented value at fixed C)
        asym_o = best_o + (asym_s - best_s) * (1.0 if mt not in LOSSES else
                                               (np.log(grp.C.iloc[0]) if mt == "log_loss" else 1 - 1 / grp.C.iloc[0]))
        rows.append(dict(dataset=ds, learner=fm, metric=mt, best_o=best_o, best_s=best_s, asym_c=c,
                         asym_s=asym_s, asym_o=asym_o, best_strategy=grp.loc[ib, "strategy"], best_m=grp.loc[ib, "m"],
                         fit_ok=bool(np.isfinite(c))))
    pl = pd.DataFrame(rows)
    long = long.merge(pl, on=["dataset", "learner", "metric"])
    own = long.groupby(["dataset", "learner", "metric", "strategy"])
    long["own_o"], long["own_s"] = own.orient.transform("max"), own.score.transform("max")
    long["share_own"] = long.score / long.own_s
    long["gap_own"] = long.own_o - long.orient
    long["share_best"] = long.score / long.best_s
    long["share_asym"] = long.score / long.asym_s
    long["gap_best"] = long.best_o - long.orient
    long["gap_asym"] = long.asym_o - long.orient
    return long, pl


# ----------------------------------------------------------------- stage 3: required budget
def required(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, fm, mt, st), g in long.groupby(["dataset", "learner", "metric", "strategy"]):
        g = g.sort_values("m")
        for pt in ("best", "own", "asym"):
            gap, share = g[f"gap_{pt}"].to_numpy(), g[f"share_{pt}"].to_numpy()
            for crit in CRITS:
                ok = gap <= float(crit[3:]) + 1e-12 if crit.startswith("eps") else share >= float(crit[1:]) - 1e-12
                i = int(np.argmax(ok)) if ok.any() else None
                rows.append(dict(dataset=ds, learner=fm, metric=mt, strategy=st, plateau=pt, crit=crit,
                                 m_req=float(g.m.iloc[i]) if i is not None else inf,
                                 B_req=float(g.B.iloc[i]) if i is not None else inf,
                                 m_max=float(g.m.iloc[-1]), B_max=float(g.B.iloc[-1])))
    return pd.DataFrame(rows)


def q_inf(x, q):
    x = np.asarray(x, float)
    return float(np.quantile(x, q, method="inverted_cdf")) if len(x) else np.nan


def boot_median_ci(x, seed=SEED):
    x = np.asarray(x, float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(N_BOOT, len(x)))
    med = np.quantile(x[idx], 0.5, axis=1, method="inverted_cdf")
    return float(np.quantile(med, 0.025, method="inverted_cdf")), float(np.quantile(med, 0.975, method="inverted_cdf"))


def c_bin(c):
    return "C2" if c == 2 else ("C3-4" if c <= 4 else "C5+")


def ir_bin(v):
    return "IR<2" if v < 2 else ("IR2-5" if v <= 5 else "IR>5")


def summarize_required(req: pd.DataFrame, traits: pd.DataFrame) -> pd.DataFrame:
    req = req.join(traits[["C", "IR", "km_IR"]], on="dataset")
    req["stratum_C"] = req.C.map(c_bin)
    req["stratum_IR"] = req.IR.map(ir_bin)        # NOT label-free (uses y)
    req["stratum_kmIR"] = req.km_IR.map(ir_bin)   # label-free proxy (k-means, K = C)
    out = []
    for by in ("all", "stratum_C", "stratum_IR", "stratum_kmIR"):
        keys = ["learner", "metric", "strategy", "plateau", "crit"] + ([] if by == "all" else [by])
        for k, g in req.groupby(keys):
            d = dict(zip(keys, k))
            d["by"], d["level"] = by, ("all" if by == "all" else d.pop(by))
            d.update(n=len(g), B_med=q_inf(g.B_req, 0.5), B_q25=q_inf(g.B_req, 0.25), B_q75=q_inf(g.B_req, 0.75),
                     m_med=q_inf(g.m_req, 0.5), m_q25=q_inf(g.m_req, 0.25), m_q75=q_inf(g.m_req, 0.75),
                     n_cens=int(np.isinf(g.B_req).sum()), share_reached=float(np.isfinite(g.B_req).mean()))
            if by == "all" and d["plateau"] in ("best", "own"):
                d["B_med_lo"], d["B_med_hi"] = boot_median_ci(g.B_req)
                d["m_med_lo"], d["m_med_hi"] = boot_median_ci(g.m_req)
            out.append(d)
    return pd.DataFrame(out)


# ----------------------------------------------------------------- stage 4: share curves on total labels
def share_on_total(long: pd.DataFrame) -> pd.DataFrame:
    """Median/IQR share across tables on a common total-label grid (interpolated in log B, no extrapolation)."""
    rows = []
    lb = np.log(np.asarray(COMMON_B, float))
    for (fm, mt, st), g in long.groupby(["learner", "metric", "strategy"]):
        for pt in ("best", "own", "asym"):
            mat = []
            for _, gd in g.groupby("dataset"):
                gd = gd.sort_values("B")
                x, yv = np.log(gd.B.to_numpy(float)), gd[f"share_{pt}"].to_numpy(float)
                v = np.interp(lb, x, yv)
                v[(lb < x[0] - 1e-9) | (lb > x[-1] + 1e-9)] = np.nan
                mat.append(v)
            mat = np.array(mat)
            for j, b in enumerate(COMMON_B):
                col = mat[:, j][np.isfinite(mat[:, j])]
                rows.append(dict(learner=fm, metric=mt, strategy=st, plateau=pt, B=b, n=len(col),
                                 med=float(np.median(col)) if len(col) else np.nan,
                                 q25=float(np.quantile(col, 0.25)) if len(col) else np.nan,
                                 q75=float(np.quantile(col, 0.75)) if len(col) else np.nan))
    return pd.DataFrame(rows)


def share_on_m(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby(["learner", "metric", "strategy", "m"])
    out = []
    for pt in ("best", "own", "asym"):
        s = g[f"share_{pt}"]
        d = pd.DataFrame(dict(n=s.size(), med=s.median(), q25=s.quantile(0.25), q75=s.quantile(0.75),
                              B_med=g.B.median())).reset_index()
        d["plateau"] = pt
        out.append(d)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------- stage 5: selectors vs random
def equivalence(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fm, mt), g in long.groupby(["learner", "metric"]):
        piv = g.pivot_table(index=["dataset", "strategy"], columns="m", values="orient")
        for st in ("kmedoids_nested", "lucos_nested", "stratified"):
            for m in (1, 2, 3, 5, 7, 10):
                ivs = []
                for ds in g.dataset.unique():
                    ms, fr = a9._points(piv.loc[(ds, "random")])
                    if (ds, st) not in piv.index or m not in ms:
                        continue
                    sc = piv.loc[(ds, st)].get(m)
                    if sc is None or not np.isfinite(sc):
                        continue
                    ivs.append(a9.equivalence_interval(ms, fr, sc, m))
                if not ivs:
                    continue
                gt1 = [(lo > 1) or (hi == inf and lo >= 1) for lo, hi in ivs]
                med = a9.turnbull_median_ratio(ivs)
                lo, hi = a9.boot_ci(ivs, a9.turnbull_median_ratio, n_boot=N_BOOT, seed=SEED) if m in (1, 3, 5, 10) else (np.nan, np.nan)
                k = int(sum(gt1))
                rows.append(dict(learner=fm, metric=mt, strategy=st, m=m, n=len(ivs), rho_med=med, rho_lo=lo, rho_hi=hi,
                                 share_gt1=k / len(ivs), p_sign=a9.sign_test_greater(k, len(ivs)),
                                 n_left=int(sum(a == 0 for a, _ in ivs)), n_right=int(sum(b == inf for _, b in ivs))))
    return pd.DataFrame(rows)


def paired(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fm, mt), g in long.groupby(["learner", "metric"]):
        piv = g.pivot_table(index=["dataset", "m"], columns="strategy", values="orient")
        pivs = g.pivot_table(index=["dataset", "m"], columns="strategy", values="share_best")
        for st in ("kmedoids_nested", "lucos_nested", "stratified"):
            for m in GRID:
                sub = piv.xs(m, level="m")[["random", st]].dropna()
                if len(sub) < 6:
                    continue
                d = (sub[st] - sub["random"]).to_numpy()
                ds_ = pivs.xs(m, level="m")[["random", st]].dropna()
                s = cm.summarize(d)
                rows.append(dict(learner=fm, metric=mt, strategy=st, m=m, **s,
                                 dshare_med=float(np.median(ds_[st] - ds_["random"]))))
    out = pd.DataFrame(rows)
    out = cm.add_holm(out)  # Holm over the whole E15 paired family (4 FMs × 6 metrics × 3 strategies × budgets)
    return out


def mechanism(runs: pd.DataFrame, traits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Selector − random differences in label composition and class-level recall (descriptive)."""
    cols = ["coverage", "lab_maj_share", "lab_tv", "recall_maj", "recall_minor", "recall_mean", "pred_maj_rate"]
    cur = runs.groupby(["dataset", "learner", "strategy", "m"])[cols].mean()
    rows = []
    for fm in FMS:
        for st in ("kmedoids_nested", "lucos_nested", "stratified"):
            for m in (1, 2, 3, 5, 10):
                try:
                    a = cur.xs((fm, st, m), level=("learner", "strategy", "m"))
                    b = cur.xs((fm, "random", m), level=("learner", "strategy", "m"))
                except KeyError:
                    continue
                j = a.join(b, lsuffix="_s", rsuffix="_r", how="inner")
                for c in cols:
                    d = (j[c + "_s"] - j[c + "_r"]).dropna().to_numpy()
                    rows.append(dict(learner=fm, strategy=st, m=m, quantity=c, n=len(d),
                                     sel=float(j[c + "_s"].median()), rnd=float(j[c + "_r"].median()),
                                     d_mean=float(d.mean()), d_med=float(np.median(d)), win=float(np.mean(d > 0)),
                                     p=cm.wilcoxon_p(d)))
    mech = pd.DataFrame(rows)
    # Across tables: does the accuracy-minus-macro-F1 advantage of a selector track imbalance / majority over-sampling?
    assoc = []
    lg = runs.groupby(["dataset", "learner", "strategy", "m"])[["accuracy", "macro_f1", "lab_maj_share"]].mean()
    for fm in FMS:
        for st in SELS:
            for m in (1, 2, 3):
                try:
                    a = lg.xs((fm, st, m), level=("learner", "strategy", "m"))
                    b = lg.xs((fm, "random", m), level=("learner", "strategy", "m"))
                except KeyError:
                    continue
                j = a.join(b, lsuffix="_s", rsuffix="_r", how="inner").join(traits)
                gap = (j.accuracy_s - j.accuracy_r) - (j.macro_f1_s - j.macro_f1_r)
                for x, nm in ((np.log(j.IR), "log_IR"), (np.log(j.km_IR), "log_km_IR"),
                              (j.lab_maj_share_s - j.lab_maj_share_r, "d_lab_maj_share"), (np.log(j.C), "log_C")):
                    rho, lo, hi, p = cm.spearman_ci(x.to_numpy(), gap.to_numpy(), n_boot=2000, seed=SEED)
                    assoc.append(dict(learner=fm, strategy=st, m=m, x=nm, n=len(j), rho=rho, lo=lo, hi=hi, p=p))
    return mech, pd.DataFrame(assoc)


# ----------------------------------------------------------------- stage 6: ranks and marginal gains
def ranks(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fm, mt, m), g in long.groupby(["learner", "metric", "m"]):
        for pool, strats in (("all4", STRATS), ("blind3", STRATS[:3])):
            piv = g[g.strategy.isin(strats)].pivot_table(index="dataset", columns="strategy", values="orient").dropna()
            if len(piv) < 6 or piv.shape[1] < len(strats):
                continue
            rk = piv.rank(axis=1, ascending=False).mean()
            fr = stats.friedmanchisquare(*[piv[c] for c in piv.columns]).pvalue
            for st in strats:
                rows.append(dict(learner=fm, metric=mt, m=m, pool=pool, strategy=st, n=len(piv),
                                 mean_rank=float(rk[st]), top=bool(rk[st] == rk.min()), friedman_p=float(fr)))
    return pd.DataFrame(rows)


def marginal(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, fm, mt, st), g in long.groupby(["dataset", "learner", "metric", "strategy"]):
        g = g.sort_values("m")
        m, B, o, sh = (g[c].to_numpy(float) for c in ("m", "B", "orient", "share_best"))
        for k in range(len(m) - 1):
            rows.append(dict(dataset=ds, learner=fm, metric=mt, strategy=st, m0=m[k], m1=m[k + 1], B0=B[k], B1=B[k + 1],
                             slope_class=(o[k + 1] - o[k]) / (m[k + 1] - m[k]),
                             slope_label=(o[k + 1] - o[k]) / (B[k + 1] - B[k]),
                             pp_per_label=100 * (sh[k + 1] - sh[k]) / (B[k + 1] - B[k])))
    seg = pd.DataFrame(rows)
    agg = seg.groupby(["learner", "metric", "strategy", "m0", "m1"]).agg(
        n=("slope_label", "size"), slope_class_med=("slope_class", "median"),
        slope_label_med=("slope_label", "median"), slope_label_q25=("slope_label", lambda v: v.quantile(0.25)),
        slope_label_q75=("slope_label", lambda v: v.quantile(0.75)), pp_med=("pp_per_label", "median"),
        share_slope_ge_eps=("slope_class", lambda v: float((v >= 0.01).mean()))).reset_index()
    return agg


def flatten_points(marg: pd.DataFrame) -> pd.DataFrame:
    """First per-class budget after which the median per-class slope stays below 0.01 (and 0.005)."""
    rows = []
    for (fm, mt, st), g in marg.groupby(["learner", "metric", "strategy"]):
        g = g.sort_values("m0")
        s = g.slope_class_med.to_numpy()
        for eps in (0.005, 0.01, 0.02):
            k = next((i for i in range(len(s)) if np.all(s[i:] < eps)), None)
            rows.append(dict(learner=fm, metric=mt, strategy=st, eps=eps,
                             m_flat=float(g.m0.iloc[k]) if k is not None else inf))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- figures
COL = dict(random="#2a78d6", kmedoids_nested="#eb6834", lucos_nested="#1baf7a", stratified="#52514e")
LAB = dict(random="random (blind)", kmedoids_nested="k-medoids (label-free)", lucos_nested="LUCoS (label-free)",
           stratified="stratified (uses y)")
MLAB = dict(accuracy="Accuracy", balanced_accuracy="Balanced accuracy", macro_f1="Macro-F1", auc="AUC (macro OvR)",
            log_loss="Log-loss skill", brier="Brier skill")


def figure(df: pd.DataFrame, xcol: str, fname: str, fm: str = "fm2", nmin: int = 26) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True)
    for ax, mt in zip(axes.ravel(), METRICS):
        sub = df[(df.learner == fm) & (df.metric == mt) & (df.plateau == "best")]
        for st in STRATS:
            s = sub[(sub.strategy == st) & (sub.n >= nmin)].sort_values(xcol)
            ls = "--" if st == "stratified" else "-"
            ax.plot(s[xcol], s.med, ls, color=COL[st], lw=2, marker="o" if xcol == "m" else None, ms=4, label=LAB[st])
            if st == "random":
                ax.fill_between(s[xcol], s.q25, s.q75, color=COL[st], alpha=0.15, lw=0)
        ax.set_xscale("log")
        ticks = [2, 5, 10, 20, 50] if xcol == "B" else [1, 2, 3, 5, 10, 20, 30]
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(t) for t in ticks]))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(MLAB[mt], fontsize=10)
        ax.axhline(0.95, color="#b0afa8", lw=0.8, zorder=0)
        ax.axhline(1.0, color="#b0afa8", lw=0.8, zorder=0)
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        lo = sub[sub.n >= nmin].med.min()
        ax.set_ylim(max(min(lo, 0.95) - 0.03, -0.2), 1.01)
    xl = "total labels B (log scale)" if xcol == "B" else "labels per class m (log scale)"
    for ax in axes[1]:
        ax.set_xlabel(xl)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of best observed mean")
    h, lab_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab_, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"E15 (exploratory): median share of best, {cm.FM_NAMES[fm]}, 52 T1′ tables; band = random IQR",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    for ext in ("pdf", "png"):
        md = {"CreationDate": None} if ext == "pdf" else None  # no local timezone in the PDF
        fig.savefig(OUT / f"{fname}.{ext}", dpi=200, metadata=md)
    plt.close(fig)


# ----------------------------------------------------------------- macros
DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
MAC: dict[str, tuple[str, str]] = {}
MT = dict(accuracy="Acc", balanced_accuracy="BA", macro_f1="F", auc="AUC", log_loss="LL", brier="Brier")
ST = dict(random="Rand", kmedoids_nested="KMn", lucos_nested="LUn", stratified="Strat")
FT = dict(fm1="FMone", fm2="FMtwo", fm3="FMthree", fm5="FMfive")
CT = {"eps0.01": "EpsOne", "eps0.02": "EpsTwo", "eps0.05": "EpsFive", "q0.95": "QNinetyFive", "q0.99": "QNinetyNine"}


def put(name: str, value: str, src: str) -> None:
    name = "ExEFifteen" + re.sub(r"[^A-Za-z]", "", "".join(DIG.get(c, c) for c in name))
    if name in MAC:
        raise KeyError(name)
    MAC[name] = (value, src)


def num(x: float, nd: int = 2, sign: bool = False) -> str:
    if not np.isfinite(x):
        return "$\\infty$" if x > 0 else "$-\\infty$"
    if float(x).is_integer() and nd == 0:
        return f"{int(x)}"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def bud(x: float) -> str:
    return "$>$max" if not np.isfinite(x) else f"{x:g}"


def pf(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def macros(traits, rq, eq, pr, mech, assoc, rk, marg, flat, sht, shm, pl) -> None:
    """Quotable numbers only: TabICL (fm2, primary) in detail, other FMs as ranges or at m = 1."""
    put("NTables", str(len(traits)), "e15_traits.csv")
    put("NBinary", str(int((traits.C == 2).sum())), "e15_traits.csv")
    put("NMulti", str(int((traits.C > 2).sum())), "e15_traits.csv")
    rho, lo, hi, p = cm.spearman_ci(np.log(traits.IR.to_numpy()), np.log(traits.km_IR.to_numpy()), n_boot=2000, seed=SEED)
    put("KmIRvsIRRho", num(rho, 2, True), "e15_traits.csv")  # label-free proxy vs true pool IR
    put("KmIRvsIRLo", num(lo, 2, True), "e15_traits.csv")
    put("KmIRvsIRHi", num(hi, 2, True), "e15_traits.csv")
    put("KmIRvsIRP", pf(p), "e15_traits.csv")
    src = "e15_required.csv"
    PT = dict(best="", own="Own", asym="Asym")
    a = rq[(rq.by == "all") & (rq.learner == "fm2")]
    for _, r in a.iterrows():
        k = f"FMtwo{MT[r.metric]}{ST[r.strategy]}{CT[r.crit]}{PT[r.plateau]}"
        put(k + "Bmed", bud(r.B_med), src)
        put(k + "Cens", str(int(r.n_cens)), src)
        if r.plateau != "asym":
            put(k + "Bqlo", bud(r.B_q25), src)
            put(k + "Bqhi", bud(r.B_q75), src)
            put(k + "Mmed", bud(r.m_med), src)
            if r.plateau == "best" or r.crit in ("eps0.02", "q0.95"):
                put(k + "BmedLo", bud(r.B_med_lo), src)
                put(k + "BmedHi", bud(r.B_med_hi), src)
    for mt in METRICS:  # FM range of the medians (all four FMs)
        for st in STRATS:
            for crit in ("eps0.02", "q0.95"):
                for pt in ("best", "own"):
                    s = rq[(rq.by == "all") & (rq.metric == mt) & (rq.strategy == st) & (rq.plateau == pt) & (rq.crit == crit)]
                    k = f"FMs{MT[mt]}{ST[st]}{CT[crit]}{PT[pt]}"
                    put(k + "BmedMin", bud(s.B_med.min()), src)
                    put(k + "BmedMax", bud(s.B_med.max()), src)
    for by, tag in (("stratum_C", "ByC"), ("stratum_IR", "ByIR"), ("stratum_kmIR", "ByKmIR")):
        sub = rq[(rq.by == by) & (rq.learner == "fm2") & rq.plateau.isin(["best", "own"]) & rq.crit.isin(["eps0.02", "q0.95"])]
        for _, r in sub.iterrows():
            lv = {"C2": "Ctwo", "C3-4": "Cthreefour", "C5+": "Cfiveplus", "IR<2": "Low", "IR2-5": "Mid", "IR>5": "High"}[r.level]
            k = f"FMtwo{MT[r.metric]}{ST[r.strategy]}{CT[r.crit]}{PT[r.plateau]}{tag}{lv}"
            put(k + "Bmed", bud(r.B_med), src)
            put(k + "Cens", str(int(r.n_cens)), src)
            nk = f"FMtwo{tag}{lv}N"
            if "ExEFifteen" + nk not in MAC:
                put(nk, str(int(r.n)), src)
    src = "e15_equivalence.csv"
    for _, r in eq.iterrows():
        if r.learner != "fm2" and r.m != 1:
            continue
        k = f"{FT[r.learner]}{MT[r.metric]}{ST[r.strategy]}RhoM{int(r.m)}"
        put(k, num(r.rho_med), src)
        if r.learner == "fm2":
            if np.isfinite(r.rho_lo):
                put(k + "Lo", num(r.rho_lo), src)
                put(k + "Hi", num(r.rho_hi), src)
            put(k + "ShareGtOne", num(r.share_gt1), src)
            put(k + "PSign", pf(r.p_sign), src)
    for mt in METRICS:
        for st in ("kmedoids_nested", "lucos_nested", "stratified"):
            s = eq[(eq.metric == mt) & (eq.strategy == st) & (eq.m == 1)]
            put(f"FMs{MT[mt]}{ST[st]}RhoMOneMin", num(s.rho_med.min()), src)
            put(f"FMs{MT[mt]}{ST[st]}RhoMOneMax", num(s.rho_med.max()), src)
    src = "e15_paired.csv"
    put("PairedFamilySize", str(len(pr)), src)
    for _, r in pr.iterrows():
        if r.m not in (1, 2, 3, 5, 10, 30) or (r.learner != "fm2" and r.m != 1):
            continue
        k = f"{FT[r.learner]}{MT[r.metric]}{ST[r.strategy]}DiffM{int(r.m)}"
        put(k + "Mean", num(r["mean"], 3, True), src)
        put(k + "Holm", pf(r.p_holm), src)
        if r.learner == "fm2":
            put(k + "Lo", num(r.ci_lo, 3, True), src)
            put(k + "Hi", num(r.ci_hi, 3, True), src)
            put(k + "Med", num(r["median"], 3, True), src)
            put(k + "Win", num(r.win), src)
            put(k + "N", str(int(r.n)), src)
    for fm in FMS:
        for st in STRATS[1:]:
            s = pr[(pr.learner == fm) & (pr.strategy == st)]
            put(f"{FT[fm]}{ST[st]}NSigHolmPos", str(int(((s.p_holm < 0.05) & (s["mean"] > 0)).sum())), src)
            put(f"{FT[fm]}{ST[st]}NSigHolmNeg", str(int(((s.p_holm < 0.05) & (s["mean"] < 0)).sum())), src)
            put(f"{FT[fm]}{ST[st]}NTests", str(len(s)), src)
    src = "e15_mechanism.csv"
    QT = dict(coverage="Cov", lab_maj_share="LabMaj", lab_tv="LabTV", recall_maj="RecMaj", recall_minor="RecMin",
              recall_mean="RecMean", pred_maj_rate="PredMaj")
    for _, r in mech[(mech.learner == "fm2") & mech.m.isin([1, 3])].iterrows():
        k = f"Mech{ST[r.strategy]}{QT[r.quantity]}M{int(r.m)}"
        put(k + "Sel", num(r.sel, 3), src)
        put(k + "Rnd", num(r.rnd, 3), src)
        put(k + "D", num(r.d_mean, 3, True), src)
        put(k + "Win", num(r.win), src)
        put(k + "P", pf(r.p), src)
    src = "e15_mech_assoc.csv"
    XT = dict(log_IR="LogIR", log_km_IR="LogKmIR", d_lab_maj_share="DLabMaj", log_C="LogC")
    for _, r in assoc[assoc.learner == "fm2"].iterrows():
        k = f"Assoc{ST[r.strategy]}M{int(r.m)}{XT[r.x]}"
        put(k + "Rho", num(r.rho, 2, True), src)
        put(k + "Lo", num(r.lo, 2, True), src)
        put(k + "Hi", num(r.hi, 2, True), src)
        put(k + "P", pf(r.p), src)
    src = "e15_ranks.csv"
    for _, r in rk[(rk.learner == "fm2") & rk.m.isin([1, 3, 10, 30])].iterrows():
        k = f"Rank{MT[r.metric]}{r.pool.replace('all4', 'All').replace('blind3', 'Blind')}M{int(r.m)}"
        put(k + ST[r.strategy], num(r.mean_rank), src)
        if r.strategy == "random":
            put(k + "FriedmanP", pf(r.friedman_p), src)
    src = "e15_marginal.csv"
    for _, r in marg[(marg.learner == "fm2") & (marg.strategy.isin(["random", "kmedoids_nested", "lucos_nested"]))].iterrows():
        k = f"Marg{MT[r.metric]}{ST[r.strategy]}M{int(r.m0)}to{int(r.m1)}"
        put(k + "PerLabel", num(r.slope_label_med, 4, True), src)
        put(k + "PerClass", num(r.slope_class_med, 4, True), src)
        put(k + "PP", num(r.pp_med, 2, True), src)
        put(k + "ShareGeEps", num(r.share_slope_ge_eps), src)
    src = "e15_flatten.csv"
    for _, r in flat[flat.learner == "fm2"].iterrows():
        et = {0.005: "Half", 0.01: "One", 0.02: "Two"}[r.eps]
        put(f"FlatFMtwo{MT[r.metric]}{ST[r.strategy]}Eps{et}", bud(r.m_flat), src)
    src = "e15_share_total.csv"
    for _, r in sht[(sht.learner == "fm2") & sht.plateau.isin(["best"]) & sht.B.isin([10, 20, 50, 100])].iterrows():
        put(f"ShareFMtwo{MT[r.metric]}{ST[r.strategy]}B{int(r.B)}", num(r.med), src)
        put(f"ShareFMtwo{MT[r.metric]}{ST[r.strategy]}B{int(r.B)}N", str(int(r.n)), src)
    src = "e15_share_m.csv"
    for _, r in shm[(shm.learner == "fm2") & (shm.plateau == "best") & shm.m.isin([1, 3, 5, 10, 30])].iterrows():
        put(f"ShareFMtwo{MT[r.metric]}{ST[r.strategy]}M{int(r.m)}", num(r.med), src)
    src = "e15_plateau.csv"
    p2 = pl[pl.learner == "fm2"]
    for mt in METRICS:
        s = p2[p2.metric == mt]
        put(f"PlateauFMtwo{MT[mt]}ShareAtMax", num(float((s.best_m == 30).mean())), src)
        put(f"PlateauFMtwo{MT[mt]}ShareSelector", num(float(s.best_strategy.isin(SELS).mean())), src)
        put(f"PlateauFMtwo{MT[mt]}ShareRandom", num(float((s.best_strategy == "random").mean())), src)
        put(f"PlateauFMtwo{MT[mt]}AsymGapMed", num(float((s.asym_s - s.best_s).median()), 3), src)


def write_macros() -> None:
    lines = ["% Generated by paper/p1/exploratory/e15_budget_metrics.py. Do not edit. EXPLORATORY (E15, not pre-registered).",
             "% Study 9 T1' stored results, 52 tables x 4 FMs. Each line ends with its source file in out/."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}} % E15 exploratory; source: out/{s}" for k, (v, s) in sorted(MAC.items())]
    (OUT / "e15_numbers.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame([dict(name=k, value=v, src=s) for k, (v, s) in sorted(MAC.items())]).to_csv(OUT / "e15_macros.csv", index=False)


# ----------------------------------------------------------------- summary markdown
def summary(traits, pl, rq, eq, pr, mech, assoc, rk, flat, n_auc_nan) -> None:  # noqa: C901
    L = ["# E15 (EXPLORATORY): how many labels, by metric and strategy (Study 9 T1′, stored results)", "",
         f"52 tables × 4 FMs; {int((traits.C == 2).sum())} binary. AUC NaN runs imputed 0.5: {n_auc_nan}. "
         f"Power-law asymptote fit failed on {int((~pl.fit_ok).sum())}/{len(pl)} (table, FM, metric) random curves.", "",
         "## Required total budget B (median [IQR], censored n), TabICL (fm2), plateau = best observed mean", ""]
    sub = rq[(rq.by == "all") & (rq.learner == "fm2") & (rq.plateau == "best")]
    L += ["| metric | strategy | " + " | ".join(CRITS) + " |", "|---|---|" + "---|" * len(CRITS)]
    for mt in METRICS:
        for st in STRATS:
            cells = []
            for c in CRITS:
                r = sub[(sub.metric == mt) & (sub.strategy == st) & (sub.crit == c)].iloc[0]
                cells.append(f"{bud(r.B_med)} [{bud(r.B_q25)}, {bud(r.B_q75)}] ({int(r.n_cens)})")
            L.append(f"| {mt} | {st} | " + " | ".join(cells) + " |")
    L += ["", "Same, per class m:", "", "| metric | strategy | " + " | ".join(CRITS) + " |", "|---|---|" + "---|" * len(CRITS)]
    for mt in METRICS:
        for st in STRATS:
            cells = []
            for c in CRITS:
                r = sub[(sub.metric == mt) & (sub.strategy == st) & (sub.crit == c)].iloc[0]
                cells.append(f"{bud(r.m_med)} [{bud(r.m_q25)}, {bud(r.m_q75)}]")
            L.append(f"| {mt} | {st} | " + " | ".join(cells) + " |")
    L += ["", "Asymptote plateau (secondary), median B (censored n):", "",
          "| metric | strategy | " + " | ".join(CRITS) + " |", "|---|---|" + "---|" * len(CRITS)]
    sa = rq[(rq.by == "all") & (rq.learner == "fm2") & (rq.plateau == "asym")]
    for mt in METRICS:
        for st in STRATS:
            cells = [f"{bud(r.B_med)} ({int(r.n_cens)})" for c in CRITS
                     for r in [sa[(sa.metric == mt) & (sa.strategy == st) & (sa.crit == c)].iloc[0]]]
            L.append(f"| {mt} | {st} | " + " | ".join(cells) + " |")
    L += ["", "Own-curve plateau (robustness: where each strategy tops out on its own grid), median B [IQR] (censored n) / median m:", "",
          "| metric | strategy | " + " | ".join(CRITS) + " |", "|---|---|" + "---|" * len(CRITS)]
    so = rq[(rq.by == "all") & (rq.learner == "fm2") & (rq.plateau == "own")]
    for mt in METRICS:
        for st in STRATS:
            cells = [f"{bud(r.B_med)} [{bud(r.B_q25)}, {bud(r.B_q75)}] ({int(r.n_cens)}) / m {bud(r.m_med)}" for c in CRITS
                     for r in [so[(so.metric == mt) & (so.strategy == st) & (so.crit == c)].iloc[0]]]
            L.append(f"| {mt} | {st} | " + " | ".join(cells) + " |")
    p2 = pl[pl.learner == "fm2"]
    L += ["", "Where the best observed mean sits (fm2): share of tables with best at m = 30; share won by a 3-rep selector; "
          "median asymptote − best (skill/share scale):", "", "| metric | best at m=30 | best is a selector | best is random | asym − best |",
          "|---|---|---|---|---|"]
    for mt in METRICS:
        s = p2[p2.metric == mt]
        L.append(f"| {mt} | {(s.best_m == 30).mean():.2f} | {s.best_strategy.isin(SELS).mean():.2f} | "
                 f"{(s.best_strategy == 'random').mean():.2f} | {(s.asym_s - s.best_s).median():.3f} |")
    L += ["", "Median B across FMs (min–max of the four FM medians), ε = 0.02, best plateau:", "",
          "| metric | " + " | ".join(STRATS) + " |", "|---|" + "---|" * 4]
    s2 = rq[(rq.by == "all") & (rq.plateau == "best") & (rq.crit == "eps0.02")]
    for mt in METRICS:
        L.append(f"| {mt} | " + " | ".join(
            f"{bud(s2[(s2.metric == mt) & (s2.strategy == st)].B_med.min())}–{bud(s2[(s2.metric == mt) & (s2.strategy == st)].B_med.max())}"
            for st in STRATS) + " |")
    L += ["", "## Strata (fm2, best plateau, ε = 0.02): median B (median m), n", ""]
    for by, note in (("stratum_C", "C (known task, label-free given the class list)"),
                     ("stratum_IR", "pool IR from labels: NOT label-free"),
                     ("stratum_kmIR", "k-means (K = C) cluster IR: label-free proxy")):
        L += [f"### {note}", "", "| metric | level | random | k-medoids | LUCoS | stratified |", "|---|---|---|---|---|---|"]
        s = rq[(rq.by == by) & (rq.learner == "fm2") & (rq.plateau == "best") & (rq.crit == "eps0.02")]
        for mt in METRICS:
            for lv in sorted(s.level.unique()):
                cells = []
                for st in STRATS:
                    r = s[(s.metric == mt) & (s.level == lv) & (s.strategy == st)]
                    cells.append("–" if r.empty else f"{bud(r.iloc[0].B_med)} ({bud(r.iloc[0].m_med)}), n={int(r.iloc[0].n)}")
                L.append(f"| {mt} | {lv} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Label equivalence ρ (Turnbull median; CI at m ∈ {1,3,5,10}), fm2", "",
          "| metric | strategy | ρ(1) [CI] | share ρ(1)>1 | ρ(2) | ρ(3) | ρ(5) [CI] | ρ(10) [CI] |", "|---|---|---|---|---|---|---|---|"]
    e2 = eq[eq.learner == "fm2"]
    for mt in METRICS:
        for st in ("kmedoids_nested", "lucos_nested", "stratified"):
            g = e2[(e2.metric == mt) & (e2.strategy == st)].set_index("m")

            def rr(m, ci=False):
                if m not in g.index:
                    return "–"
                r = g.loc[m]
                return f"{r.rho_med:.2f}" + (f" [{r.rho_lo:.2f}, {r.rho_hi:.2f}]" if ci else "")
            L.append(f"| {mt} | {st} | {rr(1, True)} | {g.loc[1].share_gt1:.2f} | {rr(2)} | {rr(3)} | {rr(5, True)} | {rr(10, True)} |")
    L += ["", "ρ(1) medians, all FMs:", "", "| metric | strategy | fm1 | fm2 | fm3 | fm5 |", "|---|---|---|---|---|---|"]
    for mt in METRICS:
        for st in ("kmedoids_nested", "lucos_nested", "stratified"):
            L.append(f"| {mt} | {st} | " + " | ".join(
                f"{eq[(eq.learner == fm) & (eq.metric == mt) & (eq.strategy == st) & (eq.m == 1)].rho_med.iloc[0]:.2f}" for fm in FMS) + " |")
    L += ["", f"## Paired differences (strategy − random, oriented so + = better), fm2; Holm over all {len(pr)} E15 paired tests", "",
          "| metric | strategy | m=1 mean (Holm p) | m=2 | m=3 | m=5 | m=10 | m=30 |", "|---|---|---|---|---|---|---|---|"]
    p2 = pr[pr.learner == "fm2"]
    for mt in METRICS:
        for st in ("kmedoids_nested", "lucos_nested", "stratified"):
            cells = []
            for m in (1, 2, 3, 5, 10, 30):
                r = p2[(p2.metric == mt) & (p2.strategy == st) & (p2.m == m)]
                cells.append("–" if r.empty else f"{r.iloc[0]['mean']:+.3f} ({pf(r.iloc[0].p_holm)})")
            L.append(f"| {mt} | {st} | " + " | ".join(cells) + " |")
    L += ["", "## Mechanism (fm2): selector − random, median of table values (mean diff, win, Wilcoxon p unadjusted)", "",
          "| strategy | m | quantity | selector | random | mean diff | win | p |", "|---|---|---|---|---|---|---|---|"]
    for _, r in mech[(mech.learner == "fm2") & mech.m.isin([1, 3])].iterrows():
        L.append(f"| {r.strategy} | {r.m} | {r.quantity} | {r.sel:.3f} | {r.rnd:.3f} | {r.d_mean:+.3f} | {r.win:.2f} | {pf(r.p)} |")
    L += ["", "Spearman across tables of [Δaccuracy − Δmacro-F1] (selector − random) with:", "",
          "| strategy | m | x | ρ [CI] | p |", "|---|---|---|---|---|"]
    for _, r in assoc[assoc.learner == "fm2"].iterrows():
        L.append(f"| {r.strategy} | {r.m} | {r.x} | {r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] | {pf(r.p)} |")
    L += ["", "## Mean rank (1 = best) of the strategies, fm2", "",
          "| metric | pool | m | random | k-medoids | LUCoS | stratified | Friedman p |", "|---|---|---|---|---|---|---|---|"]
    r2 = rk[(rk.learner == "fm2") & rk.m.isin([1, 3, 10, 30])]
    for (mt, pool, m), g in r2.groupby(["metric", "pool", "m"]):
        g = g.set_index("strategy")
        L.append(f"| {mt} | {pool} | {m} | " + " | ".join(f"{g.loc[s].mean_rank:.2f}" if s in g.index else "–" for s in STRATS)
                 + f" | {pf(g.friedman_p.iloc[0])} |")
    L += ["", "## Flattening: first m after which the median per-class slope stays < ε (fm2)", "",
          "| metric | strategy | ε=0.005 | ε=0.01 | ε=0.02 |", "|---|---|---|---|---|"]
    f2 = flat[flat.learner == "fm2"]
    for (mt, st), g in f2.groupby(["metric", "strategy"]):
        g = g.set_index("eps")
        L.append(f"| {mt} | {st} | " + " | ".join(bud(g.loc[e].m_flat) for e in (0.005, 0.01, 0.02)) + " |")
    (OUT / "E15_SUMMARY.md").write_text("\n".join(L) + "\n")


def main() -> None:
    if RUNS.exists() and (OUT / "e15_traits.csv").exists():
        runs = pd.read_parquet(RUNS)
        traits = pd.read_csv(OUT / "e15_traits.csv", index_col="dataset")
    else:
        runs, traits = build_runs()
        runs.to_parquet(RUNS, index=False)
        traits.to_csv(OUT / "e15_traits.csv")
    n_auc_nan = int(runs.auc_raw.isna().sum())
    long, pl = build_curves(runs)
    pl.to_csv(OUT / "e15_plateau.csv", index=False)
    req = required(long)
    req.to_csv(OUT / "e15_required_tables.csv.gz", index=False)
    rq = summarize_required(req, traits)
    rq.to_csv(OUT / "e15_required.csv", index=False)
    sht, shm = share_on_total(long), share_on_m(long)
    sht.to_csv(OUT / "e15_share_total.csv", index=False)
    shm.to_csv(OUT / "e15_share_m.csv", index=False)
    print("required/share done", file=sys.stderr)
    eq = equivalence(long)
    eq.to_csv(OUT / "e15_equivalence.csv", index=False)
    print("equivalence done", file=sys.stderr)
    pr = paired(long)
    pr.to_csv(OUT / "e15_paired.csv", index=False)
    mech, assoc = mechanism(runs, traits)
    mech.to_csv(OUT / "e15_mechanism.csv", index=False)
    assoc.to_csv(OUT / "e15_mech_assoc.csv", index=False)
    rk = ranks(long)
    rk.to_csv(OUT / "e15_ranks.csv", index=False)
    marg = marginal(long)
    marg.to_csv(OUT / "e15_marginal.csv", index=False)
    flat = flatten_points(marg)
    flat.to_csv(OUT / "e15_flatten.csv", index=False)
    figure(sht, "B", "e15_share_total")
    figure(shm, "m", "e15_share_perclass")
    macros(traits, rq, eq, pr, mech, assoc, rk, marg, flat, sht, shm, pl)
    write_macros()
    summary(traits, pl, rq, eq, pr, mech, assoc, rk, flat, n_auc_nan)
    print(f"wrote {len(MAC)} macros", file=sys.stderr)


if __name__ == "__main__":
    main()
