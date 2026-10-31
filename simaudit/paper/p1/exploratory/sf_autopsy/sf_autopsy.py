"""Autopsy of the Study 15 non-replication (EXPLORATORY; post hoc; no verdict changes).

Study 15 (prereg/STUDY15_SECOND_FRAME.md) did not replicate C1 on the second frame
(0/4 replicated). This script asks, on the stored per-fit metrics only (no model fit):

Q1 moderator   per-table macro-F1 gap (S - R, mean over m in {1,3,5,10}; the paper's estimator,
               simaudit.analysis.study12.cell_means/delta) against pool imbalance
               (r = pi_min * C, 1 = balanced) and C = 2, with set indicators; OLS per FM with a
               table-level bootstrap stratified by set; composition-predicted second-frame mean
               from a model fitted on the OpenML frame (DEV + HELD-OUT) only; imbalance bins.
Q2 decomposition  coverage part = mean over R draws of (mean S - R draw) x 1[draw missed a class],
               balance residual = gap - coverage part (definitions of
               ../review_reanalysis/mech_recompute.py, metric columns only); exact binary
               coverage prediction of prop:coverage(v) at m = 1.
Q3 metric map  per-set accuracy / macro-F1 / balanced-accuracy gaps; ordering per table;
               prop:balance(ii) accuracy threshold B* = 1/(pi1 pi2 tau*^2) per binary table.
Q4 matched     binary DEV + HELD-OUT tables with pi_min below the second-frame median.
Q5 pipeline    class counts (npz vs pool yaml vs prereg table vs screen), S/R draws of one
               cell recomputed from the stored probabilities, near-constant predictors.

Inputs (read-only): results/study7/a (DEV), results/study12 (PMLB/UCI), results/study13/heldout
(arm T), results/study15/second_frame (arm T), the e10 pool-trait tables, the integrity screens,
data/pool_second_frame.yaml, data/cache/second_frame/npz. sealed_heldout/ is never touched.
HELD-OUT tables are written only in aggregate (no keys, no per-table rows).

Run from simaudit/:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  nice -n 15 .venv/bin/python paper/p1/exploratory/sf_autopsy/sf_autopsy.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score  # noqa: E402

from simaudit.analysis import study12 as a12  # noqa: E402
from simaudit.analysis.report import bootstrap_ci  # noqa: E402
from simaudit.data import second_frame as sf  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
RES = settings.results_dir
EXP = PROJECT_ROOT / "paper" / "p1" / "exploratory" / "out"
FMS = ("fm1", "fm2", "fm3", "fm5")
TEXFM = {"fm1": "One", "fm2": "Two", "fm3": "Three", "fm5": "Five"}
MULTS = a12.MULTS
METRICS = ("macro_f1", "accuracy", "balanced_accuracy")
COLS = ["dataset", "learner", "strategy", "mult", "rep", "coverage", *METRICS]
SETS = ("dev", "pmlb", "heldout", "sf")
SETNAME = {"dev": "DEV", "pmlb": "PMLB/UCI", "heldout": "HELD-OUT", "sf": "second frame"}
NBOOT = 5000
SEED = 20261031
BINS = (0.0, 0.35, 0.6, 0.85, 1.0001)


# ============================================================== loading
def _read(path: Path, arm: bool) -> pd.DataFrame:
    cols = COLS + (["arm"] if arm else [])
    t = pq.read_table(path, columns=cols).to_pandas()
    if arm:
        t = t[t.arm == "T"].drop(columns="arm")
    t = t[t.strategy.isin(["stratified", "random"])]
    t["dataset"] = t.dataset.astype(str)
    return t


def load_set(name: str) -> pd.DataFrame:
    folder = {
        "dev": RES / "study7" / "a",
        "pmlb": RES / "study12",
        "heldout": RES / "study13" / "heldout",
        "sf": RES / "study15" / "second_frame",
    }[name]
    arm = name in ("heldout", "sf")
    parts = [_read(p, arm) for fm in FMS for p in sorted(folder.glob(f"*_{fm}.parquet")) if ".partial" not in p.name]
    d = pd.concat(parts, ignore_index=True)
    d["set"] = name
    return d


def counts_from(s: str) -> list[int]:
    return [int(c) for c in str(s).split(";")]


def traits() -> pd.DataFrame:
    rows = []
    for name, f in (("dev", EXP / "e10_pools_dev.csv"), ("pmlb", EXP / "e10_pools_t1prime.csv")):
        t = pd.read_csv(f)
        for r in t.itertuples():
            rows.append(dict(set=name, dataset=str(r.dataset), n=int(r.n), C=int(r.C), pi_min=float(r.pi_min), counts=None))
    for name, f in (("heldout", RES / "study13" / "heldout" / "integrity_screen.csv"),
                    ("sf", RES / "study15" / "second_frame" / "integrity_screen.csv")):
        t = pd.read_csv(f, dtype=str, keep_default_na=False)
        for r in t.itertuples():
            c = counts_from(r.class_counts)
            rows.append(dict(set=name, dataset=str(r.key), n=int(r.n), C=int(r.C), pi_min=min(c) / sum(c), counts=c))
    t = pd.DataFrame(rows)
    t["r"] = t.pi_min * t.C  # normalised minority share, 1 = balanced
    t["binary"] = (t.C == 2).astype(int)
    return t


# ============================================================== estimators
def table_gaps(runs: pd.DataFrame, metric: str, mults=MULTS) -> pd.DataFrame:
    """Per (set, table) paper gap per FM (a12.cell_means + a12.delta)."""
    out = []
    for s, g in runs.groupby("set"):
        means = a12.cell_means(g, metric)
        cols = {fm: a12.delta(means, (fm, "stratified"), (fm, "random"), tuple(mults), metric) for fm in FMS}
        df = pd.DataFrame(cols)
        df["set"] = s
        out.append(df.rename_axis("dataset").reset_index())
    return pd.concat(out, ignore_index=True)


def ci(x) -> tuple[float, float, float, int]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    lo, hi = bootstrap_ci(x)
    return float(x.mean()), lo, hi, len(x)


def strat_boot_idx(sets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = []
    for s in np.unique(sets):
        pos = np.flatnonzero(sets == s)
        idx.append(rng.choice(pos, size=len(pos)))
    return np.concatenate(idx)


def design(df: pd.DataFrame, extra: tuple[str, ...] = ()) -> tuple[np.ndarray, list[str]]:
    names = ["const", "r", "binary", "ind_pmlb", "ind_sf", *extra]
    X = np.column_stack([
        np.ones(len(df)), df.r, df.binary,
        (df.set == "pmlb").astype(float), (df.set == "sf").astype(float),
        *[df[e] for e in extra],
    ])
    keep = [i for i, n in enumerate(names) if np.ptp(X[:, i]) > 0 or n == "const"]
    return X[:, keep], [names[i] for i in keep]


def ols_boot(df: pd.DataFrame, y: str, extra: tuple[str, ...] = (), seed: int = SEED) -> pd.DataFrame:
    X, names = design(df, extra)
    yv = df[y].to_numpy(float)
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    rng = np.random.default_rng(seed)
    sets = df.set.to_numpy()
    bs = []
    for _ in range(NBOOT):
        i = strat_boot_idx(sets, rng)
        bs.append(np.linalg.lstsq(X[i], yv[i], rcond=None)[0])
    bs = np.array(bs)
    return pd.DataFrame(dict(term=names, coef=beta, lo=np.quantile(bs, 0.025, 0), hi=np.quantile(bs, 0.975, 0)))


def composition_prediction(df: pd.DataFrame, y: str, seed: int = SEED, extra: tuple[str, ...] = ()) -> dict:
    """Fit gap ~ r + binary on OpenML (DEV+HELD-OUT); predict SF tables; obs - pred with bootstrap."""
    ref = df[df.set.isin(["dev", "heldout"])]
    tgt = df[df.set == "sf"]

    def fit_pred(a, b):
        X = np.column_stack([np.ones(len(a)), a.r, a.binary, *[a[e] for e in extra]])
        beta = np.linalg.lstsq(X, a[y].to_numpy(float), rcond=None)[0]
        Xt = np.column_stack([np.ones(len(b)), b.r, b.binary, *[b[e] for e in extra]])
        return float((Xt @ beta).mean()), float(b[y].mean())

    pred, obs = fit_pred(ref, tgt)
    rng = np.random.default_rng(seed)
    diffs, preds = [], []
    for _ in range(NBOOT):
        a = ref.iloc[strat_boot_idx(ref.set.to_numpy(), rng)]
        b = tgt.iloc[rng.choice(len(tgt), size=len(tgt))]
        p, o = fit_pred(a, b)
        diffs.append(o - p)
        preds.append(p)
    return dict(pred=pred, pred_lo=float(np.quantile(preds, 0.025)), pred_hi=float(np.quantile(preds, 0.975)),
                obs=obs, diff=obs - pred, diff_lo=float(np.quantile(diffs, 0.025)),
                diff_hi=float(np.quantile(diffs, 0.975)))


def two_sample_diff(a, b, seed: int = SEED) -> tuple[float, float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(NBOOT)]
    return float(a.mean() - b.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


# ============================================================== Q2
def p_miss(counts, B):
    from scipy.special import gammaln

    n = int(sum(counts))

    def lc(a, k):
        return -np.inf if k > a else float(gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1))

    return np.array([np.exp(lc(n - c, B) - lc(n, B)) for c in counts])


def decomposition(runs: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tri = tr.set_index(["set", "dataset"])
    for (s, ds, fm, m), g in runs.groupby(["set", "dataset", "learner", "mult"]):
        st, rd = g[g.strategy == "stratified"], g[g.strategy == "random"]
        if st.empty or rd.empty:
            continue
        miss = (rd.coverage < 1).to_numpy()
        row = dict(set=s, dataset=ds, fm=fm, mult=int(m), p_miss_obs=float(miss.mean()))
        for met in METRICS:
            mS = float(st[met].mean())
            gap = mS - float(rd[met].mean())
            cov = float(((mS - rd[met].to_numpy()) * miss).mean())
            row.update({f"{met}_gap": gap, f"{met}_cov": cov, f"{met}_bal": gap - cov})
        t = tri.loc[(s, ds)]
        row.update(C=int(t.C), r=float(t.r))
        if int(t.C) == 2 and t.counts is not None:
            c = np.asarray(t.counts)
            n, B = int(c.sum()), 2 * int(m)
            q = p_miss(c, B)
            other = (c[::-1] - B) / (n - B)
            row["macro_f1_cov_pred"] = float((q * (float(st.macro_f1.mean()) - other / (1 + other))).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================== signal / constant predictions
def signal(runs: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """Table signal from stored metrics: chance-normalised balanced accuracy at m = 10,
    (BA - 1/C) / (1 - 1/C), mean over S and R draws and over the four FMs."""
    g = runs[runs.mult == 10].groupby(["set", "dataset", "strategy"]).balanced_accuracy.mean().unstack().mean(axis=1)
    out = g.rename("ba10").reset_index().merge(tr[["set", "dataset", "C"]], on=["set", "dataset"])
    out["signal"] = (out.ba10 - 1 / out.C) / (1 - 1 / out.C)
    return out[["set", "dataset", "ba10", "signal"]]


def const_share(folder: Path, fms=FMS) -> pd.DataFrame:
    """Share of arm-T S/R draws whose argmax gives one class to >= 99% of query rows (no labels read)."""
    rows = []
    for fm in fms:
        for p in sorted(folder.glob(f"*_{fm}.parquet")):
            t = pq.read_table(p, columns=["arm", "strategy", "mult", "n_classes", "proba"]).to_pandas()
            t = t[(t.arm == "T") & t.strategy.isin(["stratified", "random"])]
            for r in t.itertuples():
                P = np.asarray(r.proba).reshape(-1, int(r.n_classes))
                top = np.bincount(P.argmax(1), minlength=int(r.n_classes)).max() / len(P)
                rows.append(dict(fm=fm, strategy=r.strategy, mult=int(r.mult), const=top >= 0.99))
    return pd.DataFrame(rows)


def reencode(pool: dict, keys: list[str]) -> list[dict]:
    """Rebuild X, y of spot tables from the raw files with the module's own encoder; compare with npz."""
    root = settings.data_cache_dir / "second_frame"
    out = []
    for d in pool["datasets"]:
        if d["key"] not in keys:
            continue
        spec = sf.spec_by_did(int(d["did"]))
        header, rows = sf.read_table(spec, root / "raw" / spec.key)
        X, y, info = sf.encode(spec, header, rows)
        with np.load(root / "npz" / f"{d['did']}.npz", allow_pickle=False) as z:
            Xn, yn = np.asarray(z["X"], float), np.asarray(z["y"], np.int64)
        t = list(header).index(spec.target)
        raw = [r[t] for r in rows if not sf.is_missing(r[t])]
        lv, cn = np.unique(np.asarray(raw, dtype=str), return_counts=True)
        out.append(dict(key=d["key"], target=spec.target, raw_levels=dict(zip(lv.tolist(), cn.tolist())),
                        y_equal=bool(np.array_equal(y, yn)),
                        X_equal=bool(X.shape == Xn.shape and np.allclose(X, Xn, equal_nan=True)),
                        counts=np.bincount(y).tolist()))
    return out


# ============================================================== Q5
def spot_checks(pool: dict) -> dict:
    out: dict = {}
    loader = sf.SecondFrameLoader(settings.data_cache_dir / "second_frame" / "npz",
                                  {int(d["did"]): str(d["sha256"]) for d in pool["datasets"]})
    prereg = (PROJECT_ROOT / "prereg" / "STUDY15_SECOND_FRAME.md").read_text()
    screen = pd.read_csv(RES / "study15" / "second_frame" / "integrity_screen.csv", dtype=str, keep_default_na=False)
    screen = screen.set_index("key")
    count_rows, ys = [], {}
    for d in pool["datasets"]:
        data = loader(int(d["did"]), Path())
        y = data.y
        ys[d["key"]] = y
        npz = np.bincount(y).tolist()
        m = re.search(rf"^\| {re.escape(d['key'])} \|.*\| (\d+(?:/\d+)+) \|", prereg, re.MULTILINE)
        pre = [int(v) for v in m.group(1).split("/")] if m else None
        count_rows.append(dict(key=d["key"], C=int(d["C"]), npz=npz, pool=list(d["class_counts"]),
                               prereg=pre, screen=counts_from(screen.loc[d["key"], "class_counts"]),
                               n_npz=len(y), n_pool=int(d["n"]),
                               ok=npz == list(d["class_counts"]) == pre == counts_from(screen.loc[d["key"], "class_counts"])
                               and len(y) == int(d["n"])))
    out["counts"] = count_rows
    out["counts_all_ok"] = all(r["ok"] for r in count_rows)

    # near-constant predictors, and S/R cell recompute
    folder = RES / "study15" / "second_frame"
    degen, recompute = [], []
    spot = ["f33111881_frailty", "z22705412_hpv", "z22216010_pvt", "z22767593_edmort", "f33328002_tea"]
    for key, y in ys.items():
        C = int(y.max()) + 1
        for fm in FMS:
            t = pd.read_parquet(folder / f"{key}_{fm}.parquet",
                                columns=["arm", "strategy", "mult", "rep", "labeled", "query", "proba", "coverage", *METRICS])
            t = t[(t.arm == "T") & t.strategy.isin(["stratified", "random"])]
            for r in t.itertuples():
                P = np.asarray(r.proba, dtype=np.float64).reshape(-1, C)
                pred = P.argmax(1)
                share = np.bincount(pred, minlength=C).max() / len(pred)
                degen.append(dict(key=key, fm=fm, strategy=r.strategy, mult=int(r.mult), top_share=float(share),
                                  bacc=float(r.balanced_accuracy)))
            if key in spot:
                for strat in ("stratified", "random"):
                    r = t[(t.strategy == strat) & (t.mult == 3) & (t.rep == 0)].iloc[0]
                    L, Q = np.asarray(r.labeled), np.asarray(r["query"])
                    P = np.asarray(r.proba, dtype=np.float64).reshape(-1, C)
                    pred = P.argmax(1)
                    comp = np.bincount(y[L], minlength=C).tolist()
                    recompute.append(dict(
                        key=key, fm=fm, strategy=strat, B=len(L), comp=comp,
                        disjoint=len(np.intersect1d(L, Q)) == 0, complement=len(L) + len(Q) == len(y),
                        rows_P=P.shape[0], cov=float(r.coverage),
                        d_f1=abs(f1_score(y[Q], pred, labels=list(range(C)), average="macro", zero_division=0) - r.macro_f1),
                        d_acc=abs(accuracy_score(y[Q], pred) - r.accuracy),
                        d_bacc=abs(balanced_accuracy_score(y[Q], pred) - r.balanced_accuracy)
                        if len(np.unique(y[Q])) == C else np.nan,
                        pred_share=np.round(np.bincount(pred, minlength=C) / len(pred), 3).tolist(),
                    ))
    out["recompute"] = recompute
    dg = pd.DataFrame(degen)
    out["degenerate_draw_share"] = float((dg.top_share >= 0.99).mean())
    tab = dg.groupby(["key", "fm"]).agg(const_share=("top_share", lambda s: float((s >= 0.99).mean())),
                                        bacc=("bacc", "mean")).reset_index()
    out["near_chance_tables"] = sorted(set(tab[(tab.bacc < 0.55)].key))
    out["per_table_fm"] = tab
    out["by_strategy"] = dg.assign(const=dg.top_share >= 0.99).groupby(["strategy", "mult"]).const.mean().round(3).to_dict()
    return out


# ============================================================== main
def main() -> None:
    tr = traits()
    runs = pd.concat([load_set(s) for s in SETS], ignore_index=True)
    lines: list[str] = ["% AUTO-GENERATED by paper/p1/exploratory/sf_autopsy/sf_autopsy.py. "
                        "EXPLORATORY; post hoc. Do not edit. HELD-OUT in aggregate only."]
    notes: dict = {}

    def put(name: str, value: str, what: str) -> None:
        assert re.fullmatch(r"[A-Za-z]+", name), name
        lines.append(f"\\newcommand{{\\ExSFA{name}}}{{{value}}} % EXPLORATORY; post hoc; {what}")

    def f(x, d=3, sign=True):
        return "n/a" if x is None or not np.isfinite(x) else (f"${x:+.{d}f}$" if sign else f"${x:.{d}f}$")

    # ---- per-table gaps for the three metrics
    gaps = {met: table_gaps(runs, met) for met in METRICS}
    long = []
    for met, g in gaps.items():
        x = g.melt(id_vars=["set", "dataset"], value_vars=list(FMS), var_name="fm", value_name="gap")
        x["metric"] = met
        long.append(x)
    long = pd.concat(long).merge(tr[["set", "dataset", "n", "C", "pi_min", "r", "binary"]], on=["set", "dataset"])
    long["logn"] = np.log10(long.n)
    long["ind_ho"] = (long.set == "heldout").astype(float)

    # sanity: reproduce the reported per-set means
    check = long[long.metric == "macro_f1"].groupby(["set", "fm"]).gap.mean().unstack().round(3)
    print("reproduced macro-F1 gap means per set:\n", check)
    notes["reproduced_means"] = check.to_dict()

    # ---- Q1 moderator model
    mod_rows, pred_rows = [], []
    for met in METRICS:
        for fm in FMS:
            d = long[(long.metric == met) & (long.fm == fm)].dropna(subset=["gap"]).reset_index(drop=True)
            for spec, dd, extra in (("main", d, ()),
                                    ("plus_logn", d, ("logn",)),
                                    ("plus_ho", d, ("ind_ho",)),
                                    ("binary_only", d[d.binary == 1].reset_index(drop=True), ()),
                                    ("openml_sf_only", d[d.set != "pmlb"].reset_index(drop=True), ())):
                t = ols_boot(dd.assign(gap=dd.gap), "gap", extra)
                t["metric"], t["fm"], t["spec"], t["n"] = met, fm, spec, len(dd)
                mod_rows.append(t)
            cp = composition_prediction(d, "gap")
            pred_rows.append(dict(metric=met, fm=fm, **cp))
    mod = pd.concat(mod_rows, ignore_index=True)
    mod.to_csv(OUT / "q1_model.csv", index=False)
    pred = pd.DataFrame(pred_rows)
    pred.to_csv(OUT / "q1_composition_prediction.csv", index=False)
    print(mod[(mod.metric == "macro_f1") & (mod.spec == "main")].round(3).to_string())
    print(pred.round(3).to_string())

    for fm in FMS:
        w = TEXFM[fm]
        for spec, tag in (("main", ""), ("plus_logn", "Logn"), ("plus_ho", "HoInd"), ("binary_only", "BinOnly"), ("openml_sf_only", "NoPmlb")):
            sub = mod[(mod.metric == "macro_f1") & (mod.fm == fm) & (mod.spec == spec)].set_index("term")
            terms = [("ind_sf", "SF")] + ([("r", "Imb"), ("binary", "Bin"), ("ind_pmlb", "Pmlb")] if spec == "main" else [])
            terms += [("ind_ho", "HO")] if spec == "plus_ho" else []
            for term, tn in terms:
                if term in sub.index:
                    r = sub.loc[term]
                    put(f"Mod{tag}{tn}FM{w}", f"{f(r.coef)} [{f(r.lo)}, {f(r.hi)}]",
                        f"macro-F1 gap ~ r + binary + set; {spec}; coef of {term}, {fm}; table bootstrap by set (q1_model.csv)")
        for met, mt in (("macro_f1", ""), ("accuracy", "Acc"), ("balanced_accuracy", "BA")):
            r = pred[(pred.metric == met) & (pred.fm == fm)].iloc[0]
            put(f"Pred{mt}FM{w}", f(r.pred), f"{met}; SF mean gap predicted from OpenML (DEV+HO) fit gap ~ r + binary; {fm}")
            put(f"Obs{mt}FM{w}", f(r.obs), f"{met}; SF observed mean gap; {fm}")
            put(f"Res{mt}FM{w}", f"{f(r["diff"])} [{f(r.diff_lo)}, {f(r.diff_hi)}]",
                f"{met}; SF observed minus composition-predicted mean gap, bootstrap CI; {fm}")
        if True:
            sub = mod[(mod.metric == "accuracy") & (mod.fm == fm) & (mod.spec == "main")].set_index("term")
            put(f"ModAccImbFM{w}", f"{f(sub.loc['r'].coef)} [{f(sub.loc['r'].lo)}, {f(sub.loc['r'].hi)}]",
                f"accuracy gap slope on r = pi_min C (main model); {fm}")
            put(f"ModAccSFFM{w}", f"{f(sub.loc['ind_sf'].coef)} [{f(sub.loc['ind_sf'].lo)}, {f(sub.loc['ind_sf'].hi)}]",
                f"accuracy gap SF indicator (main model); {fm}")

    # bins
    long["bin"] = pd.cut(long.r, BINS, right=False, labels=["<0.35", "0.35-0.6", "0.6-0.85", ">=0.85"])
    binrows = []
    for (met, fm, s, b), g in long.groupby(["metric", "fm", "set", "bin"], observed=True):
        m, lo, hi, n = ci(g.gap)
        binrows.append(dict(metric=met, fm=fm, set=s, bin=b, n=n, mean=m, lo=lo, hi=hi))
    bins = pd.DataFrame(binrows)
    bins.to_csv(OUT / "q1_bins.csv", index=False)
    print(bins[bins.metric == "macro_f1"].pivot_table(index=["fm", "bin"], columns="set", values="mean").round(3))
    print(bins[bins.metric == "macro_f1"].pivot_table(index=["fm", "bin"], columns="set", values="n").iloc[:4])
    # composition summary per set
    comp = tr.groupby("set").agg(n_tables=("dataset", "size"), share_binary=("binary", "mean"),
                                 median_r=("r", "median"), median_pimin=("pi_min", "median"),
                                 share_r_below_06=("r", lambda s: float((s < 0.6).mean())),
                                 median_n=("n", "median"))
    comp.to_csv(OUT / "composition_by_set.csv")
    print(comp.round(3))
    for s in SETS:
        c = comp.loc[s]
        tag = {"dev": "Dev", "pmlb": "Pmlb", "heldout": "HO", "sf": "SF"}[s]
        put(f"Comp{tag}ShareBin", f"{100 * c.share_binary:.0f}\\%", f"{SETNAME[s]}: share of binary tables")
        put(f"Comp{tag}MedR", f(c.median_r, 2, False), f"{SETNAME[s]}: median pi_min*C")
        put(f"Comp{tag}ShareImb", f"{100 * c.share_r_below_06:.0f}\\%", f"{SETNAME[s]}: share of tables with pi_min*C < 0.6")

    # plot
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    style = {"dev": ("o", "#4C78A8"), "heldout": ("s", "#54A24B"), "pmlb": ("^", "#BAB0AC"), "sf": ("D", "#E45756")}
    for ax, fm in zip(axes, FMS):
        d = long[(long.metric == "macro_f1") & (long.fm == fm)]
        for s in ("pmlb", "dev", "heldout", "sf"):
            g = d[d.set == s]
            mk, col = style[s]
            ax.scatter(g.r, g.gap, marker=mk, s=18, c=col, alpha=0.8, label=SETNAME[s],
                       facecolors=col if s != "pmlb" else "none", edgecolors=col)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(fm)
        ax.set_xlabel(r"$\pi_{\min} C$ (1 = balanced)")
    axes[0].set_ylabel("macro-F1 gap S-R (mean over m)")
    axes[0].legend(fontsize=7)
    fig.suptitle("EXPLORATORY, post hoc: per-table macro-F1 gap vs pool imbalance", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "q1_gap_vs_imbalance.png", dpi=130)
    plt.close(fig)

    # ---- Q2 decomposition
    dec = decomposition(runs, tr)
    dec_tab = []
    for (s, fm), g in dec.groupby(["set", "fm"]):
        for mlab, sub in (("1", g[g.mult == 1]), ("3", g[g.mult == 3]), ("ge3", g[g.mult >= 3]), ("all", g)):
            per = sub.groupby("dataset")[[f"{m}_{k}" for m in METRICS for k in ("gap", "cov", "bal")] + ["p_miss_obs"]].mean()
            row = dict(set=s, fm=fm, m=mlab, n=len(per))
            for col in per.columns:
                mm, lo, hi, _ = ci(per[col])
                row.update({col: mm, f"{col}_lo": lo, f"{col}_hi": hi})
            if mlab == "1":
                b = sub[sub.C == 2].dropna(subset=["macro_f1_cov_pred"])
                row["bin_cov_pred"] = float(b.macro_f1_cov_pred.mean()) if len(b) else np.nan
                row["bin_cov_obs"] = float(b.macro_f1_cov.mean()) if len(b) else np.nan
                row["bin_n"] = len(b)
            dec_tab.append(row)
    dec_tab = pd.DataFrame(dec_tab)
    dec_tab.to_csv(OUT / "q2_decomposition.csv", index=False)
    show = ["set", "fm", "m", "n", "macro_f1_gap", "macro_f1_cov", "macro_f1_bal", "accuracy_cov", "accuracy_bal",
            "balanced_accuracy_bal", "p_miss_obs"]
    print(dec_tab[dec_tab.m.isin(["1", "ge3", "all"]) & dec_tab.set.isin(["heldout", "sf", "dev"])][show].round(3).to_string())
    print(dec_tab[dec_tab.m == "1"][["set", "fm", "bin_n", "bin_cov_pred", "bin_cov_obs"]].round(3).to_string())
    for fm in FMS:
        w = TEXFM[fm]
        for s, tag in (("sf", "SF"), ("heldout", "HO")):
            for mlab, mt in (("all", "All"), ("ge3", "GeThree")):
                r = dec_tab[(dec_tab.set == s) & (dec_tab.fm == fm) & (dec_tab.m == mlab)].iloc[0]
                for comp_, cn in (("cov", "Cov"), ("bal", "Bal")):
                    col = f"macro_f1_{comp_}"
                    put(f"Dec{tag}{cn}{mt}FM{w}", f"{f(r[col])} [{f(r[col + '_lo'])}, {f(r[col + '_hi'])}]",
                        f"{SETNAME[s]}; macro-F1 {'coverage part' if comp_ == 'cov' else 'balance residual'}; "
                        f"m={'1,3,5,10 mean' if mlab == 'all' else '3,5,10 mean'}; {fm} (q2_decomposition.csv)")
                put(f"Dec{tag}AccBal{mt}FM{w}", f"{f(r['accuracy_bal'])} [{f(r['accuracy_bal_lo'])}, {f(r['accuracy_bal_hi'])}]",
                    f"{SETNAME[s]}; accuracy balance residual; m={mlab}; {fm}")

    # ---- Q3 metric map
    q3 = long.groupby(["set", "fm", "metric"]).gap.mean().unstack("metric")[list(METRICS)]
    q3.to_csv(OUT / "q3_metric_means.csv")
    print(q3.round(3))
    wide = long.pivot_table(index=["set", "dataset", "fm"], columns="metric", values="gap").reset_index()
    wide = wide.merge(tr[["set", "dataset", "r", "binary"]], on=["set", "dataset"])
    wide["ordered"] = (wide.accuracy <= wide.macro_f1) & (wide.macro_f1 <= wide.balanced_accuracy)
    order = wide.groupby("set").ordered.mean()
    order_imb = wide[wide.r < 0.6].groupby("set").ordered.mean()
    notes["ordering_share"] = order.round(3).to_dict()
    notes["ordering_share_r_below_0.6"] = order_imb.round(3).to_dict()
    for s, tag in (("sf", "SF"), ("heldout", "HO"), ("dev", "Dev"), ("pmlb", "Pmlb")):
        put(f"Order{tag}", f"{100 * order[s]:.0f}\\%", f"{SETNAME[s]}: share of table x FM with acc gap <= macro-F1 gap <= BA gap")
    # Prop 2(ii) threshold per binary table
    b = tr[tr.binary == 1].copy()
    p1 = b.pi_min.clip(1e-9, 0.5 - 1e-9)
    tau = np.log((1 - p1) / p1)
    b["Bstar"] = 1.0 / (p1 * (1 - p1) * tau**2)
    b["mstar"] = b.Bstar / 2
    thr = b.groupby("set").agg(n=("mstar", "size"), med_mstar=("mstar", "median"),
                               share_le3=("mstar", lambda s: float((s <= 3).mean())),
                               share_le1=("mstar", lambda s: float((s <= 1).mean())))
    thr.to_csv(OUT / "q3_prop2ii_threshold.csv")
    print(thr.round(3))
    for s, tag in (("sf", "SF"), ("heldout", "HO"), ("dev", "Dev"), ("pmlb", "Pmlb")):
        if s in thr.index:
            put(f"Mstar{tag}Med", f(thr.loc[s].med_mstar, 1, False), f"{SETNAME[s]} binary: median m* = B*/2, B* = 1/(pi1 pi2 tau*^2), prop:balance(ii)")
            put(f"Mstar{tag}ShareLeThree", f"{100 * thr.loc[s].share_le3:.0f}\\%", f"{SETNAME[s]} binary: share with m* <= 3")
    # per-m accuracy and macro-F1 gap per set
    perm = []
    for m in MULTS:
        for met in ("macro_f1", "accuracy", "balanced_accuracy"):
            g = table_gaps(runs, met, (m,))
            for s in SETS:
                for fm in FMS:
                    mm, lo, hi, n = ci(g[g.set == s][fm])
                    perm.append(dict(metric=met, mult=m, set=s, fm=fm, mean=mm, lo=lo, hi=hi, n=n))
    perm = pd.DataFrame(perm)
    perm.to_csv(OUT / "q3_per_m.csv", index=False)
    print(perm[perm.set.isin(["sf", "heldout"])].pivot_table(index=["metric", "fm"], columns=["set", "mult"], values="mean").round(3))

    # ---- Q3b sign condition implied by Lemma 1 + prop:coverage(v) + prop:onelabel(i): at m = 1 on a binary table,
    # gap = P(miss) (M^S - p/(1+p)), so the gap is negative when the stratified macro-F1 is below the
    # macro-F1 of the constant majority prediction, p/(1+p) (p = majority share; pool share used for U).
    sm = runs[runs.strategy == "stratified"].groupby(["set", "dataset", "learner", "mult"]).macro_f1.mean().rename("mS")
    sm = sm.reset_index().merge(tr[["set", "dataset", "pi_min", "binary"]], on=["set", "dataset"])
    sm = sm[sm.binary == 1]
    pmaj = 1 - sm.pi_min
    sm["base"] = pmaj / (1 + pmaj)
    sm["below"] = sm.mS < sm.base
    g1 = table_gaps(runs, "macro_f1", (1,)).melt(id_vars=["set", "dataset"], var_name="learner", value_name="gap1")
    sm1 = sm[sm.mult == 1].merge(g1, on=["set", "dataset", "learner"])
    below = sm1.groupby("set").agg(n=("below", "size"), share_below=("below", "mean"))
    below["gap1_below"] = sm1[sm1.below].groupby("set").gap1.mean()
    below["gap1_above"] = sm1[~sm1.below].groupby("set").gap1.mean()
    below["sign_agree"] = sm1.assign(a=(sm1.gap1 < 0) == sm1.below).groupby("set").a.mean()
    tab_below = sm.groupby(["set", "dataset"]).below.mean().groupby("set").apply(lambda x: float((x >= 0.5).mean()))
    below["share_tables_mostly_below_all_m"] = tab_below
    below.to_csv(OUT / "q3b_majority_baseline.csv")
    print(below.round(3))
    for s_, tag in (("sf", "SF"), ("heldout", "HO"), ("dev", "Dev"), ("pmlb", "Pmlb")):
        r = below.loc[s_]
        put(f"BelowBase{tag}", f"{100 * r.share_below:.0f}\\%",
            f"{SETNAME[s_]} binary, m=1: share of table x FM with stratified macro-F1 < p/(1+p) (constant-majority macro-F1)")
        put(f"BelowBaseGap{tag}", f(r.gap1_below), f"{SETNAME[s_]} binary, m=1: mean gap of those table x FM")
        put(f"BelowBaseSignAgree{tag}", f"{100 * r.sign_agree:.0f}\\%",
            f"{SETNAME[s_]} binary, m=1: share of table x FM where sign(gap<0) matches the condition M^S < p/(1+p)")

    # tables above the baseline at m = 1 (and all multiclass tables): SF vs OpenML, gap over m
    flag = sm1[["set", "dataset", "learner", "below"]].rename(columns={"learner": "fm"})
    ab = []
    for fm in FMS:
        d = long[(long.metric == "macro_f1") & (long.fm == fm)].merge(flag, on=["set", "dataset", "fm"], how="left")
        d["below"] = d.below.fillna(False).astype(bool)
        a = d[~d.below]
        ref, sfa = a[a.set.isin(["dev", "heldout"])].gap, a[a.set == "sf"].gap
        ho = a[a.set == "heldout"].gap
        dd, lo, hi = two_sample_diff(sfa, ref)
        mr, rlo, rhi, rn = ci(ref)
        ms, slo, shi, sn = ci(sfa)
        mh, hlo, hhi, hn = ci(ho)
        ab.append(dict(fm=fm, openml=mr, openml_lo=rlo, openml_hi=rhi, openml_n=rn, ho=mh, ho_lo=hlo, ho_hi=hhi, ho_n=hn,
                       sf=ms, sf_lo=slo, sf_hi=shi, sf_n=sn, diff=dd, diff_lo=lo, diff_hi=hi))
        w = TEXFM[fm]
        put(f"AboveSFFM{w}", f"{f(ms)} [{f(slo)}, {f(shi)}]", f"SF tables not below the majority baseline at m=1; gap over m; {fm}; n={sn}")
        put(f"AboveOpenMLFM{w}", f"{f(mr)} [{f(rlo)}, {f(rhi)}]", f"DEV+HO tables not below the baseline; gap over m; {fm}; n={rn}")
        put(f"AboveDiffFM{w}", f"{f(dd)} [{f(lo)}, {f(hi)}]", f"SF minus DEV+HO, tables not below the baseline; {fm}")
    ab = pd.DataFrame(ab)
    ab.to_csv(OUT / "q3b_above_baseline_gaps.csv", index=False)
    print(ab.round(3).to_string())

    # ---- Q4 matched
    sf_tr = tr[tr.set == "sf"]
    med_all = float(sf_tr.pi_min.median())
    med_bin = float(sf_tr[sf_tr.binary == 1].pi_min.median())
    notes["sf_median_pimin_all"] = med_all
    notes["sf_median_pimin_binary"] = med_bin
    put("SFMedPiMin", f(med_all, 3, False), "median minority share of the 20 second-frame tables")
    q4 = []
    for thr_name, thr_v in (("all20", med_all), ("bin17", med_bin)):
        for met in METRICS:
            for fm in FMS:
                d = long[(long.metric == met) & (long.fm == fm)]
                ref = d[d.set.isin(["dev", "heldout"]) & (d.binary == 1) & (d.pi_min < thr_v)].gap
                ref_ho = d[(d.set == "heldout") & (d.binary == 1) & (d.pi_min < thr_v)].gap
                pml = d[(d.set == "pmlb") & (d.binary == 1) & (d.pi_min < thr_v)].gap
                sfa = d[d.set == "sf"].gap
                sfb = d[(d.set == "sf") & (d.binary == 1) & (d.pi_min < thr_v)].gap
                row = dict(threshold=thr_name, thr=thr_v, metric=met, fm=fm)
                for nm, x in (("openml", ref), ("heldout", ref_ho), ("pmlb", pml), ("sf_all", sfa), ("sf_matched", sfb)):
                    mm, lo, hi, n = ci(x)
                    row.update({f"{nm}_mean": mm, f"{nm}_lo": lo, f"{nm}_hi": hi, f"{nm}_n": n})
                dd, dlo, dhi = two_sample_diff(sfb, ref) if len(ref) and len(sfb) else (np.nan,) * 3
                row.update(diff_sfm_minus_openml=dd, diff_lo=dlo, diff_hi=dhi)
                q4.append(row)
    q4 = pd.DataFrame(q4)
    q4.to_csv(OUT / "q4_matched.csv", index=False)
    print(q4[q4.metric == "macro_f1"].round(3).T.to_string())
    for fm in FMS:
        w = TEXFM[fm]
        r = q4[(q4.threshold == "all20") & (q4.metric == "macro_f1") & (q4.fm == fm)].iloc[0]
        put(f"MatchOpenMLFM{w}", f"{f(r.openml_mean)} [{f(r.openml_lo)}, {f(r.openml_hi)}]",
            f"binary DEV+HO tables with pi_min < SF median; macro-F1 gap; {fm}")
        put(f"MatchSFFM{w}", f"{f(r.sf_matched_mean)} [{f(r.sf_matched_lo)}, {f(r.sf_matched_hi)}]",
            f"binary SF tables with pi_min < SF median; macro-F1 gap; {fm}")
        put(f"MatchDiffFM{w}", f"{f(r.diff_sfm_minus_openml)} [{f(r.diff_lo)}, {f(r.diff_hi)}]",
            f"matched SF minus matched OpenML, two-sample bootstrap; {fm}")
        put(f"MatchPmlbFM{w}", f"{f(r.pmlb_mean)} [{f(r.pmlb_lo)}, {f(r.pmlb_hi)}]",
            f"binary PMLB/UCI tables with pi_min < SF median; macro-F1 gap; {fm}")
    r0 = q4[(q4.threshold == "all20") & (q4.metric == "macro_f1")].iloc[0]
    put("MatchOpenMLN", str(int(r0.openml_n)), "n binary DEV+HO tables below the SF median pi_min")
    put("MatchHON", str(int(r0.heldout_n)), "of which HELD-OUT")
    put("MatchSFN", str(int(r0.sf_matched_n)), "n binary SF tables below the SF median pi_min")
    put("MatchPmlbN", str(int(r0.pmlb_n)), "n binary PMLB/UCI tables below the SF median pi_min")

    # ---- Q1b table signal (stored metrics): is the frame residual a low-signal effect?
    sig = signal(runs, tr)
    long = long.merge(sig, on=["set", "dataset"], how="left")
    sigset = sig.groupby("set").agg(median_signal=("signal", "median"),
                                    share_low=("signal", lambda s: float((s < 0.2).mean())),
                                    median_ba10=("ba10", "median"))
    sigset.to_csv(OUT / "q1b_signal_by_set.csv")
    print(sigset.round(3))
    for s_ in SETS:
        tag = {"dev": "Dev", "pmlb": "Pmlb", "heldout": "HO", "sf": "SF"}[s_]
        put(f"Sig{tag}Med", f(sigset.loc[s_].median_signal, 2, False),
            f"{SETNAME[s_]}: median table signal = (BA - 1/C)/(1 - 1/C) at m=10, mean of S and R, 4 FMs")
        put(f"Sig{tag}ShareLow", f"{100 * sigset.loc[s_].share_low:.0f}\\%", f"{SETNAME[s_]}: share of tables with signal < 0.2")
    sig_rows, sig_pred = [], []
    for met in METRICS:
        for fm in FMS:
            d = long[(long.metric == met) & (long.fm == fm)].dropna(subset=["gap"]).reset_index(drop=True)
            t = ols_boot(d, "gap", ("signal",))
            t["metric"], t["fm"] = met, fm
            sig_rows.append(t)
            sig_pred.append(dict(metric=met, fm=fm, **composition_prediction(d, "gap", extra=("signal",))))
    sigmod = pd.concat(sig_rows, ignore_index=True)
    sigmod.to_csv(OUT / "q1b_model_with_signal.csv", index=False)
    sigpred = pd.DataFrame(sig_pred)
    sigpred.to_csv(OUT / "q1b_prediction_with_signal.csv", index=False)
    print(sigmod[sigmod.metric == "macro_f1"].round(3).to_string())
    print(sigpred.round(3).to_string())
    for fm in FMS:
        w = TEXFM[fm]
        sub = sigmod[(sigmod.metric == "macro_f1") & (sigmod.fm == fm)].set_index("term")
        for term, tn in (("ind_sf", "SF"), ("signal", "Sig")):
            r = sub.loc[term]
            put(f"ModSig{tn}FM{w}", f"{f(r.coef)} [{f(r.lo)}, {f(r.hi)}]",
                f"macro-F1 gap ~ r + binary + signal + set; coef of {term}; {fm} (q1b_model_with_signal.csv)")
        r = sigpred[(sigpred.metric == "macro_f1") & (sigpred.fm == fm)].iloc[0]
        put(f"ResSigFM{w}", f"{f(r['diff'])} [{f(r.diff_lo)}, {f(r.diff_hi)}]",
            f"SF observed minus predicted from OpenML fit gap ~ r + binary + signal; {fm}")
        put(f"PredSigFM{w}", f(r.pred), f"SF mean gap predicted from OpenML fit gap ~ r + binary + signal; {fm}")
    # shares of the shortfall (OpenML mean - SF mean) explained by composition and by composition + signal
    shares = []
    for fm in FMS:
        d = long[(long.metric == "macro_f1") & (long.fm == fm)]
        ref = float(d[d.set.isin(["dev", "heldout"])].gap.mean())
        p0 = float(pred[(pred.metric == "macro_f1") & (pred.fm == fm)].iloc[0].pred)
        p1 = float(sigpred[(sigpred.metric == "macro_f1") & (sigpred.fm == fm)].iloc[0].pred)
        obs = float(d[d.set == "sf"].gap.mean())
        short = ref - obs
        shares.append(dict(fm=fm, openml_mean=ref, pred_comp=p0, pred_comp_signal=p1, sf_obs=obs, shortfall=short,
                           share_comp=(ref - p0) / short, share_comp_signal=(ref - p1) / short))
        w = TEXFM[fm]
        put(f"OpenMLMeanFM{w}", f(ref), f"DEV+HELD-OUT pooled mean macro-F1 gap; {fm}")
        put(f"ShareCompFM{w}", f"{100 * (ref - p0) / short:.0f}\\%", f"share of the OpenML-minus-SF shortfall explained by r + binary; {fm}")
        put(f"ShareCompSigFM{w}", f"{100 * (ref - p1) / short:.0f}\\%", f"share explained by r + binary + signal; {fm}")
    shares = pd.DataFrame(shares)
    shares.to_csv(OUT / "q1_shortfall_shares.csv", index=False)
    print(shares.round(3).to_string())

    # stratified macro-F1 level at m=1 (drives the coverage part, prop:coverage(v))
    lvl = runs[(runs.mult == 1) & (runs.strategy == "stratified")].groupby(["set", "dataset"]).macro_f1.mean()
    lvl = lvl.groupby("set").median()
    notes["median_stratified_macro_f1_m1"] = lvl.round(3).to_dict()
    print("median S macro-F1 at m=1:", lvl.round(3).to_dict())
    for s_ in SETS:
        tag = {"dev": "Dev", "pmlb": "Pmlb", "heldout": "HO", "sf": "SF"}[s_]
        put(f"LevelSOne{tag}", f(lvl[s_], 2, False), f"{SETNAME[s_]}: median over tables of stratified macro-F1 at m=1 (4 FMs)")

    # ---- Q5 pipeline
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_second_frame.yaml").read_text())
    sc = spot_checks(pool)
    pd.DataFrame(sc["counts"]).to_csv(OUT / "q5_class_counts.csv", index=False)
    rec = pd.DataFrame(sc["recompute"])
    rec.to_csv(OUT / "q5_recompute.csv", index=False)
    sc["per_table_fm"].to_csv(OUT / "q5_degeneracy.csv", index=False)
    print(rec.to_string())
    print("counts ok:", sc["counts_all_ok"], "degenerate draw share:", sc["degenerate_draw_share"],
          "near-chance tables:", sc["near_chance_tables"])
    print("const-prediction share by strategy/m:", sc["by_strategy"])
    maxd = float(np.nanmax(rec[["d_f1", "d_acc", "d_bacc"]].to_numpy()))
    put("QFiveMaxAbsDiff", f"{maxd:.1e}", "max |recomputed - stored| metric over the spot-checked S and R draws")
    put("QFiveConstShare", f"{100 * sc['degenerate_draw_share']:.1f}\\%",
        "share of SF arm-T S/R draws whose argmax predicts one class for >= 99% of query rows")
    notes["q5"] = dict(counts_all_ok=sc["counts_all_ok"], degenerate_draw_share=sc["degenerate_draw_share"],
                       near_chance_tables=sc["near_chance_tables"], max_abs_diff=maxd,
                       const_by_strategy_m={f"{k[0]}|{k[1]}": v for k, v in sc["by_strategy"].items()})
    re_ = reencode(pool, ["f33111881_frailty", "z22705412_hpv", "z22216010_pvt", "z22767593_edmort",
                          "f33328002_tea", "k11395735_malaria"])
    (OUT / "q5_reencode.json").write_text(json.dumps(re_, indent=1, ensure_ascii=False))
    print(json.dumps(re_, ensure_ascii=False))
    notes["q5"]["reencode_all_equal"] = all(r["y_equal"] and r["X_equal"] for r in re_)
    cs = pd.concat([const_share(RES / "study13" / "heldout").assign(set="heldout"),
                    const_share(RES / "study15" / "second_frame").assign(set="sf")])
    cst = cs.groupby(["set", "strategy", "mult"]).const.mean().unstack("mult").round(3)
    cst.to_csv(OUT / "q5_const_share_ho_sf.csv")
    print(cst)
    for s_, tag in (("heldout", "HO"), ("sf", "SF")):
        for m_, mt in ((3, "Three"), (10, "Ten")):
            put(f"Const{tag}RM{mt}", f"{100 * cst.loc[(s_, 'random'), m_]:.0f}\\%",
                f"{SETNAME[s_]}: share of random arm-T draws (m={m_}, 4 FMs) whose argmax gives one class to >= 99% of rows")
    # per-table SF gaps (SF and public sets only; no HELD-OUT rows)
    pub = long[long.set != "heldout"].pivot_table(index=["set", "dataset", "C", "pi_min", "r"], columns=["metric", "fm"],
                                                 values="gap")
    pub.columns = [f"{a}_{b}" for a, b in pub.columns]
    pub.reset_index().to_csv(OUT / "per_table_gaps_public_sets.csv", index=False)

    (OUT / "notes.json").write_text(json.dumps(notes, indent=1, default=str))
    (HERE / "sf_autopsy_numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 1} macros")


if __name__ == "__main__":
    main()
