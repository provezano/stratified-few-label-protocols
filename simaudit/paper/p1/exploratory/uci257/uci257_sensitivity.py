"""UCI 257 label-spelling sensitivity (EXPLORATORY; not pre-registered; no confirmatory weight).

T1' table pmlb_1000000257 (UCI 257, User Knowledge Modeling) is stored with C = 5, but
two of its label strings, "Very Low" and "very_low", are the same class (the declared
spelling rule of simaudit.data.labels merges them). The true C is 4.

Inputs (read-only): results/study9/t1prime/*.parquet (stored Study 9 fits),
data/cache/npz (stored codes, via simaudit.data.openml_loader.load), data/cache/uci/257.csv
(raw label strings, checked against the frozen UCI snapshot and the pool file hash),
data/pool_t1prime.yaml and data/pool_ext_uci.yaml. Nothing is refitted. sealed_heldout/
is never touched. Study 12 is not used: its run had not finished when this ran
(logs/study12_run.log at 206/208 jobs, process alive).

Steps
1. Confirm the issue: raw strings, counts, code -> string map, normalised strings, the
   merge the declared rule makes, and where each spelling sits in the raw file.
2. Sensitivity (a), DROP: every quantity recomputed on the other 51 tables.
3. Sensitivity (b), MERGE: the stored probability matrices of table 257 are re-scored
   with the columns of "Very Low" and "very_low" summed and the true labels merged
   (C = 4). For draws that do not use the labels (random, k-medoids, LUCoS) this is a
   valid post-hoc evaluation of the same draw. For stratified draws it is only
   approximate: the draw stratified over 5 classes, so the merged class received 2m of
   the 5m labels (a 4-class stratified draw would give it m of 4m), and the stratified
   grid stops at m = 15 (smallest 5-class pool class).
   Budget unit: the draw at nominal m has B = 5m labels, i.e. 1.25m per merged class.
   rho = m_eq / m is a ratio of budgets, so it does not depend on this unit. The
   per-m gap is reported at the nominal m (B = 5m) for this one table. For m* (a slope
   per label per class), the 257 curve is put on m' = 1.25m and its interval is widened
   outward to the frozen grid (conservative).

Quantities (all on stored fits; curves = mean over repetitions as in analysis.study9):
- Gap per m: stratified - random macro-F1 per table, mean over tables with a 10k
  bootstrap CI and Wilcoxon p (as E15 out/e15_paired.csv), 4 FMs.
- rho(1) of the stratified protocol (H2a, TabICL): Turnbull median, 2000-resample CI
  (prereg seed), # tables with rho > 1, share, sign-test p; medians at m in {1,..,10};
  TabPFN v2 replication; the two selectors.
- Binary/multiclass split: counts per C bin; H1 m* distribution (TabICL, random,
  eps = 0.01, 2000 resamples) overall and per C bin; AFT beta for log C (H1b);
  rho(1) within binary and multiclass tables (E14 form).

Writes out/uci257_labels.csv, out/uci257_gap.csv, out/uci257_rho.csv,
out/uci257_mstar.csv, out/uci257_aft.csv, out/uci257_split.csv, out/uci257_rescore_check.csv,
out/UCI257_SUMMARY.md, out/uci257_numbers.tex.

Run (from simaudit/): OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python paper/p1/exploratory/uci257/uci257_sensitivity.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib  # noqa: E402
from math import inf  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

from simaudit.analysis import study9 as a9  # noqa: E402
from simaudit.analysis.report import bootstrap_ci  # noqa: E402
from simaudit.analysis.study7 import metric_set  # noqa: E402
from simaudit.data import labels  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.runner import build_pool_uci as bpu  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
S9 = settings.results_dir / "study9" / "t1prime"
DID, KEY, UCI_ID = -1000000257, "pmlb_1000000257", 257
FMS = ("fm2", "fm1", "fm3", "fm5")
GRID = (1, 2, 3, 5, 7, 10, 15, 20, 30)
SCALE = 5 / 4  # labels per merged class per nominal m
COLS = ["dataset", "learner", "strategy", "m", "rep", "n_classes", "macro_f1", "accuracy"]
VARIANTS = ("base", "drop", "merge")


# ------------------------------------------------------------------ step 1
def label_check() -> tuple[np.ndarray, labels.LabelRecord, pd.DataFrame, list[str]]:
    """Raw strings of UCI 257, verified against the stored codes and the frozen sources."""
    uci = yaml.safe_load((PROJECT_ROOT / "data" / "pool_ext_uci.yaml").read_text())
    entry = next(d for d in uci["datasets"] if int(d["uci_id"]) == UCI_ID)
    snap = bpu.load_snapshot(PROJECT_ROOT / "data" / "uci_api_snapshot.json.gz", uci["snapshot_sha256"])
    meta = next(d for d in snap["datasets"] if int(d["uci_id"]) == UCI_ID)
    target = bpu.target_column(meta)
    csv = settings.data_cache_dir / "uci" / f"{UCI_ID}.csv"
    raw_bytes = csv.read_bytes()
    file_ok = hashlib.sha256(raw_bytes).hexdigest() == entry["file_sha256"]
    frame = pd.read_csv(csv)
    X_chk, y_chk = bpu.parse_csv(raw_bytes, meta)
    data = load(DID, settings.data_cache_dir)
    raw = frame[target].astype(str).to_numpy()
    rec = labels.label_record(DID, data.y, raw.astype(np.str_))
    notes = [
        f"target column (frozen snapshot): {target!r}",
        f"raw file SHA-256 matches pool_ext_uci.yaml file_sha256: {file_ok}",
        f"parse_csv(raw) reproduces the stored X and y: "
        f"{np.array_equal(X_chk, data.X) and np.array_equal(y_chk, data.y)}",
        f"labels.label_record status: {rec.status}; merges: {rec.merges}",
    ]
    rows = []
    for code, name in enumerate(rec.raw):
        pos = np.flatnonzero(raw == name)
        rows.append(dict(code=code, raw=name, normalised=rec.normalised[code], merged_code=rec.mapping[code],
                         count=int(len(pos)), first_row=int(pos.min()), last_row=int(pos.max()),
                         rows_0_257=int((pos < 258).sum()), rows_258_402=int((pos >= 258).sum())))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "uci257_labels.csv", index=False)
    return data.y, rec, tab, notes


# ------------------------------------------------------------------ re-scoring
def rescore(y: np.ndarray, mapping: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every stored fit of table 257: original scores re-derived (check) and merged scores."""
    C, K = len(mapping), int(mapping.max()) + 1
    out, chk = [], []
    for fm in FMS:
        df = pd.read_parquet(S9 / f"{KEY}_{fm}.parquet")
        for r in df.itertuples(index=False):
            q = np.asarray(r.query, dtype=np.int64)
            P = np.asarray(r.proba, dtype=np.float64).reshape(len(q), C)
            seen = np.asarray(r.seen, dtype=np.int64)
            yt = y[q]
            orig = metric_set(yt, P, np.arange(C, dtype=np.int64), seen)
            Pm = np.zeros((len(q), K))
            np.add.at(Pm.T, mapping, P.T)
            mm = metric_set(mapping[yt], Pm, np.arange(K, dtype=np.int64), np.unique(mapping[seen]))
            chk.append(dict(learner=fm, strategy=r.strategy, m=r.m, rep=r.rep,
                            d_f1=orig["macro_f1"] - r.macro_f1, d_acc=orig["accuracy"] - r.accuracy))
            out.append(dict(dataset=KEY, learner=fm, strategy=r.strategy, m=r.m, rep=r.rep, n_classes=K,
                            macro_f1=mm["macro_f1"], accuracy=mm["accuracy"],
                            macro_f1_c5=r.macro_f1, accuracy_c5=r.accuracy))
    return pd.DataFrame(out), pd.DataFrame(chk)


# ------------------------------------------------------------------ helpers
def c_bin(c: int) -> str:
    return "C2" if c == 2 else "C3-4" if c <= 4 else "C5-10"


def summarize(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, float)
    lo, hi = bootstrap_ci(x)
    p = float(stats.wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
    return dict(n=len(x), mean=float(x.mean()), lo=lo, hi=hi, median=float(np.median(x)),
                win=float(np.mean(x > 0)), p=p)


def widen(iv: tuple[float, float]) -> tuple[float, float]:
    """Interval on m' = 1.25 m, widened outward to the frozen grid cells."""
    lo, hi = iv
    lo_g = max([g for g in GRID if g <= lo], default=0.0) if lo > 0 else 0.0
    hi_g = inf if hi == inf else min([g for g in GRID if g >= hi], default=inf)
    return float(lo_g), float(hi_g)


def mstar_intervals(curve: pd.DataFrame, learner: str, merged: bool) -> pd.Series:
    sub = a9._xs(curve, learner, "random")
    out = {}
    for k, r in sub.iterrows():
        ms, f = a9._points(r)
        if merged and k == KEY:
            out[k] = widen(a9.saturation_interval(ms * SCALE, f))
        else:
            out[k] = a9.saturation_interval(ms, f)
    return pd.Series(out, dtype=object)


def traits() -> pd.DataFrame:
    """n, d, C, IR per T1' table (as analysis.study9.dataset_traits, without H3 budgets)."""
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())["datasets"]
    rows = []
    for d in pool:
        data = load(int(d["did"]), settings.data_cache_dir)
        full = np.bincount(data.y)
        rows.append(dict(dataset=f"pmlb_{-int(d['did'])}" if int(d["did"]) < 0 else f"openml_{d['did']}",
                         n=len(data.y), d=data.X.shape[1], C=int(data.y.max()) + 1,
                         IR=float(full.max() / full.min())))
    return pd.DataFrame(rows).set_index("dataset")


# ------------------------------------------------------------------ analysis
def analyse(raw: pd.DataFrame, tr: pd.DataFrame, variant: str) -> dict[str, pd.DataFrame]:
    curve = a9.curves(raw)
    merged = variant == "merge"
    keys = sorted(set(raw.dataset))
    tr = tr.loc[keys]
    # gap per m
    gap = []
    for fm in FMS:
        st, rd = a9._xs(curve, fm, "stratified"), a9._xs(curve, fm, "random")
        for m in GRID:
            if m not in st.columns:
                continue
            d = (st[m] - rd[m]).dropna()
            gap.append(dict(variant=variant, learner=fm, m=m, **summarize(d.to_numpy())))
    # rho
    rho = []
    for fm in ("fm2", "fm1"):
        rnd = a9._xs(curve, fm, "random")
        for st in ("stratified", *a9.SELECTORS):
            r = a9._rho_intervals(a9._xs(curve, fm, st), rnd)
            ivs = r[1]
            wins = sum(lo > 1 for lo, _ in ivs)
            lo, hi = a9.boot_ci(ivs, a9.turnbull_median_ratio)
            rec = dict(variant=variant, learner=fm, strategy=st, group="all", n=len(ivs),
                       median=a9.turnbull_median_ratio(ivs), lo=lo, hi=hi, count_gt1=wins,
                       share_gt1=wins / len(ivs), p_sign=a9.sign_test_greater(wins, len(ivs)),
                       left_cens=int(sum(a == 0 for a, _ in ivs)), right_cens=int(sum(b == inf for _, b in ivs)))
            for m in (1, 2, 3, 5, 7, 10):
                rec[f"med_m{m}"] = a9.turnbull_median_ratio(r[m]) if r[m] else np.nan
            rho.append(rec)
        # within binary / multiclass (stratified, frozen form; E14)
        sub = a9._xs(curve, fm, "stratified")
        for g, ks in (("binary", tr.index[tr.C == 2]), ("multiclass", tr.index[tr.C > 2])):
            r = a9._rho_intervals(sub.loc[sub.index.intersection(ks)], rnd.loc[rnd.index.intersection(ks)])[1]
            lo, hi = a9.boot_ci(r, a9.turnbull_median_ratio)
            wins = sum(a > 1 for a, _ in r)
            rho.append(dict(variant=variant, learner=fm, strategy="stratified", group=g, n=len(r),
                            median=a9.turnbull_median_ratio(r), lo=lo, hi=hi, count_gt1=wins,
                            share_gt1=wins / len(r), p_sign=a9.sign_test_greater(wins, len(r))))
    # m* (H1, TabICL, random)
    ivs = mstar_intervals(curve, "fm2", merged)
    ms_rows = []
    groups = [("all", ivs.index)] + [(b, tr.index[tr.C.map(c_bin) == b]) for b in ("C2", "C3-4", "C5-10")]
    for g, ks in groups:
        v = ivs.loc[ivs.index.intersection(ks)].tolist()
        dsum = a9.distribution_summary(v, n_boot=a9.N_BOOT)
        row = dict(variant=variant, group=g, n=len(v), median=str(dsum["median"]),
                   q25=str(dsum["q25"]), q75=str(dsum["q75"]))
        for k in ("P(m*<=5)", "P(m*<=10)", "P(m*>20)"):
            row[k], row[k + "_lo"], row[k + "_hi"] = dsum[k]
        ms_rows.append(row)
    # AFT (H1b)
    cov = tr.loc[ivs.index, ["C", "IR", "n", "d"]].astype(float).apply(np.log).add_prefix("log_")
    fit = a9.aft_fit(ivs.tolist(), cov)
    aft = pd.DataFrame([dict(variant=variant, coef=fit.loc["log_C", "coef"], se=fit.loc["log_C", "se"],
                             p_greater=fit.loc["log_C", "p_greater"], beta_IR=fit.loc["log_IR", "coef"],
                             n=len(ivs))])
    split = pd.DataFrame([dict(variant=variant, n=len(tr), binary=int((tr.C == 2).sum()),
                               multiclass=int((tr.C > 2).sum()),
                               **{b: int((tr.C.map(c_bin) == b).sum()) for b in ("C2", "C3-4", "C5-10")})])
    return dict(gap=pd.DataFrame(gap), rho=pd.DataFrame(rho), mstar=pd.DataFrame(ms_rows), aft=aft, split=split)


# ------------------------------------------------------------------ output
WORD = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "5": "Five", "7": "Seven"}
FMW = dict(fm1="FMone", fm2="FMtwo", fm3="FMthree", fm5="FMfive")


def mw(m: int) -> str:
    return "M" + "".join(WORD[c] for c in str(m))


def sgn(x: float, d: int = 3) -> str:
    return f"$+${x:.{d}f}" if x >= 0 else f"$-${abs(x):.{d}f}"


def pfmt(p: float) -> str:
    if p >= 0.01:
        return f"{p:.2f}"
    e = int(np.floor(np.log10(p)))
    return f"${p / 10**e:.1f}\\times10^{{{e}}}$"


def macros(res: dict[str, dict[str, pd.DataFrame]], lab: pd.DataFrame) -> list[str]:
    L = ["% AUTO-GENERATED by paper/p1/exploratory/uci257/uci257_sensitivity.py. EXPLORATORY "
         "(UCI 257 label-spelling sensitivity; stored Study 9 fits only; not pre-registered). Do not edit."]

    def add(name: str, val: str, src: str) -> None:
        L.append(f"\\newcommand{{\\ExUCI{name}}}{{{val}}} % EXPLORATORY; source: {src}")

    for r in lab.itertuples():
        nm = "".join(w.capitalize() for w in r.raw.replace("_", " ").split())
        tag = "Underscore" if "_" in r.raw else ""
        add(f"Raw{nm}{tag}N", str(r.count), "out/uci257_labels.csv (count of raw string)")
    add("MergedClassN", str(int(lab[lab.merged_code == lab.merged_code.max()]["count"].sum())),
        "out/uci257_labels.csv (very low after merge)")
    for v in ("Drop", "Merge"):
        vv = v.lower()
        g = res[vv]["gap"]
        for fm in FMS:
            for m in (1, 3, 5, 10):
                row = g[(g.learner == fm) & (g.m == m)].iloc[0]
                src = f"out/uci257_gap.csv ({vv}, {fm}, m={m})"
                add(f"{v}Gap{FMW[fm]}{mw(m)}", sgn(row["mean"]), src)
                if fm == "fm2":
                    add(f"{v}Gap{FMW[fm]}{mw(m)}Lo", sgn(row.lo), src)
                    add(f"{v}Gap{FMW[fm]}{mw(m)}Hi", sgn(row.hi), src)
        rho = res[vv]["rho"]
        for fm, tag in (("fm2", ""), ("fm1", "FMone")):
            for st, sw in (("stratified", "Strat"), ("kmedoids_nested", "KMn"), ("lucos_nested", "LUn")):
                row = rho[(rho.learner == fm) & (rho.strategy == st) & (rho.group == "all")].iloc[0]
                src = f"out/uci257_rho.csv ({vv}, {fm}, {st}, all)"
                add(f"{v}{tag}RhoOne{sw}", f"{row['median']:.2f}", src)
                add(f"{v}{tag}Rho{sw}Share", f"{row.share_gt1:.2f}", src)
                add(f"{v}{tag}Rho{sw}P", pfmt(row.p_sign), src)
                if st == "stratified":
                    add(f"{v}{tag}RhoOne{sw}Lo", f"{row.lo:.2f}", src)
                    add(f"{v}{tag}RhoOne{sw}Hi", f"{row.hi:.2f}", src)
                    add(f"{v}{tag}Rho{sw}Count", str(int(row.count_gt1)), src)
                    add(f"{v}{tag}Rho{sw}N", str(int(row.n)), src)
                    if fm == "fm2":
                        add(f"{v}RhoOne{sw}LeftCens", str(int(row.left_cens)), src)
                        add(f"{v}Holm{sw}Bound", pfmt(min(1.0, 9 * row.p_sign)), src + " x9 (Holm if smallest of 9)")
                        for m in (2, 3, 5, 7, 10):
                            add(f"{v}Rho{sw}Med{mw(m)}", f"{row[f'med_m{m}']:.2f}", src)
            for grp, gw in (("binary", "Bin"), ("multiclass", "Multi")):
                row = rho[(rho.learner == fm) & (rho.strategy == "stratified") & (rho.group == grp)].iloc[0]
                src = f"out/uci257_rho.csv ({vv}, {fm}, stratified, {grp})"
                add(f"{v}{tag}RhoOneStrat{gw}", f"{row['median']:.2f}", src)
                add(f"{v}{tag}RhoOneStrat{gw}Lo", f"{row.lo:.2f}", src)
                add(f"{v}{tag}RhoOneStrat{gw}Hi", f"{row.hi:.2f}", src)
        sp = res[vv]["split"].iloc[0]
        src = f"out/uci257_split.csv ({vv})"
        add(f"{v}N", str(int(sp.n)), src)
        add(f"{v}BinCTwoN", str(int(sp.C2)), src)
        add(f"{v}MultiN", str(int(sp.multiclass)), src)
        add(f"{v}BinCThreeFourN", str(int(sp["C3-4"])), src)
        add(f"{v}BinCFiveTenN", str(int(sp["C5-10"])), src)
        ms = res[vv]["mstar"]
        for grp, gw in (("all", "All"), ("C3-4", "CThreeFour"), ("C5-10", "CFiveTen")):
            row = ms[ms.group == grp].iloc[0]
            src = f"out/uci257_mstar.csv ({vv}, fm2 random, {grp})"
            a, b = eval(row["median"])  # noqa: S307 - own tuple repr
            add(f"{v}HOne{gw}Median", f"$({a:g}, {b:g}]$", src)
            for k, kw in (("P(m*<=5)", "Pfive"), ("P(m*<=10)", "Pten"), ("P(m*>20)", "Ptwenty")):
                add(f"{v}HOne{gw}{kw}", f"{row[k]:.2f}", src)
                add(f"{v}HOne{gw}{kw}Lo", f"{row[k + '_lo']:.2f}", src)
                add(f"{v}HOne{gw}{kw}Hi", f"{row[k + '_hi']:.2f}", src)
        a = res[vv]["aft"].iloc[0]
        src = f"out/uci257_aft.csv ({vv}, log_C)"
        add(f"{v}BetaC", f"{a.coef:.3f}", src)
        add(f"{v}BetaCSE", f"{a.se:.3f}", src)
        add(f"{v}BetaCP", f"{a.p_greater:.2f}", src)
    return L


def main() -> None:
    y, rec, lab, notes = label_check()
    mapping = np.asarray(rec.mapping, dtype=np.int64)
    merged257, chk = rescore(y, mapping)
    chk.to_csv(OUT / "uci257_rescore_check.csv", index=False)
    paths = sorted(p for p in S9.glob("*.parquet") if "partial" not in p.name)
    base = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    assert base.dataset.nunique() == 52, base.dataset.nunique()
    tr = traits()
    tr_m = tr.copy()
    y_m = mapping[y]
    tr_m.loc[KEY, "C"] = int(y_m.max()) + 1
    tr_m.loc[KEY, "IR"] = float(np.bincount(y_m).max() / np.bincount(y_m).min())
    raws = dict(base=base, drop=base[base.dataset != KEY],
                merge=pd.concat([base[base.dataset != KEY], merged257[COLS]], ignore_index=True))
    res = {v: analyse(raws[v], tr_m if v == "merge" else tr, v) for v in VARIANTS}
    for k in ("gap", "rho", "mstar", "aft", "split"):
        pd.concat([res[v][k] for v in VARIANTS]).to_csv(OUT / f"uci257_{k}.csv", index=False)
    merged257.to_csv(OUT / "uci257_merged_scores.csv", index=False)
    own = []  # table 257 itself: C = 5 as stored vs merged (C = 4)
    for lab_, col in (("stored C=5", "macro_f1_c5"), ("merged C=4", "macro_f1")):
        cv = merged257.groupby(["learner", "strategy", "m"])[col].mean().unstack("m")
        for fm in FMS:
            ms, fr = a9._points(cv.loc[(fm, "random")])
            fs = cv.loc[(fm, "stratified")]
            iv = a9.equivalence_interval(ms, fr, fs[1], 1)
            own.append(dict(scoring=lab_, learner=fm, f_strat_m1=fs[1], f_rand_m1=fr[0],
                            gap_m1=fs[1] - fr[0], gap_m3=fs[3] - cv.loc[(fm, "random"), 3],
                            gap_m10=fs[10] - cv.loc[(fm, "random"), 10], rho1_lo=iv[0], rho1_hi=iv[1]))
    own = pd.DataFrame(own)
    own.to_csv(OUT / "uci257_table257.csv", index=False)
    (OUT / "uci257_numbers.tex").write_text("\n".join(macros(res, lab)) + "\n")

    # ----------------------------------------------------------- summary
    S = ["# UCI 257 label-spelling sensitivity (EXPLORATORY)", "",
         "Exploratory, post hoc, not pre-registered; no confirmatory weight. Stored Study 9 T1' fits only "
         "(results/study9/t1prime); nothing refitted. Study 12 not used (run unfinished at analysis time).", "",
         "## 1. The issue", "", *[f"- {n}" for n in notes], "", lab.to_markdown(index=False), "",
         f"- re-derived 5-class scores vs stored: max |d macro-F1| = {chk.d_f1.abs().max():.2e}, "
         f"max |d accuracy| = {chk.d_acc.abs().max():.2e} over {len(chk)} fits", "",
         "Table 257 itself (stored 5-class scores vs merged re-score; stratified merged = approximate):", "",
         own.round(4).to_markdown(index=False), ""]
    S += ["## 2. Gap per m (stratified - random macro-F1; mean over tables, 10k bootstrap CI)", ""]
    g = pd.concat([res[v]["gap"] for v in VARIANTS])
    piv = g.assign(cell=lambda d: d["mean"].map(lambda x: f"{x:+.4f}") + " [" + d.lo.map(lambda x: f"{x:+.3f}")
                   + ", " + d.hi.map(lambda x: f"{x:+.3f}") + "] n=" + d.n.astype(str))
    S += [piv.pivot_table(index=["learner", "m"], columns="variant", values="cell", aggfunc="first")
          [list(VARIANTS)].reset_index().to_markdown(index=False), ""]
    S += ["## 3. rho(1)", "", pd.concat([res[v]["rho"] for v in VARIANTS]).round(4).to_markdown(index=False), ""]
    S += ["## 4. m* (TabICL, random, eps 0.01, 2000 resamples)", "",
          pd.concat([res[v]["mstar"] for v in VARIANTS]).round(3).to_markdown(index=False), ""]
    S += ["## 5. AFT log C (H1b) and split counts", "",
          pd.concat([res[v]["aft"] for v in VARIANTS]).round(4).to_markdown(index=False), "",
          pd.concat([res[v]["split"] for v in VARIANTS]).to_markdown(index=False), ""]
    S += ["## Caveats", "",
          "- MERGE, stratified draws of 257: approximate. The draw stratified over 5 classes, so the merged "
          "class got 2m of 5m labels, and the stratified grid stops at m = 15. A true 4-class stratified draw "
          "was not run (no refit allowed).",
          "- MERGE, budget unit: nominal m on 257 means B = 5m = 1.25m per merged class. rho is a budget ratio "
          "(unit-free); the per-m gap uses nominal m; m* of 257 is put on m' = 1.25m and widened to grid cells.",
          "- Holm: only the H2a stratified p is recomputed; the bound 9p equals the Holm value only while it "
          "stays the smallest of the 9 family p-values (H3b, H4 not recomputed here).",
          "- Study 12 (primary gap) not analysed: run unfinished (logs/study12_run.log 206/208, process alive).", ""]
    (OUT / "UCI257_SUMMARY.md").write_text("\n".join(S) + "\n")
    print("\n".join(S))


if __name__ == "__main__":
    main()
