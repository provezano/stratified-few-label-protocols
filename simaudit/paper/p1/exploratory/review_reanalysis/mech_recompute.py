"""Round-2 CRITICAL #1: the Sec. 6.5 coverage/balance numbers and Fig. 4(a), recomputed on
the stored pre-registered fits (Study 7 Part (a) DEV, Study 12 T1') for all four
pre-registered FMs. A re-analysis of pre-registered fits; EXPLORATORY as an analysis.

Definitions are those of reports/theory_checks.py (Sec. 4) and reports/coverage_model.py,
which produced the Study 4 (FM5) numbers now in main.tex:
- q_k = hypergeometric P(a uniform B-draw misses class k), full-table counts, B = mC.
- Fig. 4(a) "coverage alone" (Prop. 1(iii), first order): f1_gap_pred = mean_k q_k f_k,
  f_k = mean per-class F1 of the stratified draws on their evaluation rows; against the
  observed gap f1_gap_obs = mean S macro-F1 - mean R macro-F1.
- Decomposition: f1_miss_obs = mean over R draws of (S mean - R draw) x 1[draw missed];
  f1_bal_obs = gap - f1_miss_obs. f1_cov_pred: exact binary formula (Prop. 1(v);
  q_k (M^S - p/(1+p)), p = (n_other - B)/(n - B)), first order for multiclass. Same for
  accuracy (binary exact: q_k (A^S - p); multiclass first order q_k pi_k f_k as in
  theory_checks).
- Balance, FITTED (not predicted): kappa = leave-one-table-out least-squares slope through
  the origin of -(R macro-F1 - S mean) on D (squared distance of the context shares from
  uniform), covered R draws of the other tables at the same m; f1_bal_fit = kappa x
  E[D | covered] x P(covered) (P exact by inclusion-exclusion).
Evaluation rows: DEV = complement of L (Study 7); T1' = the stored query rows E \\ L
(Study 12; for n > 2000 a uniform subsample of the complement). q and p use full-table
counts on both sets, as in the Study 4 code.
Writes out/mech_tables.csv (per table x m x FM x set), out/mech_summary.csv, out/rv_mech.tex.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from itertools import combinations  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.special import gammaln  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rv_common import FMS, FMW, MW, OUT, Macros, fmt  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.runner.study9 import dataset_key  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

MULTS = (1, 3, 5, 10)


def log_comb(a, b):
    return -np.inf if b > a else float(gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1))


def p_miss(counts, B):
    n = int(counts.sum())
    return np.array([np.exp(log_comb(n - c, B) - log_comb(n, B)) for c in counts])


def p_full(counts, B):
    n = int(counts.sum())
    tot = 0.0
    for s in range(len(counts) + 1):
        for sub in combinations(range(len(counts)), s):
            tot += (-1) ** s * np.exp(log_comb(n - int(counts[list(sub)].sum()), B) - log_comb(n, B))
    return tot


def per_class_f1(y_eval, pred, C):
    cm = np.bincount(y_eval * C + pred, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)


def ids(set_name):
    if set_name == "dev":
        return [int(d["did"]) for d in yaml.safe_load((PROJECT_ROOT / "data" / "pool_i2.yaml").read_text())["dev"]]
    return [int(d["did"]) for d in yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())["datasets"]]


def path_of(set_name, did, fm):
    if set_name == "dev":
        return settings.results_dir / "study7" / "a" / f"{did}_{fm}.parquet"
    return settings.results_dir / "study12" / f"{dataset_key(did)}_{fm}.parquet"


def job(args):
    set_name, fm = args
    tabs, draws, chk = [], [], []
    for did in ids(set_name):
        y = load(did, settings.data_cache_dir).y.astype(np.int64)
        n = len(y)
        counts = np.bincount(y)
        C = len(counts)
        pi = counts / n
        cols = ["strategy", "mult", "labeled", "proba", "macro_f1", "accuracy", "coverage"]
        if set_name == "t1prime":
            cols.append("query")
        d = pd.read_parquet(path_of(set_name, did, fm), columns=cols)
        d = d[d.strategy.isin(["stratified", "random"])]
        for m in MULTS:
            g = d[d.mult == m]
            if g.empty:
                continue
            B = m * C
            st, rd = g[g.strategy == "stratified"], g[g.strategy == "random"]
            f1k = []
            for r in st.itertuples():
                L = np.asarray(r.labeled)
                ev = np.asarray(r.query) if set_name == "t1prime" else np.setdiff1d(np.arange(n), L)
                P = np.asarray(r.proba).reshape(-1, C)
                if set_name == "dev":
                    P = P[ev]
                pred = P.argmax(1)
                f = per_class_f1(y[ev], pred, C)
                f1k.append(f)
                chk.append(abs(f.mean() - r.macro_f1))
            f1k = np.mean(f1k, axis=0)
            q = p_miss(counts, B)
            mS, aS = st.macro_f1.mean(), st.accuracy.mean()
            miss = (rd.coverage < 1).to_numpy()
            if C == 2:
                other = (counts[::-1] - B) / (n - B)
                cf, ca, kind = float((q * (mS - other / (1 + other))).sum()), float((q * (aS - other)).sum()), "exact"
            else:
                cf, ca, kind = float((q * f1k).mean()), float((q * (counts / (n - B)) * f1k).sum()), "first"
            pf = p_full(counts, B)
            tabs.append(dict(set=set_name, fm=fm, dataset=did, C=C, mult=m, kind=kind, p_full=pf,
                             f1_gap=mS - rd.macro_f1.mean(), acc_gap=aS - rd.accuracy.mean(),
                             f1_gap_pred_first=float((q * f1k).mean()),
                             f1_miss_obs=float(((mS - rd.macro_f1) * miss).mean()),
                             acc_miss_obs=float(((aS - rd.accuracy) * miss).mean()),
                             f1_cov_pred=cf, acc_cov_pred=ca))
            for r in rd.itertuples():
                sh = np.bincount(y[np.asarray(r.labeled)], minlength=C) / len(r.labeled)
                draws.append(dict(set=set_name, fm=fm, dataset=did, mult=m, covered=bool((sh > 0).all()),
                                  D=float(((sh - 1 / C) ** 2).sum()), dF=float(r.macro_f1 - mS)))
    return pd.DataFrame(tabs), pd.DataFrame(draws), float(max(chk))


def add_balance(tabs, draws):
    out = []
    for (s, fm), t in tabs.groupby(["set", "fm"]):
        cov = draws[(draws.set == s) & (draws.fm == fm) & draws.covered]
        t = t.copy()
        kap, ed = [], []
        for r in t.itertuples():
            rest = cov[(cov.dataset != r.dataset) & (cov.mult == r.mult)]
            x, z = rest.D.to_numpy(), -rest.dF.to_numpy()
            kap.append(float((x * z).sum() / (x * x).sum()) if (x > 0).any() else 0.0)
            own = cov[(cov.dataset == r.dataset) & (cov.mult == r.mult)]
            ed.append(float(own.D.mean()) if len(own) else 0.0)
        t["kappa_f1"] = kap
        t["f1_bal_fit"] = np.array(kap) * np.array(ed) * t.p_full
        t["f1_bal_obs"] = t.f1_gap - t.f1_miss_obs
        t["acc_bal_obs"] = t.acc_gap - t.acc_miss_obs
        t["f1_total_fit"] = t.f1_cov_pred + t.f1_bal_fit
        out.append(t)
    return pd.concat(out, ignore_index=True)


def main():
    with Pool(4) as pool:
        res = pool.map(job, [(s, fm) for s in ("dev", "t1prime") for fm in FMS])
    tabs = pd.concat([r[0] for r in res], ignore_index=True)
    draws = pd.concat([r[1] for r in res], ignore_index=True)
    print("max |recomputed - stored macro-F1| of stratified fits:", max(r[2] for r in res))
    tabs = add_balance(tabs, draws)
    tabs.to_csv(OUT / "mech_tables.csv", index=False)
    cols = ["f1_gap", "f1_gap_pred_first", "f1_miss_obs", "f1_cov_pred", "f1_bal_obs", "f1_bal_fit",
            "acc_gap", "acc_miss_obs", "acc_cov_pred", "acc_bal_obs"]
    allm = tabs.groupby(["set", "fm", "mult"])[cols].mean().assign(kind="all", n=tabs.groupby(["set", "fm", "mult"]).size())
    byk = tabs.groupby(["set", "fm", "kind", "mult"])[cols].mean()
    byk["n"] = tabs.groupby(["set", "fm", "kind", "mult"]).size()
    byk = byk.reset_index()
    summ = pd.concat([allm.reset_index(), byk], ignore_index=True)
    corr = []
    for (s, fm, m), g in tabs.groupby(["set", "fm", "mult"]):
        corr.append(dict(set=s, fm=fm, mult=m, kind="corr",
                         r_first_vs_gap=np.corrcoef(g.f1_gap_pred_first, g.f1_gap)[0, 1],
                         r_cov=np.corrcoef(g.f1_cov_pred, g.f1_miss_obs)[0, 1],
                         r_bal=np.corrcoef(g.f1_bal_fit, g.f1_bal_obs)[0, 1] if m > 1 else np.nan))
    summ = pd.concat([summ, pd.DataFrame(corr)], ignore_index=True)
    summ.to_csv(OUT / "mech_summary.csv", index=False)
    pd.set_option("display.width", 250)
    show = summ[summ.mult.isin([1, 3])]
    print(show[show.kind.isin(["all", "exact"])][["set", "fm", "kind", "mult", "n", "f1_gap", "f1_gap_pred_first",
                                                  "f1_miss_obs", "f1_cov_pred", "f1_bal_obs", "f1_bal_fit",
                                                  "acc_miss_obs", "acc_cov_pred"]].round(4).to_string())
    print(summ[summ.kind == "corr"].round(3).to_string())
    macros(summ)


def macros(summ):
    mac = Macros("mech")
    SW = dict(dev="Dev", t1prime="Tp")
    prov = "re-analysis of pre-registered fits (Study 7 Part (a) DEV / Study 12 T1'), exploratory as analysis"
    for r in summ[summ.kind.isin(["all", "exact", "first"])].itertuples():
        if r.mult not in (1, 3):
            continue
        k = dict(all="All", exact="Bin", first="Multi")[r.kind]
        stem = f"Mech{SW[r.set]}{FMW[r.fm]}{k}{MW[r.mult]}"
        src = f"mech_summary.csv ({r.set}, {r.fm}, {r.kind} tables, m={r.mult}, n={r.n}; {prov})"
        mac.put(stem + "Gap", fmt(r.f1_gap, 3, sign=True), src + ", observed macro-F1 gap")
        mac.put(stem + "FCovObs", fmt(r.f1_miss_obs, 3), src + ", observed coverage part, macro-F1")
        mac.put(stem + "FCovPred", fmt(r.f1_cov_pred, 3), src + ", predicted coverage part (Prop. 1), macro-F1")
        mac.put(stem + "AccCovObs", fmt(r.acc_miss_obs, 4), src + ", observed coverage part, accuracy")
        mac.put(stem + "AccCovPred", fmt(r.acc_cov_pred, 4), src + ", predicted coverage part, accuracy")
        mac.put(stem + "FirstPred", fmt(r.f1_gap_pred_first, 3), src + ", first-order coverage gap (Fig. 4a x-axis)")
        if r.mult == 3:
            mac.put(stem + "FBalObs", fmt(r.f1_bal_obs, 3, sign=True), src + ", observed balance part")
            mac.put(stem + "FBalFit", fmt(r.f1_bal_fit, 3, sign=True), src + ", FITTED (leave-one-table-out) balance part")
    # ranges over the four FMs, for text
    for s in ("dev", "t1prime"):
        for kind, m, col, nm, nd in [("exact", 1, "f1_cov_pred", "BinFCovPred", 3), ("exact", 1, "f1_miss_obs", "BinFCovObs", 3),
                                     ("exact", 1, "acc_cov_pred", "BinAccCovPred", 4), ("exact", 1, "acc_miss_obs", "BinAccCovObs", 4),
                                     ("all", 3, "f1_bal_obs", "FBalObsMThree", 3), ("all", 3, "f1_bal_fit", "FBalFitMThree", 3),
                                     ("all", 3, "f1_miss_obs", "FCovObsMThree", 3), ("all", 3, "f1_gap", "GapMThree", 3)]:
            v = summ[(summ.set == s) & (summ.kind == kind) & (summ.mult == m)][col]
            sg = col.startswith("f1_bal") or col == "f1_gap"
            mac.put(f"Mech{SW[s]}{nm}Min", fmt(v.min(), nd, sign=sg), f"mech_summary.csv ({s}, {kind}, m={m}, {col}, min over 4 FMs; {prov})")
            mac.put(f"Mech{SW[s]}{nm}Max", fmt(v.max(), nd, sign=sg), f"mech_summary.csv ({s}, {kind}, m={m}, {col}, max over 4 FMs; {prov})")
        c = summ[(summ.set == s) & (summ.kind == "corr") & (summ.mult == 1)].r_first_vs_gap
        mac.put(f"Mech{SW[s]}RFirstMOneMin", fmt(c.min(), 2), f"mech_summary.csv ({s}, m=1, Pearson r first-order pred vs gap, min over FMs)")
        mac.put(f"Mech{SW[s]}RFirstMOneMax", fmt(c.max(), 2), f"mech_summary.csv ({s}, m=1, Pearson r first-order pred vs gap, max over FMs)")
        rb = summ[(summ.set == s) & (summ.kind == "corr") & (summ.mult == 3)].r_bal
        mac.put(f"Mech{SW[s]}RBalMThreeMin", fmt(rb.min(), 2, sign=True), f"mech_summary.csv ({s}, m=3, Pearson r fitted vs observed balance part over tables, min over FMs)")
        mac.put(f"Mech{SW[s]}RBalMThreeMax", fmt(rb.max(), 2, sign=True), f"mech_summary.csv ({s}, m=3, Pearson r fitted vs observed balance part over tables, max over FMs)")
        nb = summ[(summ.set == s) & (summ.kind == "exact") & (summ.mult == 1)].n.iloc[0]
        mac.put(f"Mech{SW[s]}BinN", str(int(nb)), f"mech_summary.csv ({s}, binary tables)")
    mac.write()


if __name__ == "__main__":
    main()
