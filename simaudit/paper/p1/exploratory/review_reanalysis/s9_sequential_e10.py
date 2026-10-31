"""(b) sequential-acquisition emulation from Study 9 nested random draws, and
(d4) a replay check of the Study 9 miss-rate deviation (E10). EXPLORATORY, no fits.

Sequential policy: label uniformly at random in batches of C rows (the nested Study 9
order) and stop at the first grid budget m' >= m whose set contains every class.
Score = stored score at m'; labels = m' C. Also the exact one-at-a-time stopping time
T (labels until every class appears) from the order re-derived from sel_seed.

E10 replay: the Study 9 selection pipeline (row_permutation -> layout -> selection_seed
-> default_rng(seed).permutation(pool)) re-run with 1,000 other base seeds on the 35
binary T1' tables (10 reps each, m = 1); z of the miss count against the exact
hypergeometric expectation per replay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from rv_common import FMS, FMW, MW, OUT, Macros, fmt, holm, pfmt
from simaudit.analysis.report import bootstrap_ci
from simaudit.data.openml_loader import load
from simaudit.runner.study9 import GRID, dataset_key, layout, row_permutation, selection_seed
from simaudit.settings import PROJECT_ROOT, settings

S9 = settings.results_dir / "study9"
BASE = settings.base_seed
TARGET = (1, 3, 5, 10)


def ids(set_name):
    if set_name == "t1prime":
        return [int(d["did"]) for d in yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())["datasets"]]
    return [int(d["did"]) for d in yaml.safe_load((PROJECT_ROOT / "data" / "pool_i2.yaml").read_text())["dev"]]


def first_cover_time(order_labels: np.ndarray, C: int) -> int:
    seen = np.zeros(C, bool)
    k = 0
    for t, c in enumerate(order_labels, 1):
        if not seen[c]:
            seen[c] = True
            k += 1
            if k == C:
                return t
    return -1


def sequential(set_name: str):
    rows, trows, checks = [], [], [0, 0]
    for did in ids(set_name):
        key = dataset_key(did)
        y = load(did, settings.data_cache_dir).y.astype(np.int64)
        n, C = len(y), int(y.max()) + 1
        perm = row_permutation(n, did, BASE)
        yp = y[perm]
        s, _ = layout(n)
        pool = np.arange(s, n, dtype=np.int64)
        for fm in FMS:
            p = S9 / set_name / f"{key}_{fm}.parquet"
            if not p.exists():
                continue
            d = pd.read_parquet(p, columns=["strategy", "m", "rep", "coverage", "macro_f1", "accuracy", "labeled", "sel_seed"])
            rnd = d[d.strategy == "random"]
            st = d[d.strategy == "stratified"].groupby("m")[["macro_f1", "accuracy"]].mean()
            grid = sorted(rnd.m.unique())
            if fm == "fm2" or (set_name == "t1prime" and fm == "fm1"):
                pass
            for rep, g in rnd.groupby("rep"):
                g = g.set_index("m").sort_index()
                if fm == FMS[0] or set_name == "dev":
                    order = np.random.default_rng(int(g.sel_seed.iloc[0])).permutation(pool)
                    top = g.index.max()
                    checks[0] += int(np.array_equal(np.sort(perm[order[: top * C]]), np.sort(np.asarray(g.loc[top, "labeled"]))))
                    checks[1] += 1
                    if fm == FMS[0] or (set_name == "dev" and fm == "fm2"):
                        trows.append(dict(set=set_name, dataset=key, C=C, rep=rep,
                                          T=first_cover_time(yp[order], C)))
                for m in TARGET:
                    if m not in g.index or m not in st.index:
                        continue
                    cands = [mm for mm in grid if mm >= m and g.loc[mm, "coverage"] >= 1.0]
                    if not cands:
                        rows.append(dict(set=set_name, dataset=key, fm=fm, m=m, rep=rep, feasible=False))
                        continue
                    mm = cands[0]
                    rows.append(dict(set=set_name, dataset=key, fm=fm, m=m, rep=rep, feasible=True, m_used=mm,
                                     seq_f1=g.loc[mm, "macro_f1"], seq_acc=g.loc[mm, "accuracy"],
                                     rnd_f1=g.loc[m, "macro_f1"], rnd_acc=g.loc[m, "accuracy"],
                                     st_f1=st.loc[m, "macro_f1"], st_acc=st.loc[m, "accuracy"]))
    return pd.DataFrame(rows), pd.DataFrame(trows), checks


def summarize_seq(runs: pd.DataFrame) -> pd.DataFrame:
    f = runs[runs.feasible]
    per = f.groupby(["set", "fm", "m", "dataset"]).agg(
        seq_f1=("seq_f1", "mean"), seq_acc=("seq_acc", "mean"), st_f1=("st_f1", "first"), st_acc=("st_acc", "first"),
        rnd_f1=("rnd_f1", "mean"), rnd_acc=("rnd_acc", "mean"), ratio=("m_used", "mean")).reset_index()
    per["ratio"] = per.ratio / per.m
    out = []
    for (s, fm, m), g in per.groupby(["set", "fm", "m"]):
        r = dict(set=s, fm=fm, m=m, n=len(g), labels_ratio=g.ratio.mean(),
                 infeasible_reps=int((~runs[(runs.set == s) & (runs.fm == fm) & (runs.m == m)].feasible).sum()))
        for met in ("f1", "acc"):
            x = (g[f"st_{met}"] - g[f"seq_{met}"]).to_numpy()
            lo, hi = bootstrap_ci(x)
            r.update({f"st_seq_{met}": x.mean(), f"st_seq_{met}_lo": lo, f"st_seq_{met}_hi": hi,
                      f"st_seq_{met}_p": float(stats.wilcoxon(x).pvalue) if np.any(x != 0) else 1.0})
            x2 = (g[f"st_{met}"] - g[f"rnd_{met}"]).to_numpy()
            r[f"st_rnd_{met}"] = x2.mean()
        out.append(r)
    return pd.DataFrame(out)


def p_miss_binary(c0: int, c1: int) -> float:
    N = c0 + c1
    return (c0 * (c0 - 1) + c1 * (c1 - 1)) / (N * (N - 1))


def e10_replay(n_rep: int = 1000):
    tabs = []
    for did in ids("t1prime"):
        y = load(did, settings.data_cache_dir).y.astype(np.int64)
        if y.max() == 1:
            tabs.append((did, y))
    zs, obs_rates = [], []
    for b in range(n_rep):
        base = 10_000_000 + b
        obs = exp = var = 0.0
        for did, y in tabs:
            n = len(y)
            yp = y[row_permutation(n, did, base)]
            s, _ = layout(n)
            pool = np.arange(s, n, dtype=np.int64)
            c = np.bincount(yp[s:], minlength=2)
            pm = p_miss_binary(int(c[0]), int(c[1]))
            for rep in range(10):
                o = np.random.default_rng(selection_seed(base, did, rep, "random")).permutation(pool)
                obs += yp[o[0]] == yp[o[1]]
                exp += pm
                var += pm * (1 - pm)
        zs.append((obs - exp) / np.sqrt(var))
        obs_rates.append(obs / (10 * len(tabs)))
    return len(tabs), np.array(zs)


def stored_dependence():
    """Stored m=1 binary misses: per-table dispersion and rep-index effect."""
    k = pd.read_csv(OUT.parents[1] / "verify" / "out" / "claim_k_per_table.csv")
    k = k[k.C == 2]
    # dispersion: sum (obs - n p)^2 / (n p (1-p)) ~ chi2(df = tables)
    chi = float((((k.obs_miss - k.n_draws * k.p_exact) ** 2) / (k.n_draws * k.p_exact * (1 - k.p_exact))).sum())
    return dict(n_tables=len(k), disp_chi2=chi, disp_df=len(k), disp_p=float(stats.chi2.sf(chi, len(k))))


def main() -> None:
    mac = Macros("s9")
    # --- sequential emulation
    allr, allt, chk = [], [], [0, 0]
    for s in ("t1prime", "dev"):
        r, t, c = sequential(s)
        allr.append(r); allt.append(t); chk = [chk[0] + c[0], chk[1] + c[1]]
    runs, times = pd.concat(allr), pd.concat(allt)
    runs.to_csv(OUT / "s9_sequential_runs.csv", index=False)
    times.to_csv(OUT / "s9_cover_times.csv", index=False)
    seq = summarize_seq(runs)
    seq.to_csv(OUT / "s9_sequential.csv", index=False)
    pd.set_option("display.width", 250)
    print("orders re-derived exactly:", chk)
    print(seq.round(4).to_string())
    times["ratio"] = times["T"] / times["C"]
    print(times.groupby("set").ratio.describe())
    mac.put("SeqOrdersOk", f"{chk[0]} of {chk[1]}", "s9_sequential_e10.py (stored nested random sets re-derived from sel_seed)")
    for r in seq.itertuples():
        stem = f"Seq{dict(dev='Dev', t1prime='Tp')[r.set]}{FMW[r.fm]}{MW[r.m]}"
        src = f"s9_sequential.csv ({r.set}, {r.fm}, m={r.m}; Study 9 nested random draws, exploratory)"
        mac.put(stem + "F", fmt(r.st_seq_f1, 3, sign=True), src + ", stratified - sequential macro-F1")
        mac.put(stem + "FLo", fmt(r.st_seq_f1_lo, 3, sign=True), src + ", bootstrap 95% CI")
        mac.put(stem + "FHi", fmt(r.st_seq_f1_hi, 3, sign=True), src + ", bootstrap 95% CI")
        mac.put(stem + "FP", pfmt(r.st_seq_f1_p), src + ", Wilcoxon, unadjusted")
        mac.put(stem + "Acc", fmt(r.st_seq_acc, 3, sign=True), src + ", stratified - sequential accuracy")
        mac.put(stem + "Labels", fmt(r.labels_ratio, 2), src + ", mean labels used / (m C)")
    for s, g in times.groupby("set"):
        w = dict(dev="Dev", t1prime="Tp")[s]
        mac.put(f"SeqCoverT{w}Median", fmt(float(g.ratio.median()), 1), f"s9_cover_times.csv ({s}): median T/C, one-at-a-time")
        mac.put(f"SeqCoverT{w}Mean", fmt(float(g.ratio.mean()), 1), f"s9_cover_times.csv ({s}): mean T/C")
        mac.put(f"SeqCoverT{w}Pninety", fmt(float(g.ratio.quantile(0.9)), 1), f"s9_cover_times.csv ({s}): 90th pct T/C")
    # --- E10 replay
    ntab, z = e10_replay()
    pd.DataFrame(dict(z=z)).to_csv(OUT / "e10_replay_z.csv", index=False)
    tail = float(np.mean(np.abs(z) >= 3.25))
    low = float(np.mean(z <= -3.25))
    ks = stats.kstest(z, "norm")
    dep = stored_dependence()
    print("E10 replay: tables", ntab, "mean z", z.mean(), "sd", z.std(ddof=1), "share |z|>=3.25", tail, "z<=-3.25", low, "KS p", ks.pvalue)
    print("stored dispersion", dep)
    src = "e10_replay_z.csv (1,000 replays of the Study 9 selection pipeline, other base seeds, 35 binary T1' tables, m=1)"
    mac.put("EtenReplayN", "1{,}000", src)
    mac.put("EtenReplayTables", str(ntab), src)
    mac.put("EtenReplayZMean", fmt(float(z.mean()), 2, sign=True), src + ", mean z")
    mac.put("EtenReplayZSd", fmt(float(z.std(ddof=1)), 2), src + ", SD of z")
    mac.put("EtenReplayTail", fmt(tail, 3), src + ", share |z| >= 3.25")
    mac.put("EtenReplayKsP", f"{ks.pvalue:.2f}", src + ", KS test against N(0,1)")
    mac.put("EtenDispP", f"{dep['disp_p']:.2f}", "claim_k_per_table.csv: per-table dispersion chi2 vs binomial")
    pd.DataFrame([dict(n_tables=ntab, z_mean=z.mean(), z_sd=z.std(ddof=1), tail_abs_325=tail, tail_low_325=low,
                       ks_p=ks.pvalue, **{f"stored_{k}": v for k, v in dep.items()})]).to_csv(OUT / "e10_replay_summary.csv", index=False)
    mac.write()


if __name__ == "__main__":
    main()
