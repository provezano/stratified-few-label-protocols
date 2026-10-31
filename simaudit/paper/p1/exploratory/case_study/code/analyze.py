"""Analysis of the TabEBM case study (PREREG Sec. 5). Run in the TabEBM venv with -I.

Writes out/cells.csv, out/ranks.csv, out/summary.json, out/case_study_numbers.tex.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cs_common as cc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

B = 10_000
METRICS = ("ba", "macro_f1", "acc")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, meta = [], []
    for f in sorted((cc.OUT / "stage_a").glob("*.json")):
        r = json.loads(f.read_text())
        key = dict(did=r["did"], n_real=r["n_real"], protocol=r["protocol"], rep=r["rep"])
        meta.append(dict(**key, status=r["status"], C=r["C"], missed_train=r.get("missed_train"),
                         n_seen_subset=r.get("n_seen_subset"), n_seen_train=r.get("n_seen_train"),
                         n_train=r.get("n_train"), smote_fallback=r.get("smote_fallback_classes"),
                         t_tabebm=r.get("t_tabebm"), tabebm_finite=r.get("tabebm_finite")))
        for x in r["rows"]:
            rows.append(dict(**key, **x))
        bf = cc.OUT / "stage_b" / f.name
        if bf.exists():
            for x in json.loads(bf.read_text())["rows"]:
                rows.append(dict(**key, **x))
    return pd.DataFrame(rows), pd.DataFrame(meta)


def per_rep_gain(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """D_{c,P,r}: mean over the cell's predictor set of tabebm - baseline."""
    w = df.pivot_table(index=["did", "n_real", "protocol", "rep", "predictor"], columns="arm", values=metric).reset_index()
    w["d"] = w["tabebm"] - w["baseline"]
    npred = w.groupby(["did", "n_real", "protocol", "rep"])["predictor"].transform("nunique")
    w["expected"] = [6 if cc.PAPER_SHAPE[d][2] <= cc.TABPFN_MAX_C else 5 for d in w["did"]]
    w = w[npred == w["expected"]]
    return w.groupby(["did", "n_real", "protocol", "rep"])["d"].mean().reset_index()


def boot_mean(x: np.ndarray, rng) -> tuple[float, float]:
    x = np.asarray(x, float)
    bs = x[rng.integers(0, len(x), size=(B, len(x)))].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def analyse(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict]:
    g = per_rep_gain(df, metric)
    rng = np.random.default_rng(cc.BASE_SEED)
    cells, boots = [], {"S": [], "R": [], "G": []}
    for did, n in cc.CELLS:
        row = dict(did=did, table=cc.NAMES[did], n_real=n)
        for p in cc.PROTOCOLS:
            x = g[(g.did == did) & (g.n_real == n) & (g.protocol == p)].set_index("rep")["d"]
            row[f"n_{p}"] = len(x)
            if len(x):
                row[f"delta_{p}"] = float(x.mean())
                row[f"lo_{p}"], row[f"hi_{p}"] = boot_mean(x.to_numpy(), rng)
                boots[p].append(x.to_numpy()[rng.integers(0, len(x), size=(B, len(x)))].mean(1))
        xs = g[(g.did == did) & (g.n_real == n) & (g.protocol == "S")].set_index("rep")["d"]
        xr = g[(g.did == did) & (g.n_real == n) & (g.protocol == "R")].set_index("rep")["d"]
        common = xs.index.intersection(xr.index)
        if len(common):
            diff = (xs.loc[common] - xr.loc[common]).to_numpy()
            row["G"] = float(diff.mean())
            row["G_lo"], row["G_hi"] = boot_mean(diff, rng)
            boots["G"].append(diff[rng.integers(0, len(diff), size=(B, len(diff)))].mean(1))
        cells.append(row)
    cdf = pd.DataFrame(cells)
    summ = {}
    for k, bl in boots.items():
        if bl:
            m = np.mean(bl, axis=0)
            col = {"S": "delta_S", "R": "delta_R", "G": "G"}[k]
            summ[f"mean_{k}"] = float(cdf[col].mean())
            summ[f"lo_{k}"], summ[f"hi_{k}"] = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
            summ[f"lo90_{k}"], summ[f"hi90_{k}"] = float(np.percentile(m, 5)), float(np.percentile(m, 95))
    return cdf, summ


def verdicts(cdf: pd.DataFrame, s: dict) -> dict:
    n = int(cdf["delta_S"].notna().sum())
    k_s = int((cdf["delta_S"] > 0).sum())
    k_r = int((cdf["delta_R"] > 0).sum())
    gate = k_s >= math.ceil(8 / 9 * n) and s["lo_S"] > 0
    any_below = bool((cdf["hi_R"] < 0).any())
    holds = k_r >= math.ceil(8 / 9 * n) and s["lo_R"] > 0
    changes_core = k_r <= math.floor(6 / 9 * n) or s["lo_R"] <= 0 or any_below
    attributed = s["lo_G"] > 0
    if not gate:
        v = "not reproduced under S (descriptive only)"
    elif holds:
        v = "Holds"
    elif changes_core and attributed:
        v = "Changes"
    else:
        v = "Weakened"
    tost = (s["lo90_G"] > -0.01) and (s["hi90_G"] < 0.01)
    return dict(n_cells=n, k_S=k_s, k_R=k_r, gate=bool(gate), any_cell_R_below=any_below,
                attributed=bool(attributed), verdict=v, G_equivalent_0=bool(tost))


def cell_label(r) -> str:
    if r["lo_R"] > 0:
        return "gain kept"
    if r["hi_R"] < 0:
        return "reversed"
    return "gain not detected"


def ranks(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out, fr = [], {}
    for p in cc.PROTOCOLS:
        m = df[df.protocol == p].groupby(["did", "n_real", "predictor", "arm"])["ba"].mean().unstack("arm").dropna()
        rk = m.rank(axis=1, ascending=False)
        out.append(dict(protocol=p, **{a: float(rk[a].mean()) for a in cc.ARMS}, blocks=len(m)))
        fr[p] = float(stats.friedmanchisquare(*[m[a] for a in cc.ARMS]).pvalue)
    return pd.DataFrame(out), fr


def miss_pred(did: int, n_real: int, draws: int = 20000) -> float:
    """P(an N_real uniform subset of a pool with the table's class shares misses a class)."""
    _, y = cc.load_table(did)
    n_test = int(min(len(y) // 2, 500))
    cnt = np.bincount(y)
    pool = np.round(cnt * (len(y) - n_test) / len(y)).astype(int)
    lab = np.repeat(np.arange(len(cnt)), pool)
    rng = np.random.default_rng(cc.BASE_SEED)
    miss = 0
    for _ in range(draws):
        miss += len(np.unique(rng.choice(lab, n_real, replace=False))) < len(cnt)
    return miss / draws


def fmt(x, sign=True, d=3):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("+", "$+$").replace("-", "$-$") if sign else s


def main():
    df, meta = load()
    meta.to_csv(cc.OUT / "runs_meta.csv", index=False)
    res = {}
    allc = []
    for metric in METRICS:
        cdf, s = analyse(df, metric)
        v = verdicts(cdf, s)
        cdf["metric"] = metric
        cdf["label_R"] = cdf.apply(cell_label, axis=1)
        allc.append(cdf)
        res[metric] = dict(summary=s, verdict=v)
    cells = pd.concat(allc)
    ok = meta[meta.status == "ok"]
    mr = ok[ok.protocol == "R"].groupby(["did", "n_real"]).apply(
        lambda d: pd.Series(dict(miss_subset=float((d.n_seen_subset < d.C).mean()), miss_train=float(d.missed_train.mean()))))
    ms = ok[ok.protocol == "S"].groupby(["did", "n_real"]).apply(
        lambda d: pd.Series(dict(miss_subset_S=float((d.n_seen_subset < d.C).mean()), miss_train_S=float(d.missed_train.mean()))))
    cov = mr.join(ms).reset_index()
    cov["miss_pred"] = [miss_pred(int(a), int(b)) for a, b in zip(cov.did, cov.n_real)]
    infeas = meta[meta.status != "ok"].groupby(["did", "n_real", "protocol"]).size().rename("n_infeasible").reset_index()
    cells = cells.merge(cov, on=["did", "n_real"], how="left")
    cells.to_csv(cc.OUT / "cells.csv", index=False)
    rk, fr = ranks(df)
    rk.to_csv(cc.OUT / "ranks.csv", index=False)
    res["ranks"] = dict(table=rk.to_dict("records"), friedman_p=fr)
    win = {r["protocol"]: min(cc.ARMS, key=lambda a: r[a]) for r in rk.to_dict("records")}
    res["ranks"]["winner"] = win
    res["ranks"]["winner_changes"] = bool(win.get("S") == "tabebm" and win.get("R") != "tabebm")
    res["infeasible"] = infeas.to_dict("records")
    res["runs"] = dict(n_ok=int((meta.status == "ok").sum()), n_total=len(meta),
                       tabebm_cpu_hours=float(meta.t_tabebm.fillna(0).sum() / 3600),
                       nonfinite_tabebm=int((meta.tabebm_finite == False).sum()))  # noqa: E712
    (cc.OUT / "summary.json").write_text(json.dumps(res, indent=2, default=float))

    # ---- macros (EXPLORATORY)
    src = "paper/p1/exploratory/case_study/out/summary.json, cells.csv"
    mac = {}
    tag = {"ba": "BA", "macro_f1": "F", "acc": "Acc"}
    for metric, t in tag.items():
        s, v = res[metric]["summary"], res[metric]["verdict"]
        for k in ("S", "R", "G"):
            mac[f"ExCaseEBM{t}Mean{k}"] = fmt(s[f"mean_{k}"])
            mac[f"ExCaseEBM{t}Mean{k}Lo"] = fmt(s[f"lo_{k}"])
            mac[f"ExCaseEBM{t}Mean{k}Hi"] = fmt(s[f"hi_{k}"])
        mac[f"ExCaseEBM{t}KS"] = str(v["k_S"])
        mac[f"ExCaseEBM{t}KR"] = str(v["k_R"])
        mac[f"ExCaseEBM{t}Verdict"] = v["verdict"]
    mac["ExCaseEBMNCells"] = str(res["ba"]["verdict"]["n_cells"])
    mac["ExCaseEBMGate"] = "passed" if res["ba"]["verdict"]["gate"] else "failed"
    cb = cells[cells.metric == "ba"]
    mac["ExCaseEBMNKeptR"] = str(int((cb.label_R == "gain kept").sum()))
    mac["ExCaseEBMNReversedR"] = str(int((cb.label_R == "reversed").sum()))
    mac["ExCaseEBMMissRMax"] = f"{cb.miss_train.max():.2f}"
    mac["ExCaseEBMMissRMin"] = f"{cb.miss_train.min():.2f}"
    for p in cc.PROTOCOLS:
        r = rk[rk.protocol == p].iloc[0]
        for a in cc.ARMS:
            mac[f"ExCaseEBMRank{p}{a.capitalize()}"] = f"{r[a]:.2f}"
    mac["ExCaseEBMWinnerS"] = win["S"]
    mac["ExCaseEBMWinnerR"] = win["R"]
    # Descriptive, NOT pre-declared in Sec. 5: absolute level S minus R per arm (mean over cells x predictors)
    for metric, t in (("ba", "BA"), ("macro_f1", "F")):
        lv = df.groupby(["did", "n_real", "protocol", "arm"])[metric].mean().unstack("protocol")
        lv = (lv["S"] - lv["R"]).groupby("arm").mean()
        res.setdefault("level_S_minus_R", {})[metric] = lv.to_dict()
        for arm in cc.ARMS:
            mac[f"ExCaseEBMLevel{t}{arm.capitalize()}"] = fmt(float(lv[arm]))
    (cc.OUT / "summary.json").write_text(json.dumps(res, indent=2, default=float))
    mac["ExCaseEBMReps"] = str(int(meta.rep.max()) + 1)
    mac["ExCaseEBMCPUHours"] = f"{res['runs']['tabebm_cpu_hours']:.0f}"
    lines = ["% AUTO-GENERATED by paper/p1/exploratory/case_study/code/analyze.py. EXPLORATORY "
             "(TabEBM case study, PREREG.md frozen R002; not part of any confirmatory family). Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}} % EXPLORATORY; source: {src}" for k, v in sorted(mac.items())]
    (cc.OUT / "case_study_numbers.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({m: res[m]["verdict"] for m in METRICS}, indent=1))
    print(cb[["table", "n_real", "delta_S", "lo_S", "hi_S", "delta_R", "lo_R", "hi_R", "G", "G_lo", "G_hi", "label_R", "miss_train", "miss_pred"]].round(3).to_string())
    print(rk.round(2).to_string(), fr)


if __name__ == "__main__":
    main()
