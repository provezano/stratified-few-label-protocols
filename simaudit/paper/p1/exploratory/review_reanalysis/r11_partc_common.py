"""Round-11 review check M11-1 (EXPLORATORY re-analysis of stored pre-registered fits; no new fits).

Study 7 Part (c) scores each prescribed composition on its own complement U = pool minus L. At fixed
budget the class shares of U move with the share s, so even a fixed uniform-prior rule changes its
macro-F1 with s. Sensitivity: rescore the stored probabilities (raw p and corrected p/pi_hat) on the
rows COMMON to all compositions of one (table, arm, budget, repetition), i.e. outside every context.
Labels from data/cache/npz; the complement scores reproduce the stored macro_f1 / pc_macro_f1
(checked, max |diff| printed).
Outputs per FM and arm, for both scorings:
- the pre-registered rule's drop (s = 0.5 to the largest feasible s, per dataset x mult, mean), raw
  and corrected, and their ratio (as r4_checks.py (2); threshold 0.5);
- the mean per-table slope of macro-F1 on s (least squares over shares, mean over mult).
Writes out/r11_partc_common.parquet (per fit), out/r11_partc.csv (per FM x arm), out/rv_r11.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r11_partc_common.py
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from rv_common import FMS, FMW, OUT, Macros, fmt
from simaudit.settings import PROJECT_ROOT, settings

COLS = ["learner", "dataset", "arm", "share", "mult", "rep", "feasible", "counts", "labeled", "proba",
        "macro_f1", "pc_macro_f1", "n_classes"]


def mf(y, pred, mask, C):
    return f1_score(y[mask], pred[mask], labels=list(range(C)), average="macro", zero_division=0)


def one_file(path: str) -> tuple[list[dict], float]:
    d = pd.read_parquet(path, columns=COLS)
    d = d[d.feasible]
    if d.empty:
        return [], 0.0
    did = int(d.dataset.iloc[0])
    y = np.load(PROJECT_ROOT / "data" / "cache" / "npz" / f"{did}.npz", allow_pickle=False)["y"]
    n, C = len(y), int(d.n_classes.iloc[0])
    rows, err = [], 0.0
    for (arm, mult, rep), g in d.groupby(["arm", "mult", "rep"]):
        union = np.zeros(n, bool)
        for lab in g.labeled:
            union[np.asarray(lab, int)] = True
        common = ~union
        for r in g.itertuples():
            P = np.asarray(r.proba)
            P = P.reshape(n, C) if P.ndim == 1 else np.vstack(P)
            comp = np.ones(n, bool)
            comp[np.asarray(r.labeled, int)] = False
            cnt = np.asarray(r.counts, float)
            pih = cnt / cnt.sum()
            Q = P / np.where(pih > 0, pih, np.inf)
            Q = Q / Q.sum(1, keepdims=True)
            pr, pq = P.argmax(1), Q.argmax(1)
            rc, qc = mf(y, pr, comp, C), mf(y, pq, comp, C)
            err = max(err, abs(rc - r.macro_f1), abs(qc - r.pc_macro_f1))
            rows.append(dict(fm=r.learner, dataset=did, arm=arm, mult=mult, rep=rep, share=r.share,
                             raw_comp=rc, pc_comp=qc, raw_com=mf(y, pr, common, C), pc_com=mf(y, pq, common, C),
                             n_common=int(common.sum())))
    return rows, err


def drop(g: pd.DataFrame, col: str) -> float:
    per = g.groupby(["dataset", "mult", "share"])[col].mean().unstack()
    d = per.apply(lambda r: r.dropna().iloc[0] - r.dropna().iloc[-1] if r.notna().sum() >= 2 else np.nan, axis=1)
    return float(d.dropna().mean())


def slope(g: pd.DataFrame, col: str) -> float:
    per = []
    for _, gd in g.groupby("dataset"):
        s = []
        for _, gm in gd.groupby("mult"):
            m = gm.groupby("share")[col].mean()
            if len(m) > 1:
                s.append(np.polyfit(m.index.to_numpy(float), m.to_numpy(), 1)[0])
        if s:
            per.append(np.mean(s))
    return float(np.mean(per))


def main() -> None:
    files = [str(p) for fm in FMS for p in sorted((settings.results_dir / "study7" / "c").glob(f"*_{fm}.parquet"))]
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 1)) as ex:
        res = list(ex.map(one_file, files))
    t = pd.DataFrame([r for rows, _ in res for r in rows])
    err = max(e for _, e in res)
    assert err < 1e-9, err
    t.to_parquet(OUT / "r11_partc_common.parquet", index=False)
    out = []
    for (fm, arm), g in t.groupby(["fm", "arm"]):
        rec = dict(fm=fm, arm=arm, n_common_median=float(g.n_common.median()))
        for sc in ("comp", "com"):
            rec[f"drop_{sc}"], rec[f"pcdrop_{sc}"] = drop(g, f"raw_{sc}"), drop(g, f"pc_{sc}")
            rec[f"ratio_{sc}"] = rec[f"pcdrop_{sc}"] / rec[f"drop_{sc}"]
            rec[f"slope_{sc}"], rec[f"pcslope_{sc}"] = slope(g, f"raw_{sc}"), slope(g, f"pc_{sc}")
        out.append(rec)
    o = pd.DataFrame(out)
    o = o[o.arm.isin(["majority", "minority"])].reset_index(drop=True)  # the anchor arm has a single share
    o.to_csv(OUT / "r11_partc.csv", index=False)
    pd.set_option("display.width", 220)
    print("max |complement - stored|:", err)
    print(o.round(3).to_string(index=False))

    m = Macros("r11")
    src = "r11_partc_common.py (M11-1; Study 7 Part (c), stored probabilities, rows common to all compositions)"
    mi = o[o.arm == "minority"].set_index("fm")
    red = 1 - mi.pcdrop_com / mi.pcdrop_comp
    m.put("ReleveMinPcDropRedMin", f"{100 * red.min():.0f}\\%", src + ": reduction of the corrected minority-arm drop, min over FMs")
    m.put("ReleveMinPcDropRedMax", f"{100 * red.max():.0f}\\%", src + ": reduction of the corrected minority-arm drop, max over FMs")
    m.put("ReleveNegPcSlopes", str(int((o.pcslope_com < 0).sum())), src + ": FM x arm cells with a negative mean corrected slope on common rows")
    m.put("ReleveCells", str(len(o)), src + ": FM x arm cells")
    for fm in FMS:
        for arm in ("majority", "minority"):
            r = o[(o.fm == fm) & (o.arm == arm)].iloc[0]
            w = f"{FMW[fm]}{arm.capitalize()}"
            m.put(f"Releve{w}RatioComp", fmt(r.ratio_comp, 2), src + f": {fm} {arm} corrected/raw drop, complement")
            m.put(f"Releve{w}RatioCom", fmt(r.ratio_com, 2), src + f": {fm} {arm} corrected/raw drop, common rows")
    # per-FM verdict on common rows by the rule's ratio part ("beyond" needs ratio >= 0.5 and a positive raw drop in both arms)
    vs = []
    for fm in FMS:
        g = o[o.fm == fm]
        vs.append("beyond" if ((g.ratio_com >= 0.5) & (g.drop_com > 0)).all() else "unresolved")
    m.put("ReleveNBeyondCom", str(vs.count("beyond")), src + ": FMs 'beyond' on common rows (ratio >= 0.5 in both arms)")
    m.put("ReleveNCommonMedian", f"{t.n_common.median():.0f}", src + ": median number of common rows per (table, arm, budget, rep)")
    m.write()
    print(dict(zip(FMS, vs)))


if __name__ == "__main__":
    main()
