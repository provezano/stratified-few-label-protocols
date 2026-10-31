"""Round-7 review checks (EXPLORATORY; read-only inputs).

(1) m7-5: T1' accuracy on the 18 imbalanced binary tables (TV >= 0.1, post hoc), at three and five
    labels per class, split by whether B = 2m reaches B* of Prop. 2(ii) (as in r6_checks.py), but
    against the random draws that cover both classes only: per table and FM, mean stratified accuracy
    minus mean accuracy of the covered random draws. Also the share of random draws that miss a class
    on the tables beyond B*, and the absolute values of the r6 decline (to avoid "falls by -x").
(2) m7-4: the stored labelled sets of Study 7, Part (a), on DEV by kind: stratified or random against
    label-free selectors (fm1 files, one row per labelled set).
(3) C7-1 (THEORY, no data): the counterexample to Prop. 2(iii) under overlap alone. Class 2 has
    l ~ w U[-3,-1] + (1-w) U[1,3], class 1 has density e^t f2(t); the classes overlap everywhere, but l
    has no mass in (-1, 1). For pi_1 = 0.30 every other condition holds, and macro-F1 is maximal and
    flat on [-1, 1], which contains the uniform prior (tau = 0).
Writes out/r7_accbin_cov.csv, out/rv_r7.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r7_checks.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from rv_common import FMS, OUT, Macros, fmt
import bc_contrasts as bc

sys.path.insert(0, str(OUT.parents[1]))
import common as cm  # noqa: E402


def accbin_cov():
    runs = bc.load_t1()
    tv = pd.read_csv(OUT / "bc_t1_pool_tv.csv").set_index("dataset")
    b = tv[(tv.C == 2) & (tv.tv >= 0.1)].copy()
    b["bstar"] = 1 / (b.pi_min * (1 - b.pi_min) * np.log((1 - b.pi_min) / b.pi_min) ** 2)
    rows = []
    for fm in FMS:
        for m in (3, 5):
            g = runs[(runs.learner == fm) & (runs.mult == m) & runs.dataset.isin(b.index)]
            s = g[g.strategy == "stratified"].groupby("dataset").accuracy.mean()
            rr = g[g.strategy == "random"]
            cov = rr[rr.coverage >= 1].groupby("dataset").accuracy.mean()
            miss = rr.groupby("dataset").coverage.apply(lambda v: float((v < 1).mean()))
            j = pd.concat([s.rename("s"), cov.rename("cov"), miss.rename("miss")], axis=1).join(b.bstar)
            j = j.dropna(subset=["s", "cov"])
            for grp, sub in (("beyond", j[2 * m >= j.bstar]), ("below", j[2 * m < j.bstar])):
                rows.append(dict(fm=fm, m=m, group=grp, n=len(sub),
                                 gap_cov=(sub.s - sub["cov"]).mean() if len(sub) else np.nan,
                                 miss=sub.miss.mean() if len(sub) else np.nan))
    return pd.DataFrame(rows)


def dev_sets_by_kind():
    out = {}
    for did in cm.dev_ids():
        d = pd.read_parquet(cm.PROJECT_ROOT / "results" / "study7" / "a" / f"{did}_fm1.parquet",
                            columns=["strategy"])
        for k, v in d.strategy.value_counts().items():
            out[k] = out.get(k, 0) + int(v)
    sr = out.get("stratified", 0) + out.get("random", 0)
    return sr, sum(out.values()) - sr, out


def counterexample(pi1: float = 0.30):
    e = np.exp
    lo_mass, hi_mass = (e(-1) - e(-3)) / 2, (e(3) - e(1)) / 2
    w = (hi_mass - 1) / (hi_mass - lo_mass)  # makes class 1 a probability density

    def a_b(t):
        t = np.asarray(t, float)

        def seg(a_, b_, c):
            return c / 2 * (e(b_) - e(np.clip(t, a_, b_)))
        a = seg(-3, -1, w) + seg(1, 3, 1 - w)  # P1(l > t)
        b = np.clip((t + 3) / 2, 0, 1) * w + np.clip((t - 1) / 2, 0, 1) * (1 - w)  # P2(l <= t)
        return a, b

    ts = np.linspace(-4, 4, 80001)
    a, b = a_b(ts)
    p2 = 1 - pi1
    tp1, tp2, fp1, fn1 = pi1 * a, p2 * b, p2 * (1 - b), pi1 * (1 - a)
    mf = (2 * tp1 / (2 * tp1 + fp1 + fn1) + 2 * tp2 / (2 * tp2 + fn1 + fp1)) / 2
    arg = ts[mf >= mf.max() - 1e-12]
    a0, b0 = a_b(0.0)
    j0 = float(a0 + b0 - 1)
    assert j0 >= pi1 / p2 and pi1 < 1 / 3
    return dict(w=float(w), pi1=pi1, j0=j0, lo=float(arg.min()), hi=float(arg.max()), tstar=float(np.log(p2 / pi1)))


def main():
    acc = accbin_cov()
    acc.to_csv(OUT / "r7_accbin_cov.csv", index=False)
    print(acc.round(4).to_string())
    r6 = pd.read_csv(OUT / "r6_accbin.csv")
    sr, sel, kinds = dev_sets_by_kind()
    print("DEV labelled sets:", kinds, "S+R", sr, "selectors", sel)
    ce = counterexample()
    print("counterexample", ce)

    mac = Macros("r7")
    s = "r7_checks.py (1) EXPLORATORY, T1' binary tables with TV >= 0.1 (post hoc), m in {3,5}"
    for grp, gw in (("beyond", "Beyond"), ("below", "Below")):
        v = acc[acc.group == grp].gap_cov.dropna()
        mac.put(f"RsevAccMidCov{gw}Min", fmt(v.min(), sign=True), s + f", {grp} B*, vs covered random draws")
        mac.put(f"RsevAccMidCov{gw}Max", fmt(v.max(), sign=True), s + f", {grp} B*, vs covered random draws")
    mb = acc[(acc.group == "beyond") & (acc.m == 3)].miss.dropna()
    mac.put("RsevMissBeyondMThree", f"{mb.mean():.2f}", s + ", share of random draws missing a class, beyond B*, m=3")
    mid = r6[(r6.m.isin([3, 5])) & (r6.group == "beyond")]["mean"].dropna()
    mac.put("RsevAccMidBeyondAbsMin", fmt(-mid.max()), "r6_accbin.csv, |decline| beyond B*, m in {3,5}")
    mac.put("RsevAccMidBeyondAbsMax", fmt(-mid.min()), "r6_accbin.csv, |decline| beyond B*, m in {3,5}")
    src = "r7_checks.py (2) Study 7 Part (a) DEV, fm1 files"
    mac.put("RsevDevSetsSR", f"{sr:,}".replace(",", "{,}"), src + ", stratified or random labelled sets")
    mac.put("RsevDevSetsSel", f"{sel:,}".replace(",", "{,}"), src + ", label-free selector labelled sets")
    st = "r7_checks.py (3) THEORY, no data: Prop. 2(iii) counterexample under overlap alone"
    mac.put("RsevCeW", f"{ce['w']:.3f}", st + ", weight of U[-3,-1] in the class-2 density of l")
    mac.put("RsevCeWc", f"{1 - ce['w']:.3f}", st + ", weight of U[1,3]")
    mac.put("RsevCePi", f"{ce['pi1']:.2f}", st + ", pi_1")
    mac.put("RsevCeJzero", f"{ce['j0']:.2f}", st + ", Youden index at tau=0")
    mac.put("RsevCeFlatLo", fmt(ce["lo"], 1, True), st + ", lower end of the flat maximum")
    mac.put("RsevCeFlatHi", fmt(ce["hi"], 1, True), st + ", upper end of the flat maximum")
    mac.write()


if __name__ == "__main__":
    main()
