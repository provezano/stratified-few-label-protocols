"""Round-6 review checks.

(1) EXPLORATORY: T1' accuracy, stratified - random, on the 18 binary tables with TV >= 0.1 (post hoc
    split), per FM and budget, split by whether B = 2m reaches B* = 1/(pi1 pi2 tau*^2) of Prop. 2(ii).
    At m = 1 (no table reaches B*), the random draws are split into covered and missed draws:
    per table, mean accuracy of stratified minus mean accuracy of the covered random draws, and the
    share of missed random draws.
(3) THEORY (no data): (a) a binary H2 task whose macro-F1 maximiser lies below the uniform prior
    (pi_1 = 0.45, class 1 ~ N(0.2, 1), class 2 ~ N(0, 0.5^2)), so the second inequality of Prop. 2(iv)
    fails; (b) a grid check of the global claim of Prop. 2(iii) on random binary Gaussian tasks that
    meet its conditions (pi_1 < 1/3, J0 >= pi_1/pi_2): MF non-decreasing on tau < 0, non-increasing on
    tau > tau*, and no value outside [0, tau*] above the maximum inside (ties allowed).
(2) EXPLORATORY: DEV, Study 7 Part (a): number of stored labelled sets in which a seen class has every
    row labelled (u_j = 0 for a seen class j), the edge case of Prop. 5(i).
Writes out/r6_accbin.csv, out/r6_m1cov.csv, out/rv_r6.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r6_checks.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from rv_common import FMS, OUT, Macros, fmt
import bc_contrasts as bc

sys.path.insert(0, str(OUT.parents[1]))
import common as cm  # noqa: E402


def accbin():
    runs = bc.load_t1()
    c = bc.cell(runs)
    tv = pd.read_csv(OUT / "bc_t1_pool_tv.csv").set_index("dataset")
    b = tv[(tv.C == 2) & (tv.tv >= 0.1)].copy()
    b["bstar"] = 1 / (b.pi_min * (1 - b.pi_min) * np.log((1 - b.pi_min) / b.pi_min) ** 2)
    rows = []
    for fm in FMS:
        for m in (1, 3, 5, 10):
            x, _ = bc.diff(c, fm, "stratified", "random", (m,), "accuracy")
            j = b.join(x.rename("dA"), how="inner")
            for grp, sub in (("beyond", j[2 * m >= j.bstar]), ("below", j[2 * m < j.bstar])):
                rows.append(dict(fm=fm, m=m, group=grp, n=len(sub), mean=sub.dA.mean() if len(sub) else np.nan))
    acc = pd.DataFrame(rows)
    # m = 1: covered vs missed random draws
    r = runs[(runs.mult == 1) & runs.dataset.isin(b.index)]
    m1 = []
    for fm in FMS:
        g = r[r.learner == fm]
        s = g[g.strategy == "stratified"].groupby("dataset").accuracy.mean()
        rr = g[g.strategy == "random"]
        cov = rr[rr.coverage >= 1].groupby("dataset").accuracy.mean()
        allr = rr.groupby("dataset").accuracy.mean()
        miss = rr.groupby("dataset").coverage.apply(lambda v: float((v < 1).mean()))
        j = pd.concat([s.rename("s"), cov.rename("cov"), allr.rename("all"), miss.rename("miss")], axis=1).dropna()
        m1.append(dict(fm=fm, n=len(j), gap_all=(j.s - j["all"]).mean(), gap_cov=(j.s - j["cov"]).mean(),
                       miss=j.miss.mean()))
    return acc, pd.DataFrame(m1)


def dev_full_class():
    tot = hit = 0
    for did in cm.dev_ids():
        data = cm.dev_data(did)
        y = np.asarray(data[1] if isinstance(data, tuple) else data.y)
        cnt = np.bincount(y)
        d = pd.read_parquet(cm.PROJECT_ROOT / "results" / "study7" / "a" / f"{did}_fm1.parquet", columns=["labeled"])
        for lab in d.labeled:
            k = np.bincount(y[lab], minlength=len(cnt))
            tot += 1
            hit += bool(np.any((k > 0) & (k == cnt)))
    return tot, hit


def _mf_curve(pi1, m1, s1, m2, s2, ts, n=8001):
    from scipy.stats import norm
    x = np.linspace(-15, 15, n)
    dx = x[1] - x[0]
    p1, p2 = norm.pdf(x, m1, s1), norm.pdf(x, m2, s2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.log(p1) - np.log(p2)
    pi2 = 1 - pi1
    out = []
    for t in ts:
        c = ll > t
        a, b = p1[c].sum() * dx, p2[~c].sum() * dx
        d1, d2 = pi1 * a + pi1 + pi2 * (1 - b), pi2 * b + pi2 + pi1 * (1 - a)
        out.append((pi1 * a / d1 + pi2 * b / d2, a, b))
    return np.array(out)


def theory():
    ts = np.round(np.arange(-1.0, 1.0005, 0.002), 3)
    v = _mf_curve(0.45, 0.2, 1.0, 0.0, 0.5, ts)
    tau_neg = float(ts[v[:, 0].argmax()])
    rng = np.random.default_rng(0)
    n_ok = n_bad = 0
    while n_ok + n_bad < 400:
        pi1 = rng.uniform(0.02, 1 / 3 - 1e-3)
        m1, m2 = rng.normal(0, 2, 2)
        s1, s2 = rng.uniform(0.3, 2, 2)
        tstar = np.log((1 - pi1) / pi1)
        a0b0 = _mf_curve(pi1, m1, s1, m2, s2, [0.0])[0]
        j0 = a0b0[1] + a0b0[2] - 1
        if j0 < pi1 / (1 - pi1) or j0 > 0.999:
            continue
        grid = np.unique(np.concatenate([np.linspace(-6, tstar + 6, 361), [0.0, tstar]]))
        mf = _mf_curve(pi1, m1, s1, m2, s2, grid)[:, 0]
        left, right = mf[grid < 0], mf[grid > tstar]
        inside = mf[(grid >= 0) & (grid <= tstar)].max()
        ok = (np.all(np.diff(left) >= -1e-9) and np.all(np.diff(right) <= 1e-9)
              and mf[(grid < 0) | (grid > tstar)].max() <= inside + 1e-9)  # ties allowed (flat MF)
        n_ok += ok
        n_bad += not ok
    return tau_neg, n_ok, n_bad


def main():
    pd.set_option("display.width", 200)
    acc, m1 = accbin()
    acc.to_csv(OUT / "r6_accbin.csv", index=False)
    m1.to_csv(OUT / "r6_m1cov.csv", index=False)
    print(acc.round(4).to_string())
    print(m1.round(4).to_string())
    tot, hit = dev_full_class()
    print("DEV labelled sets", tot, "with a fully labelled seen class", hit)

    tau_neg, n_ok, n_bad = theory()
    print("tau_MF below 0:", tau_neg, "global check ok/bad:", n_ok, n_bad)
    mac = Macros("r6")
    st = "r6_checks.py (3) THEORY, no data"
    mac.put("RsixCeNegTau", fmt(tau_neg, 2, True), st + ": pi_1=0.45, N(0.2,1) vs N(0,0.5^2), macro-F1 maximiser")
    mac.put("RsixGlobalN", str(n_ok + n_bad), st + ": random binary Gaussian tasks meeting (iii)")
    mac.put("RsixGlobalBad", str(n_bad), st + ": of them, tasks violating monotonicity outside [0, tau*] or the location of the maximiser")
    s = "r6_checks.py (1) EXPLORATORY, T1' binary tables with TV >= 0.1 (post hoc)"
    for m, w in ((1, "One"), (3, "Three"), (5, "Five")):
        for grp, gw in (("beyond", "Beyond"), ("below", "Below")):
            v = acc[(acc.m == m) & (acc.group == grp)]["mean"].dropna()
            if len(v):
                mac.put(f"RsixAccM{w}{gw}Min", fmt(v.min(), 3, True), f"{s}; m={m}, {grp} B*, min over FMs")
                mac.put(f"RsixAccM{w}{gw}Max", fmt(v.max(), 3, True), f"{s}; m={m}, {grp} B*, max over FMs")
    mid = acc[acc.m.isin([3, 5])]
    for grp, gw in (("beyond", "Beyond"), ("below", "Below")):
        v = mid[mid.group == grp]["mean"]
        mac.put(f"RsixAccMid{gw}Min", fmt(v.min(), 3, True), f"{s}; m in {{3,5}}, {grp} B*, min over FM x m")
        mac.put(f"RsixAccMid{gw}Max", fmt(v.max(), 3, True), f"{s}; m in {{3,5}}, {grp} B*, max over FM x m")
    mac.put("RsixMOneGapCovMin", fmt(m1.gap_cov.min(), 3, True), f"{s}; m=1, stratified minus covered random draws, min over FMs")
    mac.put("RsixMOneGapCovMax", fmt(m1.gap_cov.max(), 3, True), f"{s}; m=1, stratified minus covered random draws, max over FMs")
    mac.put("RsixMOneMiss", fmt(m1.miss.mean(), 2), f"{s}; m=1, mean share of random draws that missed a class")
    mac.put("RsixDevSets", f"{tot:,}".replace(",", "{,}"), "r6_checks.py (2): DEV Study 7 Part (a) stored labelled sets (fm1 files)")
    mac.put("RsixDevFullClass", str(hit), "r6_checks.py (2): of them, sets in which a seen class has every row labelled")
    mac.write()


if __name__ == "__main__":
    main()
