"""Round-5 review checks.

(1) THEORY (no data): counterexample to Prop. 2(iv) without its condition. Binary H2 plug-in,
    pi_1 in {0.48, 0.45}, class 1 ~ N(0, 0.3^2), class 2 ~ N(0, 1). The rule predicts class 2
    iff log f2/f1 + tau > 0; tau* = log(pi_2/pi_1) maximises accuracy, tau = 0 balanced accuracy.
    Grid integration (400,001 points on [-8, 8]); tau grid step 0.001. Reports tau*, the
    macro-F1 maximiser tau_MF and TP_1, TP_2 at tau*.
(2) EXPLORATORY: Wilcoxon over tables of the 0.5 -> 0.6 majority-arm step on the 6 binary DEV
    tables with pi_min <= 1/3 (input out/r4_rise.csv from r4_checks.py), per FM, two-sided.
Writes out/r5_counterexample.csv, out/r5_rise_tests.csv, out/rv_r5.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r5_checks.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from rv_common import FMS, OUT, Macros, fmt


def counterexample(pi1: float, s1: float = 0.3, s2: float = 1.0) -> dict:
    pi2 = 1 - pi1
    x = np.linspace(-8, 8, 400_001)
    dx = x[1] - x[0]
    f1, f2 = stats.norm.pdf(x, 0, s1), stats.norm.pdf(x, 0, s2)
    llr = np.log(f2) - np.log(f1)

    def mets(tau):
        p2 = llr + tau > 0
        tp1, tp2 = pi1 * f1[~p2].sum() * dx, pi2 * f2[p2].sum() * dx
        fn1, fn2 = pi1 - tp1, pi2 - tp2
        mf = (2 * tp1 / (2 * tp1 + fn2 + fn1) + 2 * tp2 / (2 * tp2 + fn1 + fn2)) / 2
        return tp1 + tp2, mf, tp1, tp2

    ts = np.round(np.arange(-0.5, 0.6005, 0.001), 3)
    m = np.array([mets(t) for t in ts])
    tstar = float(np.log(pi2 / pi1))
    a = mets(tstar)
    return dict(pi1=pi1, tau_star=tstar, tau_acc=float(ts[m[:, 0].argmax()]), tau_mf=float(ts[m[:, 1].argmax()]),
                tp1=a[2], tp2=a[3])


def main():
    ce = pd.DataFrame([counterexample(0.48), counterexample(0.45)])
    ce.to_csv(OUT / "r5_counterexample.csv", index=False)
    print(ce.round(4).to_string())
    ri = pd.read_csv(OUT / "r4_rise.csv")
    e = ri[ri.elig_pi]
    rt = pd.DataFrame([dict(fm=fm, n=len(g), mean=g.d56.mean(), p_wilcoxon=float(stats.wilcoxon(g.d56).pvalue))
                       for fm, g in e.groupby("fm")])
    rt.to_csv(OUT / "r5_rise_tests.csv", index=False)
    print(rt.round(4).to_string())

    mac = Macros("r5")
    c = ce[ce.pi1 == 0.48].iloc[0]
    src = "r5_checks.py (1) THEORY, no data: pi_1 = 0.48, N(0,0.3^2) vs N(0,1)"
    mac.put("RfiveCePi", "0.48", src + "; design constant")
    mac.put("RfiveCeTauStar", fmt(c.tau_star, 3), src + "; tau* (accuracy optimum)")
    mac.put("RfiveCeTauMF", fmt(c.tau_mf, 2), src + "; macro-F1 optimum")
    mac.put("RfiveCeTPone", fmt(c.tp1, 3), src + "; TP_1 at tau*")
    mac.put("RfiveCeTPtwo", fmt(c.tp2, 3), src + "; TP_2 at tau*")
    src2 = "r5_checks.py (2) EXPLORATORY: Wilcoxon over the 6 eligible tables, d(0.5->0.6) majority arm"
    mac.put("RfiveRisePMin", f"{rt.p_wilcoxon.min():.2f}", src2 + "; min over FMs")
    mac.put("RfiveRisePMax", f"{rt.p_wilcoxon.max():.2f}", src2 + "; max over FMs")
    assert set(rt.fm) == set(FMS)
    tv = pd.read_csv(OUT / "bc_t1_pool_tv.csv")
    b = tv[(tv.C == 2) & (tv.tv >= 0.1)].copy()
    b["bstar"] = 1 / (b.pi_min * (1 - b.pi_min) * np.log((1 - b.pi_min) / b.pi_min) ** 2)
    cross = {m: int((2 * m >= b.bstar).sum()) for m in (1, 3, 5, 10)}
    print("binary imbalanced T1' tables:", len(b), "beyond B* per m:", cross)
    for m, w in ((1, "One"), (3, "Three"), (5, "Five"), (10, "Ten")):
        mac.put(f"RfiveBstarM{w}", str(cross[m]), "r5_checks.py: binary T1' tables with TV >= 0.1 whose budget B = 2m "
                f"reaches B* = 1/(pi1 pi2 tau*^2) of Prop. 2(ii), m = {m} (theory constant per table, no fits)")
    rows = int(pd.read_csv(OUT.parents[1] / "e17_metrics" / "e17_reproduction.csv").n_rows.sum())
    mac.put("RfiveEseventeenRows", f"{rows:,}".replace(",", "{,}"),
            "r5_checks.py: e17_metrics/e17_reproduction.csv n_rows summed (same as ExESeventeenReproRows, with separator)")
    mac.write()


if __name__ == "__main__":
    main()
