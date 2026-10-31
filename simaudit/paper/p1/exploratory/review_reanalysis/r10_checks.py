"""Round-10 review checks (EXPLORATORY or THEORY; read-only inputs).

(1) M10-2: share of the DEV gap at three labels per class that the coverage part accounts for,
    per FM: f1_miss_obs / f1_gap in mech_summary.csv (DEV, all tables, m=3), unrounded.
(2) C10-1 (THEORY, no data): finite-complement example. Pool n = 106 with n1 = 21 rows of class 1
    (a T1'-sized table), stratified draw with m = 10 per class: the class-1 share of U is
    (n1 - m)/(n - 2m). Two unit-variance Gaussian classes at distance 1, Bayes plug-in rule;
    expected accuracy on U with the pool prior pi vs with U's share. Shows that "the true prior is
    optimal" (Prop. 2(ii)) holds for population metrics under pi (n >> B), not on a small U.
Writes out/rv_r10.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r10_checks.py
"""

from __future__ import annotations

import re

import numpy as np
from scipy.stats import norm

from rv_common import FMW, FMS, OUT, Macros, fmt


def val(rv: str, name: str) -> float:
    s = re.search(rf"\\newcommand{{\\{name}}}{{([^}}]*)}}", rv).group(1)
    return float(s.replace("$-$", "-").replace("$+$", "+").replace("$", ""))


def acc_u(nu1: float, s: float, delta: float = 1.0) -> float:
    """Accuracy on a set with class-1 share s; class 1 ~ N(delta, 1), class 2 ~ N(0, 1); plug-in prior nu1."""
    t = delta / 2 + np.log((1 - nu1) / nu1) / delta
    return s * norm.sf(t - delta) + (1 - s) * norm.cdf(t)


def main():
    m = Macros("r10")
    import pandas as pd
    ms = pd.read_csv(OUT / "mech_summary.csv")
    ms = ms[(ms.set == "dev") & (ms.kind == "all") & (ms.mult == 3)].set_index("fm")
    sh = [float(ms.loc[fm, "f1_miss_obs"] / ms.loc[fm, "f1_gap"]) for fm in FMS]  # unrounded (round-11 m11-11)
    src = "r10_checks.py (1) mech_summary.csv f1_miss_obs / f1_gap (DEV, all tables, m=3; unrounded)"
    m.put("RtenCovShareMThreeMin", f"{100 * min(sh):.0f}\\%", src + ", min over 4 FMs")
    m.put("RtenCovShareMThreeMax", f"{100 * max(sh):.0f}\\%", src + ", max over 4 FMs")
    n, n1, mm = 106, 21, 10
    pi1, s = n1 / n, (n1 - mm) / (n - 2 * mm)
    grid = np.linspace(0.01, 0.99, 9801)
    best = grid[np.argmax([acc_u(g, s) for g in grid])]
    a_pi, a_s = acc_u(pi1, s), acc_u(s, s)
    assert abs(best - s) < 1e-3 and a_s > a_pi
    st = "r10_checks.py (2) THEORY, no data: n=106, n1=21, m=10, Gaussian classes at distance 1"
    m.put("RtenFinN", str(n), st)
    m.put("RtenFinNone", str(n1), st)
    m.put("RtenFinM", str(mm), st)
    m.put("RtenFinPi", f"{pi1:.3f}", st + ", pool share of class 1")
    m.put("RtenFinShareU", f"{s:.3f}", st + ", class-1 share of U under S")
    m.put("RtenFinAccPi", f"{a_pi:.4f}", st + ", accuracy on U with prior pi")
    m.put("RtenFinAccU", f"{a_s:.4f}", st + ", accuracy on U with prior = U's share (the optimum)")
    m.write()
    print("cov share m3:", [round(x, 3) for x in sh], "finite:", pi1, s, best, a_pi, a_s)


if __name__ == "__main__":
    main()
