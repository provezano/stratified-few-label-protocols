"""Round-8 review checks (EXPLORATORY or THEORY; read-only inputs).

(1) C8-2 (THEORY, no data): Prop. 2(i) second-order form is binary-only. Three unit-variance Gaussian
    classes in one dimension, means 0, 0.3 and 6. For two plug-in priors nu, D = ||nu - u||^2 and the
    balanced-accuracy loss BA(u) - BA(nu): the larger D gives the smaller loss.
(2) M8-2: T1' (Study 12) tables whose evaluation panel is capped at 2,000 rows: some stored row index
    (labelled or scored, original order) is >= 2000, so n > 2000, while no fit scores more than 2000 rows.
    Cross-checked against n in data/pool_t1prime.yaml (n > 2000). STUDY12_REPORT and the prereg say 13;
    both sources here give 11.
(3) M8-3: range over the four FMs of the coverage part at three labels per class on DEV (all tables),
    from the \\ExRvMechDev*AllMThreeFCovObs macros (mech_summary.csv).
(4) M8-1: the two-sided Wilcoxon p of Study 12 (ii-c), which decides its pre-registered verdict.
Writes out/rv_r8.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r8_checks.py
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import norm

from rv_common import FMW, FMS, OUT, Macros, fmt
from simaudit.settings import settings


def ba_counterexample():
    x = np.linspace(-10, 16, 260001)
    dx = x[1] - x[0]
    P = np.array([norm.pdf(x, m, 1) for m in (0.0, 0.3, 6.0)])

    def ba(nu):
        h = np.argmax(np.log(nu)[:, None] + np.log(P + 1e-300), axis=0)
        return np.mean([P[k][h == k].sum() * dx for k in range(3)])

    u = np.ones(3) / 3
    out = []
    for nu in (np.array([1 / 3 + 0.05, 1 / 3 - 0.05, 1 / 3]), np.array([1 / 3, 1 / 3 + 0.08, 1 / 3 - 0.08])):
        out.append((float(((nu - u) ** 2).sum()), float(ba(u) - ba(nu))))
    return out


def capped_tables():
    n_cap = 0
    files = sorted((settings.results_dir / "study12").glob("*_fm2.parquet"))
    for f in files:
        d = pd.read_parquet(f, columns=["labeled", "query"])
        top = max(max(int(np.max(lab)), int(np.max(q))) for lab, q in zip(d["labeled"], d["query"]))
        n_q = max(len(q) for q in d["query"])
        n_cap += top >= 2000 and n_q <= 2000  # n > 2000 rows, and at most 2000 scored rows
    return n_cap, len(files)


def main():
    m = Macros("r8")
    (d1, l1), (d2, l2) = ba_counterexample()
    assert d2 > d1 and l2 < l1
    st = "r8_checks.py (1) THEORY, no data: C=3 Gaussian classes, means 0, 0.3, 6"
    m.put("ReightBaDOne", f"{d1:.4f}", st + ", D of prior (1/3+0.05, 1/3-0.05, 1/3)")
    m.put("ReightBaLossOne", f"{l1:.4f}", st + ", BA loss of that prior")
    m.put("ReightBaDTwo", f"{d2:.4f}", st + ", D of prior (1/3, 1/3+0.08, 1/3-0.08)")
    m.put("ReightBaLossTwo", f"{l2:.4f}", st + ", BA loss of that prior")
    n_cap, n_tab = capped_tables()
    import yaml
    from simaudit.settings import PROJECT_ROOT
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())["datasets"]
    n_pool = sum(int(d["n"]) > 2000 for d in pool)
    assert (n_pool, len(pool)) == (n_cap, n_tab), (n_pool, len(pool), n_cap, n_tab)
    m.put("ReightPanelCapped", str(n_cap), "r8_checks.py (2) results/study12 fm2 files: T1' tables with n > 2000 (panel capped)")
    m.put("ReightPanelTables", str(n_tab), "r8_checks.py (2) results/study12 fm2 files: T1' tables")
    rv = (OUT / "rv_numbers.tex").read_text(encoding="utf-8")
    vals = [float(re.search(rf"\\newcommand{{\\ExRvMechDev{FMW[fm]}AllMThreeFCovObs}}{{([^}}]*)}}", rv).group(1))
            for fm in FMS]
    src = "r8_checks.py (3) from \\ExRvMechDev*AllMThreeFCovObs (mech_summary.csv, DEV, all tables, m=3)"
    m.put("ReightCovMThreeMin", fmt(min(vals)), src + ", min over FMs")
    m.put("ReightCovMThreeMax", fmt(max(vals)), src + ", max over FMs")
    from simaudit.analysis import study12 as a12
    runs = a12.load_runs(settings.results_dir / "study12")
    t = a12.confirmatory(a12.cell_means(runs, "macro_f1")).set_index("id")
    r = t.loc["(ii-c)"]
    assert r.verdict == "difference", r.verdict
    m.put("ReightIIcWilcoxonP", f"{r.p_wilcoxon:.3f}",
          "r8_checks.py (4) Study 12 confirmatory (ii-c), two-sided Wilcoxon p that decides the pre-registered verdict 'difference'")
    m.write()
    print("(ii-c) Wilcoxon", r.p_wilcoxon, r.verdict)
    print("BA:", (d1, l1), (d2, l2), "capped", n_cap, "of", n_tab, "cov m3", min(vals), max(vals))


if __name__ == "__main__":
    main()
