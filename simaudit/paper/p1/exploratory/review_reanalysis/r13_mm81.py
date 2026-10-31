"""Round-13 check (EXPLORATORY; pure computation, no data): V_S of Prop. 4(iii) from Manski & McFadden (1981).

Manski & McFadden (1981, ch. 1, Sec. 1.5, Eq. 1.11) give the information matrix J_c of choice-based sampling
with the feature law p known and the shares Q unknown. Specialised to a saturated binary model theta = eta(z)
on a finite feature space with equal strata H(1) = H(2) = 1/2, J_c = (1/2)[diag(p r / w) - K p p'], and
B Var(pi1_hat) = p' J_c^{-1} p, which Sherman-Morrison turns into 2G/(1-KG) = V_S. This script builds J_c
literally from Eq. (1.11) and compares p' J_c^{-1} p with V_S in random cases.
Writes out/rv_r13.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r13_mm81.py
"""

from __future__ import annotations

import numpy as np

from rv_common import Macros

SEED, CASES, ZMIN, ZMAX, H = 1, 8, 3, 7, 0.5


def jc_mm81(p: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Eq. (1.11): sum_z sum_i P(i|z) p(z) H(i)/Q(i) dlnP dlnP' - sum_i H(i) dlnQ_i dlnQ_i'."""
    q1 = p @ eta
    q2 = 1 - q1
    d1 = np.diag(p * eta * H / q1 / eta**2)            # i = 1: dlnP(1|z)/deta = e_z / eta
    d2 = np.diag(p * (1 - eta) * H / q2 / (1 - eta) ** 2)  # i = 2: dlnP(2|z)/deta = -e_z / (1 - eta)
    g1, g2 = p / q1, -p / q2                            # dlnQ_1/deta, dlnQ_2/deta
    return d1 + d2 - H * np.outer(g1, g1) - H * np.outer(g2, g2)


def v_s(p: np.ndarray, eta: np.ndarray) -> float:
    pi1 = p @ eta
    pi2 = 1 - pi1
    w, r = eta * (1 - eta), (1 - eta) / pi1 + eta / pi2
    g, k = p @ (w / r), 1 / pi1**2 + 1 / pi2**2
    return 2 * g / (1 - k * g)


def main() -> None:
    rng = np.random.default_rng(SEED)
    rel, zs = [], []
    for _ in range(CASES):
        z = int(rng.integers(ZMIN, ZMAX + 1))
        p = rng.dirichlet(np.ones(z))
        eta = rng.uniform(0.03, 0.97, z)
        v_mm = p @ np.linalg.solve(jc_mm81(p, eta), p)
        vs = v_s(p, eta)
        rel.append(abs(v_mm - vs) / vs)
        zs.append(z)
        print(f"Z={z} pi1={p @ eta:.3f} MM81(1.11)={v_mm:.10f} V_S={vs:.10f}")
    mx = max(rel)
    print(f"max relative difference {mx:.2e}")
    m = Macros("r13")
    src = "r13_mm81.py (pure computation)"
    m.put("RthirteenMMCases", str(CASES), src + ": random saturated cases")
    m.put("RthirteenMMZMin", str(min(zs)), src + ": smallest number of feature values")
    m.put("RthirteenMMZMax", str(max(zs)), src + ": largest number of feature values")
    m.put("RthirteenMMDigits", str(int(np.floor(-np.log10(mx)))), src + ": significant digits of agreement (floor -log10 max rel diff)")
    m.write()


if __name__ == "__main__":
    main()
