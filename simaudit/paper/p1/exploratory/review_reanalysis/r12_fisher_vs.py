"""Round-12 check (EXPLORATORY; pure computation, no data): V_S of Prop. 5(iii) is the inverse Fisher information.

Saturated model on a finite feature space with Z values: p(x) known, eta(x) = P(y=1|x) unrestricted. One
stratified pair (x1 ~ p1, x2 ~ p2) has log-likelihood log(p eta / pi1) + log(p (1-eta) / pi2) with
pi1 = p'eta. This is the case-control likelihood of Hsieh, Manski & McFadden (1985, Eq. 12) with equal
case-control samples, no random sample of labels and a known feature law. The exact (analytic) Fisher
information I for eta per pair gives the efficient variance of pi1 = p'eta per pair, p' I^{-1} p; for B = 2m
labels the variance of sqrt(B)(pi1_hat - pi1) is 2 p' I^{-1} p, compared with V_S = 2G/(1-KG) (Appendix proof).
Writes out/rv_r12.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r12_fisher_vs.py
"""

from __future__ import annotations

import numpy as np

from rv_common import Macros

SEED, CASES, ZMIN, ZMAX = 0, 6, 3, 6


def fisher_pair(p: np.ndarray, eta: np.ndarray) -> np.ndarray:
    pi1 = p @ eta
    pi2 = 1 - pi1
    p1, p2 = p * eta / pi1, p * (1 - eta) / pi2
    eye = np.eye(len(p))
    s1 = eye / eta[:, None] - p[None, :] / pi1          # d/d eta_j of log p1(x), rows x
    s2 = -eye / (1 - eta)[:, None] + p[None, :] / pi2   # d/d eta_j of log p2(x)
    return s1.T @ (p1[:, None] * s1) + s2.T @ (p2[:, None] * s2)


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
        eta = rng.uniform(0.05, 0.95, z)
        eff = 2 * p @ np.linalg.solve(fisher_pair(p, eta), p)
        vs = v_s(p, eta)
        rel.append(abs(eff - vs) / vs)
        zs.append(z)
        print(f"Z={z} pi1={p @ eta:.3f} 2p'I^-1p={eff:.10f} V_S={vs:.10f}")
    mx = max(rel)
    print(f"max relative difference {mx:.2e}")
    m = Macros("r12")
    src = "r12_fisher_vs.py (pure computation)"
    m.put("RtwelveFisherCases", str(CASES), src + ": random saturated cases")
    m.put("RtwelveFisherZMin", str(min(zs)), src + ": smallest number of feature values")
    m.put("RtwelveFisherZMax", str(max(zs)), src + ": largest number of feature values")
    m.put("RtwelveFisherDigits", str(int(np.floor(-np.log10(mx)))), src + ": significant digits of agreement (floor -log10 max rel diff)")
    m.write()


if __name__ == "__main__":
    main()
