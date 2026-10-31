"""E3 (exploratory): multiclass Bayes plug-in simulation for the parts of Sec. 4 proved for C = 2.

Model (H2 of Prop. 2): C Gaussian classes N(mu_k, I_C), mu_k = (Delta/sqrt 2) e_k
(pairwise distance Delta), geometric priors pi_k ∝ r^k with imbalance ratio IR =
pi_max/pi_min. The learner knows the class-conditional densities and takes its
prior from the context: nu = N / B (L0: an unseen class has nu_k = 0 and is never
predicted). Stratified: nu = u. Random: N ~ multivariate hypergeometric from a
pool of n = 5000 rows with counts round(pi * n).

Population metrics are computed from a confusion matrix estimated on a large
fixed per-class test sample (common random numbers across priors):
TP_k = pi_k R_kk, FP_k = sum_{j≠k} pi_j R_jk, FN_k = pi_k (1 - R_kk).

Checks (all exploratory):
 1. Total gap Δ = M(u) − E M(N/B), split into a coverage part (draws that miss a
    class) and a balance part (covered draws), for BA, accuracy and macro-F1.
 2. Prop. 1(iii) first-order coverage term (1/C) Σ q_k f_k (H1, no spillover)
    against the simulated coverage part of macro-F1.
 3. Prop. 2(i): BA balance part > 0 (uniform prior optimal).
 4. Prop. 2(iii), multiclass analogue: the macro-F1-optimal prior. (a) along the
    log-linear path log nu(s) = s log pi + const, the maximiser s*; (b) the
    unrestricted optimum (Nelder-Mead in C−1 log-ratios), its projection on the
    path and the share of classes whose centred log prior lies between 0 and
    that of pi.
 5. Prop. 2(iv) ordering Δacc ≤ ΔF1 ≤ ΔBA of the balance parts.
 6. Second-order expressions (as in the binary check of App. A):
    Δ_bal/P(cov) ≈ ½ [E_cov q(t̂) − q(t_u)], q(t) = (t − t*)ᵀ H (t − t*), with
    t = C−1 log-ratios to the last class, H the numerical Hessian of −M at its
    optimum t*, E_cov over the simulated covered compositions; sign agreement
    and correlation with the simulated balance parts.
 7. Prop. 2(ii), multiclass analogue of the accuracy crossover: the delta method
    gives B_cross ≈ φ tr(H_A Σ1) / q_A(t_u), Σ1 = diag(1/pi_k)_{k<C} + 1/pi_C;
    for C = 2 this is the paper's 1/(pi1 pi2 tau*²). Compared with the first
    simulated budget where random beats stratified in accuracy (covered draws).

Writes out/e3_cells.csv, out/e3_configs.csv, out/E3_SUMMARY.md.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize, minimize_scalar  # noqa: E402
from scipy.special import gammaln  # noqa: E402

import common as cm  # noqa: E402

SEED = 20261027
CS = (2, 3, 5, 10)  # C = 2 is the anchor (proved case)
IRS = (1.0, 4.0, 20.0)
DELTAS = (1.0, 2.0)
MS = (1, 2, 3, 5, 7, 10, 15, 25, 50)
POOL = 5000
DRAWS = 2000
N_TEST = {2: 100_000, 3: 60_000, 5: 30_000, 10: 12_000}
H_STEP = 0.2
B0 = 200  # reference budget for the Monte Carlo curvature of the crossover
K_SAMPLES = 400


def priors(C: int, ir: float) -> np.ndarray:
    r = ir ** (-1 / (C - 1))
    p = r ** np.arange(C)
    return p / p.sum()


class Pop:
    def __init__(self, C: int, delta: float, pi: np.ndarray, rng: np.random.Generator):
        self.C, self.pi = C, pi
        mu = np.eye(C) * delta / np.sqrt(2)
        n = N_TEST[C]
        self.LL = []
        for k in range(C):
            x = mu[k] + rng.standard_normal((n, C))
            self.LL.append(-0.5 * ((x[:, None, :] - mu[None]) ** 2).sum(-1))

    def confusion(self, nu: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            lp = np.log(nu)
        R = np.zeros((self.C, self.C))
        for k, ll in enumerate(self.LL):
            pred = np.argmax(ll + lp, axis=1)
            R[k] = np.bincount(pred, minlength=self.C) / len(pred)
        return R

    def metrics(self, nu: np.ndarray) -> dict[str, float]:
        R, pi = self.confusion(nu), self.pi
        tp = pi * np.diag(R)
        fn = pi - tp
        fp = (pi[:, None] * R).sum(0) - tp
        den = 2 * tp + fp + fn
        f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
        return dict(ba=float(np.diag(R).mean()), acc=float(tp.sum()), mf1=float(f1.mean()), f1k=f1)

    def from_t(self, t: np.ndarray, metric: str) -> float:
        z = np.append(t, 0.0)
        nu = np.exp(z - z.max())
        return self.metrics(nu / nu.sum())[metric]


def p_miss(counts: np.ndarray, b: int) -> np.ndarray:
    n = counts.sum()

    def lc(a, k):
        return gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1) if k <= a else -np.inf

    return np.array([np.exp(lc(n - c, b) - lc(n, b)) for c in counts])


def hessian(f, t0: np.ndarray, h: float = H_STEP) -> np.ndarray:
    d = len(t0)
    H = np.zeros((d, d))
    f0 = f(t0)
    for i in range(d):
        for j in range(i, d):
            ei, ej = np.eye(d)[i] * h, np.eye(d)[j] * h
            if i == j:
                H[i, i] = (f(t0 + ei) - 2 * f0 + f(t0 - ei)) / h**2
            else:
                H[i, j] = H[j, i] = (f(t0 + ei + ej) - f(t0 + ei - ej) - f(t0 - ei + ej) + f(t0 - ei - ej)) / (4 * h**2)
    return H


def run_config(job: tuple[int, float, float]) -> tuple[dict, list[dict]]:
    C, ir, delta = job
    rng = np.random.default_rng([SEED, C, int(ir), int(10 * delta)])
    pi = priors(C, ir)
    pop = Pop(C, delta, pi, rng)
    u = np.full(C, 1 / C)
    t_pi = np.log(pi[:-1] / pi[-1])
    t_u = np.zeros(C - 1)
    S = pop.metrics(u)
    # optimal priors and curvature (negated metrics are minimised)
    opt = {"ba": t_u, "acc": t_pi}
    res = minimize(lambda t: -pop.from_t(t, "mf1"), x0=0.5 * t_pi, method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-6, maxiter=400 * C))
    opt["mf1"] = res.x
    path = minimize_scalar(lambda s: -pop.from_t(s * t_pi, "mf1"), bounds=(-1, 2), method="bounded") if ir > 1 else None
    H = {m: -hessian(lambda t, m=m: pop.from_t(t, m), opt[m]) for m in ("ba", "acc", "mf1")}
    cfg = dict(C=C, IR=ir, Delta=delta, M_u_mf1=S["mf1"], M_pi_mf1=pop.metrics(pi)["mf1"],
               M_opt_mf1=-res.fun)
    if ir > 1:
        c_opt = np.append(opt["mf1"], 0.0)
        c_opt -= c_opt.mean()
        c_pi = np.log(pi) - np.log(pi).mean()
        between = ((c_opt >= np.minimum(0, c_pi) - 1e-9) & (c_opt <= np.maximum(0, c_pi) + 1e-9)).mean()
        cfg.update(s_path=float(path.x), proj=float(c_opt @ c_pi / (c_pi @ c_pi)), between_share=float(between))
        # multiclass crossover prediction (accuracy)
        # delta method without a finite-difference Hessian: K = B0 E[M(t*) - M(t* + e)],
        # e ~ N(0, Sigma1 / B0), estimates (1/2) tr(H_A Sigma1); L_u = M(pi) - M(u) is the
        # exact accuracy loss of the stratified prior. B_cross = K / L_u (C = 2: 1/(pi1 pi2 tau*^2)).
        Sig1 = np.diag(1 / pi[:-1]) + 1 / pi[-1]
        eps = rng.multivariate_normal(np.zeros(C - 1), Sig1 / B0, size=K_SAMPLES)
        m_star = pop.from_t(t_pi, "acc")
        K = B0 * float(np.mean([m_star - pop.from_t(t_pi + e, "acc") for e in eps]))
        L_u = m_star - S["acc"]
        cfg.update(K_acc=K, L_u_acc=L_u)
        cfg["B_cross_pred_raw"] = float(K / L_u) if L_u > 0 and K > 0 else np.inf
        # pure second-order form (the paper's): curvature along d = t_u - t* at a small step
        d = t_u - t_pi
        L_q = (m_star - pop.from_t(t_pi + 0.3 * d, "acc")) / 0.09
        cfg["B_cross_quad"] = float(K / L_q) if L_q > 0 and K > 0 else np.inf
        if C == 2:
            cfg["B_cross_closed"] = float(1 / (pi[0] * pi[1] * np.log(pi[0] / pi[1]) ** 2))
    counts = np.maximum(np.round(pi * POOL).astype(int), 1)
    n = counts.sum()
    rows = []
    for m in MS:
        b = m * C
        if m >= counts.min():
            continue
        cache: dict[tuple, dict] = {}
        comps = rng.multivariate_hypergeometric(counts, b, size=DRAWS)
        vals = []
        for comp in comps:
            key = tuple(comp)
            if key not in cache:
                cache[key] = pop.metrics(comp / b)
            vals.append(cache[key])
        cov = (comps > 0).all(axis=1)
        row = dict(C=C, IR=ir, Delta=delta, m=m, B=b, p_cov=float(cov.mean()))
        for met in ("ba", "acc", "mf1"):
            v = np.array([x[met] for x in vals])
            row[f"gap_{met}"] = S[met] - v.mean()
            row[f"cov_{met}"] = float(((S[met] - v) * ~cov).mean())
            row[f"bal_{met}"] = float(((S[met] - v) * cov).mean())
            if cov.any():
                tc = np.log(comps[cov][:, :-1] / comps[cov][:, -1:])
                d = tc - opt[met]
                eq = float(np.einsum("ij,jk,ik->i", d, H[met], d).mean())
                qu = float((t_u - opt[met]) @ H[met] @ (t_u - opt[met]))
                row[f"bal2_{met}"] = 0.5 * (eq - qu) * cov.mean()
            else:
                row[f"bal2_{met}"] = np.nan
        q = p_miss(counts, b)
        row["cov_pred_mf1"] = float((q * S["f1k"]).mean())
        row["phi"] = (n - b) / (n - 1)
        rows.append(row)
    if ir > 1:
        bal = pd.DataFrame(rows)
        neg = bal[bal.bal_acc < 0]
        cfg["B_cross_sim"] = float(neg.B.min()) if len(neg) else np.inf
        raw = cfg["B_cross_pred_raw"]
        cfg["B_cross_pred"] = raw * (n - raw) / (n - 1) if np.isfinite(raw) else np.inf
    return cfg, rows


def main() -> None:
    jobs = list(product(CS, IRS, DELTAS))
    with ProcessPoolExecutor(5) as pool:
        res = list(pool.map(run_config, jobs))
    cfgs = pd.DataFrame([c for c, _ in res])
    cells = pd.DataFrame([r for _, rs in res for r in rs])
    cfgs.to_csv(cm.OUT / "e3_configs.csv", index=False)
    cells.to_csv(cm.OUT / "e3_cells.csv", index=False)
    summary(cfgs, cells)


def resummarize() -> None:
    summary(pd.read_csv(cm.OUT / "e3_configs.csv"), pd.read_csv(cm.OUT / "e3_cells.csv"))


def summary(cfgs: pd.DataFrame, cells: pd.DataFrame) -> None:
    L = ["# E3: multiclass Bayes plug-in simulation (exploratory, synthetic)", "",
         f"Gaussian classes N(Δ/√2·e_k, I_C), geometric priors, pool n = {POOL}, {DRAWS} random compositions per cell, "
         "population metrics from a fixed large test sample. C = 2 rows are the proved case (anchor). "
         "Details and formulas in `e3_multiclass_sim.py`.", ""]
    mc = cells[cells.C > 2]
    imb = mc[mc.IR > 1]
    covd = mc[mc.m <= 3]
    r_cov = np.corrcoef(mc.cov_mf1, mc.cov_pred_mf1)[0, 1]
    L += ["## Coverage (Prop. 1(iii), first-order, multiclass)", "",
          f"- Simulated coverage part of macro-F1 vs (1/C)Σ q_k f_k over {len(mc)} multiclass cells: Pearson r = {r_cov:.3f}; "
          f"mean simulated {mc.cov_mf1.mean():.4f}, mean predicted {mc.cov_pred_mf1.mean():.4f}; at m ≤ 3: "
          f"{covd.cov_mf1.mean():.4f} vs {covd.cov_pred_mf1.mean():.4f} (ratio {covd.cov_mf1.sum() / covd.cov_pred_mf1.sum():.2f}).",
          f"- The macro-F1 coverage part exceeds the accuracy coverage part in {(mc.cov_mf1 >= mc.cov_acc).mean():.2f} of multiclass cells "
          f"(binary anchor: {(cells[cells.C == 2].cov_mf1 >= cells[cells.C == 2].cov_acc).mean():.2f}).", ""]
    L += ["## Balance (Prop. 2)", "",
          f"- (i) BA balance part ≥ 0 in {(mc.bal_ba >= -1e-4).mean():.2f} of {len(mc)} multiclass cells "
          f"(min {mc.bal_ba.min():+.4f}).",
          f"- (iv) ordering Δacc ≤ ΔF1 ≤ ΔBA of the balance parts: multiclass {((mc.bal_acc <= mc.bal_mf1 + 1e-4) & (mc.bal_mf1 <= mc.bal_ba + 1e-4)).mean():.2f} "
          f"of cells (imbalanced only: {((imb.bal_acc <= imb.bal_mf1 + 1e-4) & (imb.bal_mf1 <= imb.bal_ba + 1e-4)).mean():.2f}); "
          f"binary anchor {((cells[cells.C == 2].bal_acc <= cells[cells.C == 2].bal_mf1 + 1e-4) & (cells[cells.C == 2].bal_mf1 <= cells[cells.C == 2].bal_ba + 1e-4)).mean():.2f}. "
          f"Δacc ≤ ΔF1 alone: {(mc.bal_acc <= mc.bal_mf1 + 1e-4).mean():.2f}; ΔF1 ≤ ΔBA alone: {(mc.bal_mf1 <= mc.bal_ba + 1e-4).mean():.2f}. "
          f"Binary imbalanced anchor: {_order(cells[(cells.C == 2) & (cells.IR > 1)]):.2f}. In balanced pools (IR = 1) ΔF1 > ΔBA "
          "for C = 2 as well, so the raw ordering is not implied there; Prop. 2(iv) is stated per unit of curvature.", ""]
    for met in ("ba", "acc", "mf1"):
        ok = mc[[f"bal_{met}", f"bal2_{met}"]].dropna()
        ok = ok[ok[f"bal_{met}"].abs() > 1e-3]
        sign = (np.sign(ok[f"bal_{met}"]) == np.sign(ok[f"bal2_{met}"])).mean()
        r = np.corrcoef(ok[f"bal_{met}"], ok[f"bal2_{met}"])[0, 1]
        L.append(f"- Second-order expression, {met}: sign agreement {sign:.2f} over {len(ok)} multiclass cells with |Δ_bal| > 0.001; Pearson r = {r:.3f}.")
    ic = cfgs[(cfgs.IR > 1)]
    L += ["", "## Macro-F1-optimal prior (Prop. 2(iii) analogue) and accuracy crossover (Prop. 2(ii) analogue)", "",
          "s_path: maximiser along log ν = s·log π (0 = uniform, 1 = true prior); proj: projection of the unrestricted "
          "optimum on that path; between: share of classes whose centred log prior lies between 0 and that of π. "
          "B_cross: first simulated budget (on the m grid) where random beats stratified in accuracy on covered draws, "
          "against the delta-method prediction (φ-corrected).", "",
          "| C | IR | Δ | M_F1(u) | M_F1(π) | M_F1(opt) | s_path | proj | between | B_cross sim | B_cross pred (exact loss) | B_cross pred (2nd order) | closed form (C=2) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ic.itertuples():
        L.append(f"| {r.C} | {r.IR:g} | {r.Delta:g} | {r.M_u_mf1:.3f} | {r.M_pi_mf1:.3f} | {r.M_opt_mf1:.3f} | {r.s_path:.2f} | "
                 f"{r.proj:.2f} | {r.between_share:.2f} | {r.B_cross_sim:g} | {r.B_cross_pred:.1f} | {r.B_cross_quad:.1f} | "
                 f"{getattr(r, 'B_cross_closed', float('nan')):.1f} |")
    mic = ic[ic.C > 2]
    inside = ((mic.s_path > 0) & (mic.s_path < 1)).mean()
    fin = mic[np.isfinite(mic.B_cross_sim)]
    ratio = (fin.B_cross_pred / fin.B_cross_sim)
    ratio_q = (fin.B_cross_quad / fin.B_cross_sim)
    bin_ = ic[ic.C == 2]
    closed_err = (bin_.B_cross_quad / bin_.B_cross_closed).round(2).tolist()
    L += ["", f"- Multiclass: 0 < s_path < 1 in {inside:.2f} of {len(mic)} imbalanced configs; median projection of the full optimum {mic.proj.median():.2f}; "
          f"median share of classes 'between' {mic.between_share.median():.2f}.",
          f"- Accuracy crossover: simulated within the grid in {len(fin)} of {len(mic)} multiclass configs; "
          f"predicted/simulated B ratio median {ratio.median():.2f} (range {ratio.min():.2f}–{ratio.max():.2f}); "
          f"second-order form: median ratio {ratio_q.median():.2f} (range {ratio_q.min():.2f}–{ratio_q.max():.2f}); "
          "the grid makes the simulated B an upper end of an interval.",
          f"- Binary anchor: second-order numerical B_cross / closed form 1/(π1π2τ*²) = {closed_err} (should be ≈ 1).", ""]
    L += ["## Cell table (multiclass, macro-F1 and accuracy parts)", "",
          md_cells(mc)]
    (cm.OUT / "E3_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))


def _order(df: pd.DataFrame) -> float:
    return float(((df.bal_acc <= df.bal_mf1 + 1e-4) & (df.bal_mf1 <= df.bal_ba + 1e-4)).mean())


def md_cells(df: pd.DataFrame) -> str:
    cols = ["C", "IR", "Delta", "m", "p_cov", "gap_mf1", "cov_mf1", "cov_pred_mf1", "bal_mf1", "bal2_mf1",
            "bal_acc", "bal_ba"]
    return df[cols].round(4).to_markdown(index=False)


if __name__ == "__main__":
    resummarize() if "--summary" in sys.argv else main()
