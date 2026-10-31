"""E11 (exploratory, R3-W6 and R2-W7): curve-model sensitivity of ρ(1) and ε-sensitivity of m* (Study 9, T1′).

Stored results only (results/study9/t1prime, 52 tables); curves = mean macro-F1 over
repetitions per (table, learner, strategy, m), as in the frozen analysis. Primary
learner TabICL (fm2); TabPFN v2 (fm1, the pre-registered replication) alongside.

ρ_S(1) = m_eq / 1, the random budget (labels per class) that matches the stratified
score at m = 1, under four curve models for the random curve R(m):
- **frozen**: monotone envelope, linear interpolation in log m between grid points,
  censored outside the grid (analysis.study9.equivalence_interval; Turnbull median,
  2000-resample bootstrap CI). Must reproduce the reported value.
- **grid**: the same envelope without interpolation: ρ ∈ (m_{i−1}, m_i] where m_i is the
  first grid point whose envelope reaches f_S(1) (left-censored (0, 1] if R(1) already
  does). Turnbull median cell; bootstrap CI of the cell's lower and upper ends.
- **log**: OLS fit R(m) = a + b ln m on the feasible grid; m_eq = exp((f_S(1) − a)/b)
  (b ≤ 0 → ρ = ∞ if f_S(1) > a, else undefined and dropped).
- **power**: inverse power law R(m) = c − a·m^(−β), a, β > 0, c ≤ 1 (scipy
  curve_fit, bounded); m_eq = (a / (c − f_S(1)))^(1/β); ρ = ∞ when f_S(1) ≥ c (no random
  budget reaches it; Corollary 1 second case).
Parametric ρ are point values (extrapolation allowed); medians over tables with a
2000-resample bootstrap CI over tables. Fit quality per table: R² and RMSE of the log
and power fits, and the mean residual per grid point (systematic misfit).

m*: the frozen saturation interval for ε ∈ {0.005, 0.01, 0.02} (TabICL, random),
Turnbull NPMLE with 2000 resamples (prereg number). Reported: median and quartile
cells, P(m* ≤ 5), P(m* ≤ 10), and P(gain > ε after m) = P(m* > m) = 1 − F(m) for
m ∈ {3, 5, 10, 20} with bootstrap CIs; also the direct share of tables whose next
measured step after m still gains ≥ ε per label per class, with a bootstrap CI.

Writes out/e11_rho_tables.csv, out/e11_rho.csv, out/e11_fit.csv, out/e11_mstar.csv, out/E11_SUMMARY.md.
"""

from __future__ import annotations

import sys
import warnings
from math import inf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

import common as cm  # noqa: E402
from simaudit.analysis import study9 as a9  # noqa: E402

GRID = (1, 2, 3, 5, 7, 10, 15, 20, 30)
EPSS = (0.005, 0.01, 0.02)


def power(m, c, a, b):
    return c - a * np.power(m, -b)


def fit_power(ms, f):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(power, ms, f, p0=(min(1.0, f.max() + 0.02), max(f.max() - f.min(), 0.01), 0.5),
                             bounds=([0, 1e-6, 1e-3], [1.0, 5.0, 5.0]), maxfev=20000)
        return p
    except RuntimeError:
        return None


def r2(f, fhat):
    ss = ((f - f.mean()) ** 2).sum()
    return float(1 - ((f - fhat) ** 2).sum() / ss) if ss > 0 else np.nan


def grid_interval(ms, fr, score):
    env = np.maximum.accumulate(fr)
    if score <= env[0]:
        return (0.0, float(ms[0]))
    if score > env[-1]:
        return (float(ms[-1]), inf)
    i = int(np.argmax(env >= score))
    return (float(ms[i - 1]), float(ms[i]))


def median_of(vals):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    return float(np.median(v)) if len(v) else np.nan


def per_table(curve: pd.DataFrame, learner: str):
    rnd = a9._xs(curve, learner, "random")
    st = a9._xs(curve, learner, "stratified")
    rows, fits, resid = [], [], []
    for key in st.index.intersection(rnd.index):
        ms, fr = a9._points(rnd.loc[key])
        fs1 = st.loc[key].get(1.0, st.loc[key].get(1))
        if fs1 is None or not np.isfinite(fs1) or 1 not in ms:
            continue
        fro = a9.equivalence_interval(ms, fr, fs1, 1)
        gi = grid_interval(ms, fr, fs1)
        b, a = np.polyfit(np.log(ms), fr, 1)
        if b > 0:
            rho_log = float(np.exp((fs1 - a) / b))
        else:
            rho_log = inf if fs1 > a else np.nan
        p = fit_power(ms, fr)
        if p is None:
            rho_pow, r2p, rmsep = np.nan, np.nan, np.nan
        else:
            c, aa, bb = p
            rho_pow = inf if fs1 >= c else float((aa / (c - fs1)) ** (1 / bb))
            fh = power(ms, *p)
            r2p, rmsep = r2(fr, fh), float(np.sqrt(np.mean((fr - fh) ** 2)))
            for m_, e in zip(ms, fr - fh):
                resid.append(dict(learner=learner, dataset=key, model="power", m=m_, resid=e))
        fl = a + b * np.log(ms)
        for m_, e in zip(ms, fr - fl):
            resid.append(dict(learner=learner, dataset=key, model="log", m=m_, resid=e))
        rows.append(dict(learner=learner, dataset=key, f_strat1=fs1, f_rand1=fr[0], K=len(ms),
                         frozen_lo=fro[0], frozen_hi=fro[1], grid_lo=gi[0], grid_hi=gi[1],
                         rho_log=rho_log, rho_power=rho_pow, log_b=b))
        fits.append(dict(learner=learner, dataset=key, r2_log=r2(fr, fl), rmse_log=float(np.sqrt(np.mean((fr - fl) ** 2))),
                         r2_power=r2p, rmse_power=rmsep))
    return pd.DataFrame(rows), pd.DataFrame(fits), pd.DataFrame(resid)


def grid_median(ivs):
    pts = sorted({x for iv in ivs for x in iv if 0 < x < inf})
    return a9.quantile_cell(a9.turnbull(ivs, pts), pts, 0.5)


def rho_summary(t: pd.DataFrame) -> list[dict]:
    out = []
    for lr, g in t.groupby("learner"):
        ivs = list(zip(g.frozen_lo, g.frozen_hi))
        med = a9.turnbull_median_ratio(ivs)
        lo, hi = a9.boot_ci(ivs, a9.turnbull_median_ratio)
        out.append(dict(learner=lr, model="frozen", median=med, lo=lo, hi=hi, n=len(g),
                        n_inf=int(sum(h == inf for _, h in ivs)), share_gt1=float(np.mean(g.frozen_lo > 1))))
        givs = list(zip(g.grid_lo, g.grid_hi))
        a, b = grid_median(givs)
        lo_a = a9.boot_ci(givs, lambda v: grid_median(v)[0])
        lo_b = a9.boot_ci(givs, lambda v: grid_median(v)[1])
        out.append(dict(learner=lr, model="grid", median=b, median_cell_lo=a, lo=lo_a[0], hi=lo_b[1], n=len(g),
                        n_inf=int(sum(h == inf for _, h in givs)), share_gt1=float(np.mean(g.grid_lo >= 1))))
        for model in ("log", "power"):
            v = g[f"rho_{model}"].to_numpy(float)
            lo, hi = a9.boot_ci(list(v), median_of)
            out.append(dict(learner=lr, model=model, median=median_of(v), lo=lo, hi=hi, n=int((~np.isnan(v)).sum()),
                            n_inf=int(np.isinf(v).sum()), share_gt1=float(np.nanmean(v > 1)),
                            q25=float(np.nanquantile(v, 0.25)), q75=float(np.nanquantile(v, 0.75))))
    return out


def mstar(curve: pd.DataFrame) -> pd.DataFrame:
    rnd = a9._xs(curve, "fm2", "random")
    rows = []
    for eps in EPSS:
        ivs = [a9.saturation_interval(*a9._points(r), eps=eps) for _, r in rnd.iterrows()]
        d = a9.distribution_summary(ivs, n_boot=a9.N_BOOT)
        A = a9._membership(ivs, GRID)
        rng = np.random.default_rng(a9.SEED)
        idx = rng.integers(0, len(A), size=(a9.N_BOOT, len(A)))
        cdf = np.cumsum(a9._em(A))
        boot = np.array([np.cumsum(a9._em(A[row])) for row in idx])
        rec = dict(eps=eps, n=len(ivs), median_lo=d["median"][0], median_hi=d["median"][1],
                   q25_lo=d["q25"][0], q25_hi=d["q25"][1], q75_lo=d["q75"][0], q75_hi=d["q75"][1],
                   n_right_cens=int(sum(h == inf for _, h in ivs)), n_left_cens=int(sum(lo == 0 for lo, _ in ivs)))
        for k in ("P(m*<=5)", "P(m*<=10)"):
            rec[k], rec[k + "_lo"], rec[k + "_hi"] = d[k]
        for m in (3, 5, 10, 20):
            i = GRID.index(m)
            b = 1 - boot[:, i]
            rec[f"P_gain_after_{m}"] = float(1 - cdf[i])
            rec[f"P_gain_after_{m}_lo"] = float(np.quantile(b, 0.025))
            rec[f"P_gain_after_{m}_hi"] = float(np.quantile(b, 0.975))
            nxt = []
            for _, r in rnd.iterrows():
                ms, f = a9._points(r)
                if m in ms and list(ms).index(m) < len(ms) - 1:
                    j = list(ms).index(m)
                    nxt.append(float((f[j + 1] - f[j]) / (ms[j + 1] - ms[j]) >= eps))
            nxt = np.array(nxt)
            bs = [nxt[ii].mean() for ii in np.random.default_rng(0).integers(0, len(nxt), size=(a9.N_BOOT, len(nxt)))]
            rec[f"next_step_{m}"] = float(nxt.mean())
            rec[f"next_step_{m}_lo"] = float(np.quantile(bs, 0.025))
            rec[f"next_step_{m}_hi"] = float(np.quantile(bs, 0.975))
            rec[f"next_step_{m}_n"] = len(nxt)
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    raw = a9.load_results(cm.settings.results_dir / "study9" / "t1prime")
    curve = a9.curves(raw)
    tabs, fits, res = [], [], []
    for lr in ("fm2", "fm1"):
        t, f, r = per_table(curve, lr)
        tabs.append(t), fits.append(f), res.append(r)
    t, f, r = pd.concat(tabs), pd.concat(fits), pd.concat(res)
    t.to_csv(cm.OUT / "e11_rho_tables.csv", index=False)
    f.to_csv(cm.OUT / "e11_fit.csv", index=False)
    rho = pd.DataFrame(rho_summary(t))
    rho.to_csv(cm.OUT / "e11_rho.csv", index=False)
    ms = mstar(curve)
    ms.to_csv(cm.OUT / "e11_mstar.csv", index=False)
    meanres = r.groupby(["learner", "model", "m"]).resid.mean().unstack("m")
    L = ["# E11: Study 9 sensitivity (exploratory, T1′, 52 tables, stored results)", "",
         "## ρ_S(1) of the stratified protocol under four curve models", "",
         "| learner | model | median ρ(1) [95% CI] | n | ρ = ∞ | share ρ > 1 |", "|---|---|---|---|---|---|"]
    for _, x in rho.iterrows():
        med = (f"cell ({x.median_cell_lo:g}, {x['median']:g}]" if x.model == "grid" else f"{x['median']:.2f}")
        L.append(f"| {x.learner} | {x.model} | {med} [{x.lo:.2f}, {x.hi:.2f}] | {x.n} | {x.n_inf} | {x.share_gt1:.2f} |")
    L += ["", "grid: the CI is [2.5% of the median cell's lower end, 97.5% of its upper end].", "",
          "## Fit quality of the random curve (per table)", "",
          "| learner | model | median R² | share R² ≥ 0.9 | median RMSE |", "|---|---|---|---|---|"]
    for lr, g in f.groupby("learner"):
        for model in ("log", "power"):
            L.append(f"| {lr} | {model} | {g[f'r2_{model}'].median():.3f} | {(g[f'r2_{model}'] >= 0.9).mean():.2f} | "
                     f"{g[f'rmse_{model}'].median():.4f} |")
    L += ["", "Mean residual (observed − fit) per grid point (systematic misfit):", "",
          meanres.round(4).to_markdown(), "",
          "## m* for ε ∈ {0.005, 0.01, 0.02} (TabICL, random; Turnbull, 2000 resamples)", "",
          "| ε | median cell | IQR cells | P(m*≤5) | P(m*≤10) | P(gain>ε after 5) | after 10 | after 20 | right-censored |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, x in ms.iterrows():
        L.append(f"| {x.eps} | ({x.median_lo:g}, {x.median_hi:g}] | ({x.q25_lo:g}, {x.q25_hi:g}]–({x.q75_lo:g}, {x.q75_hi:g}] | "
                 f"{x['P(m*<=5)']:.2f} [{x['P(m*<=5)_lo']:.2f}, {x['P(m*<=5)_hi']:.2f}] | "
                 f"{x['P(m*<=10)']:.2f} [{x['P(m*<=10)_lo']:.2f}, {x['P(m*<=10)_hi']:.2f}] | "
                 + " | ".join(f"{x[f'P_gain_after_{m}']:.2f} [{x[f'P_gain_after_{m}_lo']:.2f}, {x[f'P_gain_after_{m}_hi']:.2f}]" for m in (5, 10, 20))
                 + f" | {int(x.n_right_cens)} |")
    L += ["", "Direct share of tables whose next measured step after m gains ≥ ε per label per class [bootstrap CI]:", "",
          "| ε | after 3 | after 5 | after 10 | after 20 |", "|---|---|---|---|---|"]
    for _, x in ms.iterrows():
        L.append(f"| {x.eps} | " + " | ".join(f"{x[f'next_step_{m}']:.2f} [{x[f'next_step_{m}_lo']:.2f}, {x[f'next_step_{m}_hi']:.2f}] (n={int(x[f"next_step_{m}_n"])})" for m in (3, 5, 10, 20)) + " |")
    (cm.OUT / "E11_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
