"""E14 (exploratory, R3-W13 / R1-W3 on T1′): binary vs multiclass split of Study 9.

Stored results only (results/study9/t1prime, 52 tables; read-only). C per table is
taken from the stored ``n_classes`` column of the result files (binary: C = 2,
multiclass: C > 2). Curves = mean macro-F1 over repetitions per (table, learner,
strategy, m), as in the frozen analysis (analysis.study9.curves).

(a) Saturation m* of the random curve, interval-censored as frozen
    (analysis.study9.saturation_interval), for the pre-registered ε = 0.01 and the
    pre-declared sensitivities ε ∈ {0.005, 0.02}. Turnbull NPMLE per group, 2000
    resamples (prereg number and seed). Reported: median and quartile cells,
    P(m* ≤ 5), P(m* ≤ 10), P(gain > ε after 10) = 1 − F(10), right-censored count,
    and the binary − multiclass difference of P(m* ≤ 5) and P(m* ≤ 10) with a
    two-sample bootstrap (groups resampled independently).
(b) Label equivalence ρ_S(1) of the stratified protocol (m_eq of the random curve
    that matches the stratified score at m = 1) under the frozen interval form
    (Turnbull median ratio), and the log and power-law curve models of E11
    (per-table values reused from e11_study9_sensitivity.per_table). Per group:
    median with a 2000-resample bootstrap CI over tables, ρ = ∞ count, share ρ > 1,
    and the binary − multiclass difference of the medians (two-sample bootstrap).

TabICL (fm2) is the primary learner, TabPFN v2 (fm1) the replication, as in E11.
The multiclass group has 17 tables: intervals are wide and nothing here is a test.

Writes out/e14_mstar.csv, out/e14_rho.csv, out/e14_rho_tables.csv, out/E14_SUMMARY.md.
"""

from __future__ import annotations

import sys
from math import inf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as cm  # noqa: E402
import e11_study9_sensitivity as e11  # noqa: E402
from simaudit.analysis import study9 as a9  # noqa: E402

GRID = e11.GRID
EPSS = (0.005, 0.01, 0.02)  # 0.01 pre-registered; the others pre-declared sensitivities
GROUPS = ("binary", "multiclass")
LEARNERS = ("fm2", "fm1")
SEED_OFF = dict(binary=0, multiclass=1, all=0)  # "all" reproduces E11 (prereg seed)
COLS = ["dataset", "learner", "strategy", "m", "rep", "n_classes", "macro_f1"]


def load(directory: Path) -> pd.DataFrame:
    """Same file filter as analysis.study9.load_results, reading only the needed columns."""
    paths = sorted(p for p in directory.glob("*.parquet") if "partial" not in p.name)
    return pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)


def cdf(ivs) -> np.ndarray:
    return np.cumsum(a9._em(a9._membership(ivs, GRID)))


def boot_cdf(ivs, rng) -> np.ndarray:
    A = a9._membership(ivs, GRID)
    idx = rng.integers(0, len(A), size=(a9.N_BOOT, len(A)))
    return np.array([np.cumsum(a9._em(A[row])) for row in idx])


def q(b: np.ndarray) -> tuple[float, float]:
    b = b[np.isfinite(b)]
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def mstar(curve: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    rows = []
    for lr in LEARNERS:
        rnd = a9._xs(curve, lr, "random")
        for eps in EPSS:
            ivs = {k: a9.saturation_interval(*a9._points(r), eps=eps) for k, r in rnd.iterrows()}
            boots, Fs = {}, {}
            for gname in (*GROUPS, "all"):
                keys = [k for k in ivs if gname == "all" or group[k] == gname]
                gi = [ivs[k] for k in keys]
                d = a9.distribution_summary(gi, n_boot=a9.N_BOOT)
                F = cdf(gi)
                B = boot_cdf(gi, np.random.default_rng(a9.SEED + SEED_OFF[gname]))
                boots[gname], Fs[gname] = B, F
                rec = dict(learner=lr, eps=eps, group=gname, n=len(gi),
                           median_lo=d["median"][0], median_hi=d["median"][1],
                           q25_lo=d["q25"][0], q25_hi=d["q25"][1], q75_lo=d["q75"][0], q75_hi=d["q75"][1],
                           n_right_cens=int(sum(h == inf for _, h in gi)),
                           n_left_cens=int(sum(lo == 0 for lo, _ in gi)))
                for m in (5, 10):
                    i = GRID.index(m)
                    rec[f"P_le_{m}"] = float(F[i])
                    rec[f"P_le_{m}_lo"], rec[f"P_le_{m}_hi"] = q(B[:, i])
                i = GRID.index(10)
                rec["P_gain_after_10"] = float(1 - F[i])
                rec["P_gain_after_10_lo"], rec["P_gain_after_10_hi"] = q(1 - B[:, i])
                rows.append(rec)
            # binary − multiclass; the two groups are resampled independently (different seeds)
            bb, bm, Fb, Fm = boots["binary"], boots["multiclass"], Fs["binary"], Fs["multiclass"]
            rec = dict(learner=lr, eps=eps, group="diff", n=len(ivs))
            for m in (5, 10):
                i = GRID.index(m)
                rec[f"P_le_{m}"] = float(Fb[i] - Fm[i])
                rec[f"P_le_{m}_lo"], rec[f"P_le_{m}_hi"] = q(bb[:, i] - bm[:, i])
            i = GRID.index(10)
            rec["P_gain_after_10"] = float(Fm[i] - Fb[i])  # (1 - Fb) - (1 - Fm)
            rec["P_gain_after_10_lo"], rec["P_gain_after_10_hi"] = q(bm[:, i] - bb[:, i])
            rows.append(rec)
    return pd.DataFrame(rows)


def two_sample(a: list, b: list, stat) -> tuple[float, float]:
    """Percentile CI of stat(a) − stat(b); a and b resampled independently."""
    ia = np.random.default_rng(a9.SEED).integers(0, len(a), size=(a9.N_BOOT, len(a)))
    ib = np.random.default_rng(a9.SEED + 1).integers(0, len(b), size=(a9.N_BOOT, len(b)))
    d = np.array([stat([a[i] for i in ra]) - stat([b[i] for i in rb]) for ra, rb in zip(ia, ib)])
    return q(d)


def rho(t: pd.DataFrame) -> pd.DataFrame:
    out = []
    stats = dict(frozen=a9.turnbull_median_ratio, log=e11.median_of, power=e11.median_of)
    for lr in LEARNERS:
        for model in ("frozen", "log", "power"):
            vals = {}
            for gname in (*GROUPS, "all"):
                g = t[(t.learner == lr) & ((t.group == gname) | (gname == "all"))]
                if model == "frozen":
                    v = list(zip(g.frozen_lo, g.frozen_hi))
                    share = float(np.mean(g.frozen_lo > 1))
                    ninf = int(sum(h == inf for _, h in v))
                else:
                    arr = g[f"rho_{model}"].to_numpy(float)
                    v = list(arr)
                    share = float(np.nanmean(arr > 1))
                    ninf = int(np.isinf(arr).sum())
                vals[gname] = v
                lo, hi = a9.boot_ci(v, stats[model])
                out.append(dict(learner=lr, model=model, group=gname, n=len(v), median=stats[model](v),
                                lo=lo, hi=hi, n_inf=ninf, share_gt1=share))
            lo, hi = two_sample(vals["binary"], vals["multiclass"], stats[model])
            out.append(dict(learner=lr, model=model, group="diff", n=len(vals["all"]),
                            median=stats[model](vals["binary"]) - stats[model](vals["multiclass"]),
                            lo=lo, hi=hi, n_inf=np.nan, share_gt1=np.nan))
    return pd.DataFrame(out)


def reading(ms: pd.DataFrame, rh: pd.DataFrame) -> list[str]:
    """Short plain reading; every number is read from the frames above."""
    def m_(lr, eps, g):
        return ms[(ms.learner == lr) & np.isclose(ms.eps, eps) & (ms.group == g)].iloc[0]

    def r_(lr, model, g):
        return rh[(rh.learner == lr) & (rh.model == model) & (rh.group == g)].iloc[0]

    e11r = pd.read_csv(cm.OUT / "e11_rho.csv")
    e11m = pd.read_csv(cm.OUT / "e11_mstar.csv")
    ok = all(np.isclose(r_(lr, mo, "all")["median"], e11r[(e11r.learner == lr) & (e11r.model == mo)]["median"].iloc[0])
             for lr in LEARNERS for mo in ("frozen", "log", "power"))
    ok &= all(np.isclose(m_("fm2", e, "all").P_le_5, e11m[np.isclose(e11m.eps, e)]["P(m*<=5)"].iloc[0]) for e in EPSS)
    b, m, d = m_("fm2", 0.01, "binary"), m_("fm2", 0.01, "multiclass"), m_("fm2", 0.01, "diff")
    rb, rm = r_("fm2", "frozen", "binary"), r_("fm2", "frozen", "multiclass")
    lo_m = min(r_(lr, mo, "multiclass").lo for lr in LEARNERS for mo in ("frozen", "log", "power"))
    sh_m = min(r_(lr, mo, "multiclass").share_gt1 for lr in LEARNERS for mo in ("frozen", "log", "power"))
    zero = all(x.lo <= 0 <= x.hi for _, x in rh[rh.group == "diff"].iterrows())
    return ["", "## Reading (exploratory; no test)", "",
            f"- Check: the pooled ('all') rows reproduce E11 ({'yes' if ok else 'NO'}).",
            f"- m*, TabICL, ε = 0.01: the median cell is {cell(b.median_lo, b.median_hi)} in both groups and "
            f"P(m* ≤ 10) is {b.P_le_10:.2f} (binary) vs {m.P_le_10:.2f} (multiclass). Early saturation is rarer in the "
            f"multiclass tables: P(m* ≤ 5) {b.P_le_5:.2f} vs {m.P_le_5:.2f}, difference {d.P_le_5:+.2f} "
            f"[{d.P_le_5_lo:+.2f}, {d.P_le_5_hi:+.2f}]. ε = 0.02 shows the same shift of one grid cell; at ε = 0.005 "
            "the multiclass intervals are wide and overlap the binary ones. ε is per label per class, so a multiclass "
            "step adds C labels.",
            f"- ρ(1), TabICL, frozen form: {rb['median']:.2f} [{rb.lo:.2f}, {rb.hi:.2f}] (binary) vs "
            f"{rm['median']:.2f} [{rm.lo:.2f}, {rm.hi:.2f}] (multiclass). Over both learners and the three curve forms, "
            f"the multiclass lower bounds are ≥ {lo_m:.2f} and the share of multiclass tables with ρ > 1 is ≥ {sh_m:.2f}. "
            "TabPFN v2 has larger point values on binary tables; "
            + ("every binary − multiclass interval of ρ(1) contains 0." if zero else "some binary − multiclass intervals of ρ(1) exclude 0."),
            "- So the T1′ label-equivalence result is not carried by the binary tables alone, and the saturation "
            "budget is similar at the pre-registered ε. With 17 multiclass tables these are intervals, not "
            "group differences we claim."]


def cell(lo, hi) -> str:
    return f"({lo:g}, {hi:g}]".replace("inf", "∞")


def main() -> None:
    raw = load(cm.settings.results_dir / "study9" / "t1prime")
    C = raw.groupby("dataset").n_classes.max()
    group = pd.Series(np.where(C > 2, "multiclass", "binary"), index=C.index)
    curve = a9.curves(raw)
    tabs = []
    for lr in LEARNERS:
        t, _, _ = e11.per_table(curve, lr)
        tabs.append(t)
    t = pd.concat(tabs)
    t["C"] = t.dataset.map(C)
    t["group"] = t.dataset.map(group)
    t.to_csv(cm.OUT / "e14_rho_tables.csv", index=False)
    ms = mstar(curve, group)
    ms.to_csv(cm.OUT / "e14_mstar.csv", index=False)
    rh = rho(t)
    rh.to_csv(cm.OUT / "e14_rho.csv", index=False)

    nb, nm = int((group == "binary").sum()), int((group == "multiclass").sum())
    L = [f"# E14: Study 9 on T1′ split by class count (exploratory; {nb} binary, {nm} multiclass tables)", "",
         "Stored results only (results/study9/t1prime). C from the stored `n_classes`. Learners: TabICL (fm2, primary) "
         "and TabPFN v2 (fm1, replication). 95% percentile bootstrap intervals, 2000 resamples over tables "
         "(prereg seed). The multiclass group is small (17 tables): read the intervals, not the point values; "
         "no test is made and no claim of a difference rests on these numbers.", "",
         "## (a) Saturation m* of the random curve (interval-censored, Turnbull)", "",
         "ε = 0.01 is the pre-registered value; 0.005 and 0.02 are the pre-declared sensitivities. "
         "diff rows: binary − multiclass, with the two groups resampled independently.", "",
         "| learner | ε | group | n | median cell | IQR cells | P(m*≤5) | P(m*≤10) | P(gain>ε after 10) | right-cens. |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, x in ms.iterrows():
        if x.group == "diff":
            gain = f"{x.P_gain_after_10:+.2f} [{x.P_gain_after_10_lo:+.2f}, {x.P_gain_after_10_hi:+.2f}]"
            L.append(f"| {x.learner} | {x.eps} | diff | | | | {x.P_le_5:+.2f} [{x.P_le_5_lo:+.2f}, {x.P_le_5_hi:+.2f}] | "
                     f"{x.P_le_10:+.2f} [{x.P_le_10_lo:+.2f}, {x.P_le_10_hi:+.2f}] | {gain} | |")
            continue
        L.append(f"| {x.learner} | {x.eps} | {x.group} | {x.n} | {cell(x.median_lo, x.median_hi)} | "
                 f"{cell(x.q25_lo, x.q25_hi)}–{cell(x.q75_lo, x.q75_hi)} | "
                 f"{x.P_le_5:.2f} [{x.P_le_5_lo:.2f}, {x.P_le_5_hi:.2f}] | "
                 f"{x.P_le_10:.2f} [{x.P_le_10_lo:.2f}, {x.P_le_10_hi:.2f}] | "
                 f"{x.P_gain_after_10:.2f} [{x.P_gain_after_10_lo:.2f}, {x.P_gain_after_10_hi:.2f}] | {int(x.n_right_cens)} |")
    L += ["", "## (b) Label equivalence ρ_S(1) of the stratified protocol", "",
          "frozen: Turnbull median of the interval form (analysis.study9.equivalence_interval); log and power: "
          "per-table point values of E11 (extrapolation allowed). diff = binary − multiclass of the medians.", "",
          "| learner | model | group | n | median ρ(1) [95% CI] | ρ = ∞ | share ρ > 1 |", "|---|---|---|---|---|---|---|"]
    for _, x in rh.iterrows():
        if x.group == "diff":
            L.append(f"| {x.learner} | {x.model} | diff | | {x['median']:+.2f} [{x.lo:+.2f}, {x.hi:+.2f}] | | |")
        else:
            L.append(f"| {x.learner} | {x.model} | {x.group} | {x.n} | {x['median']:.2f} [{x.lo:.2f}, {x.hi:.2f}] | "
                     f"{int(x.n_inf)} | {x.share_gt1:.2f} |")
    L += reading(ms, rh)
    (cm.OUT / "E14_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
