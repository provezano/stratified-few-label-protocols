"""E12 (exploratory): small checks requested by R3.

(a) R3-Q4. Recompute the 15 confirmatory contrasts of Study 7 (prereg Sec. 4.1) from
    the stored Parts (a), (b), (c) with the frozen functions (analysis.study7), report
    raw p, the Wilcoxon method actually used by scipy (exact vs normal approximation,
    zeros/ties), the Holm family (size 15) and whether 8.4e-4 is a floor.
(b) R3-W11. Prop. 1(v) needs p ≥ 1/2, p = share of the seen class in the evaluation
    set U. Over the stored Study 7 random draws on the 14 binary DEV tables that miss a
    class: share with p ≥ 1/2, and the distribution of the extra macro-F1 loss p²/(1+p).
(c) R3 (JT p of order 1e-74). Table-level paired dose-response effects (Part (c)):
    per table, FM and arm, the slope of macro-F1 on the share s (OLS on rep-level
    values, per m, then mean over the feasible m), and the drop from s = 0.5 to the
    largest feasible s (mean over m); raw, after p/π̂, and raw − corrected. Mean over
    tables with bootstrap 95% CI (10k), Wilcoxon p, share of tables with a decline.
(d) R3 ("removes about 70% of the reversal"). Definition (make_numbers.py,
    SsevReversalSharePrior): 1 − mean(ii-b)/mean(ii-a), where (ii-a) = TabICL −
    TabPFN v3.5 and (ii-b) = TabICL − TabPFN v3.5 (balanced), random protocol, per table
    mean over m ∈ {3,5,10}. Bootstrap over tables (10k, joint resampling), percentile
    and BCa-free interval; the ratio from unrounded means vs from the rounded report.

Writes out/e12_confirmatory.csv, out/e12_binary_p.csv, out/e12_slopes.csv, out/e12_ratio.csv, out/E12_SUMMARY.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy import stats  # noqa: E402

import common as cm  # noqa: E402
from simaudit.analysis import study7 as a7  # noqa: E402
from simaudit.analysis.report import holm  # noqa: E402

NAMES = dict(fm1="TabPFN v2", fm2="TabICL", fm3="Kumo", fm5="TabPFN v3.5")


def load(part: str) -> pd.DataFrame:
    frames = []
    for p in sorted((cm.S7 / part).glob("*.parquet")):
        if "partial" in p.name:
            continue
        cols = [c for c in pq.read_schema(p).names if c not in ("proba", "labeled", "seen", "counts")]
        frames.append(pd.read_parquet(p, columns=cols))
    return pd.concat(frames, ignore_index=True)


def per_ds(frame, learner, strategy, mults):
    s = frame[(frame.learner == learner) & (frame.strategy == strategy) & frame.mult.isin(mults)]
    return s.groupby(["dataset", "mult"]).macro_f1.mean().groupby("dataset").mean()


def wilcoxon_info(x):
    x = np.asarray(x, float)
    zeros = int((x == 0).sum())
    ties = int(len(x) - len(np.unique(np.abs(x[x != 0]))))
    res = stats.wilcoxon(x)
    exact = stats.wilcoxon(x, method="exact").pvalue if zeros == 0 else np.nan
    approx = stats.wilcoxon(x, method="approx").pvalue
    method = "exact" if (len(x) <= 50 and zeros == 0 and ties == 0) else "approx"
    return dict(p=float(res.pvalue), p_exact=float(exact), p_approx=float(approx), zeros=zeros, ties=ties,
                method_auto=method, W=float(res.statistic), n_neg=int((x < 0).sum()))


def jt_strata(dose, fm, arm, metric="macro_f1"):
    sub = dose[(dose.learner == fm) & (dose.arm == arm) & dose.feasible]
    return [[gg[metric].to_numpy(float) for _, gg in sorted(g.groupby("share"))] for _, g in sub.groupby(["dataset", "mult"])]


def confirmatory(a, b, c):
    rows = []
    for fm in cm.FMS:
        v = per_ds(a, fm, "stratified", (1, 3, 5, 10)) - per_ds(a, fm, "random", (1, 3, 5, 10))
        rows.append(dict(id=f"(i)-{fm}", test="Wilcoxon", mean=v.mean(), **wilcoxon_info(v)))
    ab = pd.concat([a, b])
    for cid, l, r, st in (("(ii-a)", "fm2", "fm5", "random"), ("(ii-b)", "fm2", "fm5_bal", "random"),
                          ("(ii-c)", "fm2", "fm5", "stratified")):
        v = per_ds(ab, l, st, (3, 5, 10)) - per_ds(ab, r, st, (3, 5, 10))
        if cid == "(ii-c)":
            rows.append(dict(id=cid, test="TOST", mean=v.mean(), p=a7.tost_p(v.to_numpy())))
        else:
            rows.append(dict(id=cid, test="Wilcoxon", mean=v.mean(), **wilcoxon_info(v)))
    for fm in cm.FMS:
        for arm in ("majority", "minority"):
            res = a7.jt_stratified(jt_strata(c, fm, arm))
            rows.append(dict(id=f"(iii)-{fm}-{arm}", test="JT", mean=np.nan, p=res["p"], z=res["z"]))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.tolist())
    order = np.argsort(t.p.to_numpy())
    rank = np.empty(len(t), int)
    rank[order] = np.arange(len(t))
    t["holm_multiplier"] = len(t) - rank
    t["p_times_multiplier"] = (t.p * t.holm_multiplier).clip(upper=1)
    return t


def binary_p():
    rows = []
    for did in cm.dev_ids():
        y = cm.dev_data(did).y
        if y.max() != 1:
            continue
        sel = pd.read_parquet(cm.S7 / "selections" / f"{did}_random.parquet")
        for r in sel.itertuples():
            lab = np.asarray(r.labeled, int)
            seen = np.unique(y[lab])
            if len(seen) == 2:
                continue
            mask = np.ones(len(y), bool)
            mask[lab] = False
            p = float(np.mean(y[mask] == seen[0]))
            pi_seen = float(np.mean(y == seen[0]))
            rows.append(dict(dataset=did, mult=int(r.mult), rep=int(r.rep), seen=int(seen[0]),
                             seen_is_majority=pi_seen >= 0.5, p=p, extra_loss=p ** 2 / (1 + p)))
    return pd.DataFrame(rows)


def slopes(c):
    rows = []
    feas = c[c.feasible & c.arm.isin(["majority", "minority"])]
    for (fm, arm, did), g in feas.groupby(["learner", "arm", "dataset"]):
        rec = dict(fm=fm, arm=arm, dataset=did)
        for metric in ("macro_f1", "pc_macro_f1", "accuracy"):
            sl, dr = [], []
            for _, gm in g.groupby("mult"):
                if gm.share.nunique() < 2:
                    continue
                sl.append(a7.slope(gm.share.to_numpy(float), gm[metric].to_numpy(float)))
                mm = gm.groupby("share")[metric].mean()
                dr.append(mm.iloc[0] - mm.iloc[-1])
            rec[f"slope_{metric}"] = float(np.nanmean(sl)) if sl else np.nan
            rec[f"drop_{metric}"] = float(np.mean(dr)) if dr else np.nan
        rows.append(rec)
    t = pd.DataFrame(rows)
    t["slope_diff"] = t.slope_macro_f1 - t.slope_pc_macro_f1
    t["drop_diff"] = t.drop_macro_f1 - t.drop_pc_macro_f1
    out = []
    for (fm, arm), g in t.groupby(["fm", "arm"]):
        for col in ("slope_macro_f1", "slope_pc_macro_f1", "slope_diff", "drop_macro_f1", "drop_pc_macro_f1", "drop_diff",
                    "slope_accuracy"):
            x = g[col].dropna().to_numpy()
            lo, hi = cm.bootstrap_ci(x)
            out.append(dict(fm=fm, arm=arm, quantity=col, n=len(x), mean=x.mean(), ci_lo=lo, ci_hi=hi,
                            p=cm.wilcoxon_p(x), share_neg=float(np.mean(x < 0)), share_pos=float(np.mean(x > 0))))
    return t, pd.DataFrame(out)


def ratio(a, b):
    ab = pd.concat([a, b])
    ia = (per_ds(ab, "fm2", "random", (3, 5, 10)) - per_ds(ab, "fm5", "random", (3, 5, 10))).sort_index()
    ib = (per_ds(ab, "fm2", "random", (3, 5, 10)) - per_ds(ab, "fm5_bal", "random", (3, 5, 10))).reindex(ia.index)
    x, z = ia.to_numpy(), ib.to_numpy()
    est = 1 - z.mean() / x.mean()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(x), size=(10000, len(x)))
    bs = 1 - z[idx].mean(1) / x[idx].mean(1)
    rounded = 1 - round(z.mean(), 3) / round(x.mean(), 3)
    return dict(iia=x.mean(), iib=z.mean(), ratio=est, lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)),
                ratio_rounded=rounded, share_boot_gt_half=float(np.mean(bs > 0.5)),
                per_table_removed_median=float(np.median(1 - z[x > 0] / x[x > 0])))


def main() -> None:
    a, b, c = load("a"), load("b"), load("c")
    a = a[a.strategy.isin(["stratified", "random"])]
    t = confirmatory(a, b, c)
    t.to_csv(cm.OUT / "e12_confirmatory.csv", index=False)
    bp = binary_p()
    bp.to_csv(cm.OUT / "e12_binary_p.csv", index=False)
    st, ss = slopes(c)
    st.to_csv(cm.OUT / "e12_slopes_tables.csv", index=False)
    ss.to_csv(cm.OUT / "e12_slopes.csv", index=False)
    rr = ratio(a, b)
    pd.DataFrame([rr]).to_csv(cm.OUT / "e12_ratio.csv", index=False)
    min_exact = 2 / 2 ** 20
    L = ["# E12: small checks (exploratory, DEV, stored Study 7)", "",
         "## (a) R3-Q4: the confirmatory family (15 contrasts, one Holm family)", "",
         f"scipy {__import__('scipy').__version__}: `wilcoxon` with method 'auto' uses the exact null distribution "
         "for n ≤ 50 without zeros/ties (here n = 20). Smallest attainable exact two-sided p with n = 20: "
         f"2/2^20 = {min_exact:.2e}.", "",
         "| id | test | mean Δ | raw p | method | zeros/ties | negatives | Holm multiplier | p × mult | Holm p |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        meth = r.get("method_auto", "")
        zt = f"{int(r.zeros)}/{int(r.ties)}" if isinstance(meth, str) and meth else ""
        nn = int(r.n_neg) if pd.notna(r.get("n_neg", np.nan)) else ""
        L.append(f"| {r.id} | {r.test} | {r['mean']:+.4f} | {r.p:.3g} | {meth if isinstance(meth, str) else ''} | {zt} | {nn} | "
                 f"{r.holm_multiplier} | {r.p_times_multiplier:.3g} | {r.p_holm:.3g} |")
    fm2 = t[t.id == "(i)-fm2"].iloc[0]
    iia = t[t.id == "(ii-a)"].iloc[0]
    L += ["", f"Reading: 8.4e-4 is not a floor. It is (i)-TabICL's exact p {fm2.p:.3g} × {fm2.holm_multiplier} = "
          f"{fm2.p_times_multiplier:.3g}. (ii-a) has raw p {iia.p:.3g} × {iia.holm_multiplier} = {iia.p_times_multiplier:.3g}, "
          "and Holm's step-down monotonicity raises it to the preceding adjusted value, so both print as 8.4e-4. "
          f"The exact floor for one Wilcoxon test is {min_exact:.1e}, and ×15 = {15 * min_exact:.1e}.", "",
          "## (b) R3-W11: binary missed draws and the p ≥ 1/2 condition of Prop. 1(v)", ""]
    L.append(f"Missed random draws on binary tables: {len(bp)} (20 random draws × 4 budgets × 14 tables = 1120 draws in total). "
             f"Share with p ≥ 1/2: {(bp.p >= 0.5).mean():.3f} ({int((bp.p >= 0.5).sum())}/{len(bp)}); "
             f"seen class is the pool majority in {bp.seen_is_majority.mean():.3f}. "
             f"Extra macro-F1 loss p²/(1+p): median {bp.extra_loss.median():.3f}, min {bp.extra_loss.min():.3f}; "
             f"share ≥ 1/6: {(bp.extra_loss >= 1 / 6 - 1e-12).mean():.3f}.")
    L += ["", "By budget:", "", bp.groupby("mult").agg(n=("p", "size"), share_p_ge_half=("p", lambda v: float((v >= 0.5).mean())),
                                                         min_p=("p", "min")).round(3).to_markdown(), "",
          "## (c) Table-level paired dose-response effects (Part (c))", "",
          "slope = OLS slope of macro-F1 on the share s (per m, mean over feasible m); drop = s = 0.5 minus largest feasible s; "
          "pc = after p/π̂; diff = raw − corrected. Mean over tables [bootstrap 95% CI], Wilcoxon p, share of tables with "
          "a decline (slope < 0 / drop > 0).", "",
          "| FM | arm | n | slope raw | slope pc | slope raw − pc | drop raw | drop pc | share slope<0 (raw) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for fm in cm.FMS:
        for arm in ("majority", "minority"):
            g = ss[(ss.fm == fm) & (ss.arm == arm)].set_index("quantity")
            cell = lambda q: f"{g.loc[q, 'mean']:+.3f} [{g.loc[q, 'ci_lo']:+.3f}, {g.loc[q, 'ci_hi']:+.3f}] (p={g.loc[q, 'p']:.1g})"  # noqa: E731
            L.append(f"| {NAMES[fm]} | {arm} | {int(g.loc['slope_macro_f1', 'n'])} | {cell('slope_macro_f1')} | {cell('slope_pc_macro_f1')} | "
                     f"{cell('slope_diff')} | {cell('drop_macro_f1')} | {cell('drop_pc_macro_f1')} | {g.loc['slope_macro_f1', 'share_neg']:.2f} |")
    L += ["", "## (d) Share of the reversal removed by the balance option", "",
          f"Definition: 1 − mean(ii-b)/mean(ii-a) (make_numbers.py `SsevReversalSharePrior`). Unrounded: (ii-a) {rr['iia']:+.4f}, "
          f"(ii-b) {rr['iib']:+.4f}, ratio {rr['ratio']:.3f}, 95% bootstrap CI over tables [{rr['lo']:.3f}, {rr['hi']:.3f}] "
          f"(10k, joint resampling). From the rounded report values: {rr['ratio_rounded']:.3f}. "
          f"P_boot(ratio > 0.5) = {rr['share_boot_gt_half']:.3f}. Median per-table share removed (tables with (ii-a) > 0): "
          f"{rr['per_table_removed_median']:.3f}."]
    (cm.OUT / "E12_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
