"""E16 step 2 (exploratory): why the first-order coverage formula overpredicts the multiclass gap.

Stored results only (no FM fits): reads e16_multiclass/out/{draws,tables,classes}.parquet made by
e16_extract.py from results/study9/t1prime and results/study7/a (read-only). Hypotheses H1-H7 of
the E16 brief; definitions in e16_report.md. All intervals: percentile bootstrap over tables
(2000 resamples, seed 20261027). Primary: T1' multiclass (17 tables), TabICL (fm2) first; DEV
multiclass (6 tables) is the replication set.

Run from simaudit/: OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python paper/p1/exploratory/e16_multiclass/e16_analysis.py
Writes out/e16_*.csv and out/E16_SUMMARY.md.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

import e16_common as ec  # noqa: E402

OUT = ec.OUT
MS = (1, 2, 3, 5)


def classes() -> pd.DataFrame:
    path = OUT / "classes.parquet"
    if not path.exists():
        D, T = ec.load()
        ec.per_class(D, T).to_parquet(path, index=False)
    return pd.read_parquet(path)


# ======================================================================= H5 (+H1, luck)
def decompose(L: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    """Exact per-cell split of the stratified-minus-random macro-F1 gap.

    Per class k (R = random draws of the cell, S = stratified draws, f_k^S = mean F1_k under S):
      f_k^S - E_R F1_k = q^_k f_k^S                                  coverage (own F1)
                        + P^(cov) (f_k^S - E[F1_k | cov])             balance
                        + P^(k seen, miss) (f_k^S - E[F1na_k | ..])   redistribution
                        + P^(k seen, miss) (E[F1na_k] - E[F1_k] | ..) absorption
    F1na_k is F1_k with the false positives on rows of unseen classes removed. The four terms add
    to the gap exactly because F1_k = 0 on draws that miss k (L0). Redistribution is further split
    into classes with N_k <= m ("competitor") and N_k > m ("extra labels").
    The first-order prediction MF^S * mean_k q_k (q hypergeometric) is bridged to the coverage
    term by luck = MF^S (mean q - mean q^) and selection = MF^S mean q^ - mean_k q^_k f_k^S.
    """
    tinfo = T.set_index("dataset")
    rows = []
    for (st, ds, fm, m), g in L.groupby(["set", "dataset", "fm", "m"]):
        S, R = g[g.strategy == "stratified"], g[g.strategy == "random"]
        if S.empty or R.empty:
            continue
        C = int(g.C.iloc[0])
        nR = R.draw.nunique()
        fS = S.groupby("k").f1.mean().reindex(range(C)).to_numpy()
        MFS = fS.mean()
        q = ec.q_hyp(tinfo.loc[ds, "counts_pool"], m * C)
        qh, bal, red, red_c, red_l, ab = (np.zeros(C) for _ in range(6))
        for k in range(C):
            Rk = R[R.k == k]
            qh[k] = (~Rk.seen).mean()
            cov = Rk[Rk.covered]
            if len(cov):
                bal[k] = len(cov) / nR * (fS[k] - cov.f1.mean())
            Dk = Rk[Rk.seen & ~Rk.covered]
            if len(Dk):
                p = len(Dk) / nR
                red[k] = p * (fS[k] - Dk.f1_na.mean())
                ab[k] = p * (Dk.f1_na.mean() - Dk.f1.mean())
                lo, hi = Dk[Dk.N <= m], Dk[Dk.N > m]
                # competitor part on the draws where k kept the stratified count; extra-label part
                # = what the draws with more labels add on top (exact: sums to red[k]).
                if len(lo):
                    red_c[k] = p * (fS[k] - lo.f1_na.mean())
                    red_l[k] = red[k] - red_c[k]
                else:
                    red_l[k] = red[k]
        mfr = R.groupby("draw").macro_f1.first()
        cov_own = (qh * fS).mean()
        pred = MFS * q.mean()
        rows.append(dict(set=st, dataset=ds, fm=fm, m=m, C=C, mc=C > 2, kind=tinfo.loc[ds, "kind"], MFS=MFS,
                         MFR=mfr.mean(), obs=MFS - mfr.mean(), pred=pred, pred_own=(q * fS).mean(),
                         luck=pred - MFS * qh.mean(), sel=MFS * qh.mean() - cov_own, cov_own=cov_own,
                         bal=bal.mean(), red=red.mean(), red_comp=red_c.mean(), red_lab=red_l.mean(),
                         absb=ab.mean(), miss_exp=q.sum(), miss_obs=qh.sum(), p_cov=float((R.groupby("draw").covered.first()).mean())))
    X = pd.DataFrame(rows)
    X["spill"] = X.red + X.absb
    X["identity_err"] = X.obs - (X.cov_own + X.bal + X.red + X.absb)
    X["overpred"] = X.pred - X.obs
    return X


def shapley(L: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Model-based two-player Shapley split of the gap into coverage and balance.

    Per draw: c = missed share of classes, b = KL(seen-class context shares || uniform over seen).
    Relative draw gap y = (MF^S_cell - MF_draw)/MF^S_cell is regressed (no intercept; stratified
    draws enter at c = b = 0) on c, b, c*b, per (fm, group, m). With f(c, b) = b1 c + b2 b + b3 c b
    and baseline (0, 0): phi_cov = b1 c + b3 c b / 2, phi_bal = b2 b + b3 c b / 2 (exact Shapley for
    two players), averaged over the random draws and rescaled by MF^S.
    """
    d = L.groupby("draw").agg(set=("set", "first"), dataset=("dataset", "first"), fm=("fm", "first"),
                              strategy=("strategy", "first"), m=("m", "first"), C=("C", "first"),
                              mf=("macro_f1", "first"), n_missed=("n_missed", "first"))
    Ns = L[L.seen].groupby("draw").N.apply(lambda v: np.asarray(v, float))
    d["b"] = [float(np.log(len(v)) + (v / v.sum() * np.log(v / v.sum())).sum()) for v in Ns.reindex(d.index)]
    d["c"] = d.n_missed / d.C
    d = d.merge(X[["set", "dataset", "fm", "m", "MFS", "obs"]], on=["set", "dataset", "fm", "m"])
    d["y"] = (d.MFS - d.mf) / d.MFS
    rows = []
    for (fm, mc, m), g in d.assign(mc=d.C > 2).groupby(["fm", "mc", "m"]):
        if m not in MS:
            continue
        A = np.column_stack([g.c, g.b, g.c * g.b])
        beta, *_ = np.linalg.lstsq(A, g.y, rcond=None)
        r = g[g.strategy == "random"].copy()
        r["phi_cov"] = (beta[0] * r.c + beta[2] * r.c * r.b / 2) * r.MFS
        r["phi_bal"] = (beta[1] * r.b + beta[2] * r.c * r.b / 2) * r.MFS
        per = r.groupby("dataset")[["phi_cov", "phi_bal"]].mean()
        obs = X[(X.fm == fm) & (X.mc == mc) & (X.m == m)].groupby("dataset").obs.mean()
        pc, pb = per.phi_cov.mean(), per.phi_bal.mean()
        rows.append(dict(fm=fm, mc=mc, m=m, tables=len(per), b_cov=beta[0], b_bal=beta[1], b_int=beta[2],
                         phi_cov=pc, phi_bal=pb, fitted=pc + pb, obs=obs.mean(),
                         share_cov=pc / (pc + pb) if pc + pb != 0 else np.nan))
    return pd.DataFrame(rows)


# ======================================================================= H1
def h1(L: pd.DataFrame) -> pd.DataFrame:
    """Per-class F1 under stratified sampling vs class share / rank (multiclass, m = 1)."""
    S = L[(L.strategy == "stratified") & (L.C > 2) & (L.m == 1)]
    pc = S.groupby(["set", "dataset", "fm", "k"]).agg(f1=("f1", "mean"), pi=("pi", "first"),
                                                      rank=("rank", "first"), C=("C", "first")).reset_index()
    pc["f1_c"] = pc.f1 - pc.groupby(["set", "dataset", "fm"]).f1.transform("mean")
    pc["pi_rel"] = pc.pi * pc.C  # share relative to uniform
    pc["rank_rel"] = (pc["rank"] - 1) / (pc.C - 1)  # 0 = largest, 1 = smallest
    rows = []
    for fm, g in pc.groupby("fm"):
        rs = stats.spearmanr(np.log(g.pi_rel), g.f1_c)
        rr = stats.spearmanr(g.rank_rel, g.f1_c)
        small = g[g.pi_rel < 0.5].f1_c
        rows.append(dict(fm=fm, classes=len(g), rho_logshare=rs.statistic, p_logshare=rs.pvalue,
                         rho_rank=rr.statistic, p_rank=rr.pvalue, n_small=len(small),
                         f1c_small=small.mean(), f1c_large=g[g.pi_rel >= 1.5].f1_c.mean()))
    return pd.DataFrame(rows), pc


# ======================================================================= H2
def h2(L: pd.DataFrame) -> pd.DataFrame:
    """Seen-class metrics in random draws with a miss vs the same class under stratified (m = 1)
    and vs covered random draws at the same budget (m in MS)."""
    out = []
    for m in MS:
        Lm = L[(L.C > 2) & (L.m == m)]
        ref_s = Lm[Lm.strategy == "stratified"].groupby(["set", "dataset", "fm", "k"])[["f1", "f1_na", "prec", "rec"]].mean()
        ref_c = Lm[(Lm.strategy == "random") & Lm.covered].groupby(["set", "dataset", "fm", "k"])[["f1", "prec", "rec"]].mean()
        R = Lm[(Lm.strategy == "random") & Lm.seen & ~Lm.covered]
        for grp, sub in (("all", R), ("N<=m", R[R.N <= m]), ("N>m", R[R.N > m])):
            M = sub.groupby(["set", "dataset", "fm", "k"])[["f1", "f1_na", "prec", "rec"]].mean()
            for ref_name, ref in (("stratified", ref_s), ("covered_random", ref_c)):
                J = M.join(ref, rsuffix="_ref", how="inner")
                if J.empty:
                    continue
                J = J.reset_index()
                for c in ("f1", "prec", "rec"):
                    J["d_" + c] = J[c] - J[c + "_ref"]
                J["d_f1_na"] = J.f1_na - J.f1_ref
                for (st, fm), h in J.groupby(["set", "fm"]):
                    rec = dict(set=st, fm=fm, m=m, group=grp, ref=ref_name)
                    b = ec.boot_frame(h, ["d_f1", "d_f1_na", "d_prec", "d_rec"])
                    rec.update(b)
                    out.append(rec)
    return pd.DataFrame(out)


# ======================================================================= H3
def h3(L: pd.DataFrame, D: pd.DataFrame, T: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Where do the rows of a missed class go? (random draws, multiclass, m in MS)."""
    tinfo = T.set_index("dataset")
    mcd = D[(D.C > 2) & D.m.isin(MS)].reset_index()
    # stratified confusion per (set, dataset, fm, m), row-normalised
    strat_cm = {}
    for key, g in mcd[mcd.strategy == "stratified"].groupby(["set", "dataset", "fm", "m"]):
        C = int(g.C.iloc[0])
        cm = np.sum([np.asarray(v, float).reshape(C, C) for v in g.cm], axis=0)
        strat_cm[key] = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    strat_prec = L[(L.strategy == "stratified") & (L.C > 2)].groupby(["set", "dataset", "fm", "m", "k"]).prec.mean()
    ev = []
    for r in mcd[mcd.strategy == "random"].itertuples(index=False):
        C = r.C
        N = np.asarray(r.N)
        seen = np.flatnonzero(N > 0)
        missed = np.flatnonzero(N == 0)
        if len(missed) == 0 or len(seen) < 2:
            continue
        cm = np.asarray(r.cm, float).reshape(C, C)
        cd = np.asarray(tinfo.loc[r.dataset, "cdist"], float).reshape(C, C)
        pool = np.asarray(tinfo.loc[r.dataset, "counts_pool"], float)
        scm = strat_cm.get((r.set, r.dataset, r.fm, r.m))
        ordi = ec.ORDINAL.get(r.dataset)
        pp, tp = cm.sum(0), np.diag(cm)
        fp_abs = cm[missed].sum(0)
        for j in missed:
            a = cm[j, seen]
            if a.sum() == 0:
                continue
            sh = a / a.sum()
            top = seen[np.argmax(a)]
            def hit(score):  # fractional hit when the reference is tied
                best = seen[score == score.max()]
                return 1.0 / len(best) if top in best else 0.0
            # precision loss of the top absorber: with vs without all absorbed rows (from every missed class)
            prec = tp[top] / pp[top] if pp[top] > 0 else np.nan
            pp2 = pp[top] - fp_abs[top]
            prec_na = tp[top] / pp2 if pp2 > 0 else np.nan
            rec = dict(set=r.set, dataset=r.dataset, fm=r.fm, m=r.m, C=C, kind=tinfo.loc[r.dataset, "kind"],
                       n_seen=len(seen), n_missed=len(missed), u_j=a.sum(), top_share=sh.max(),
                       eff_absorbers=float(np.exp(-(sh[sh > 0] * np.log(sh[sh > 0])).sum())),
                       chance=1 / len(seen),
                       hit_centroid=hit(-cd[j, seen]), hit_ctx=hit(N[seen].astype(float)),
                       hit_pool=hit(pool[seen]),
                       hit_confusion=hit(scm[j, seen]) if scm is not None else np.nan,
                       prec_top=prec, prec_top_na=prec_na, prec_top_strat=strat_prec.get((r.set, r.dataset, r.fm, r.m, top), np.nan),
                       share_rows_absorbed_top=a.max() / pp[top] if pp[top] > 0 else np.nan)
            if ordi is not None:
                rk = np.asarray(ordi["rank"])
                dist = np.abs(rk[seen] - rk[j])
                rec["hit_adjacent"] = float(dist[np.argmax(a)] <= 1)
                rec["chance_adjacent"] = float((dist <= 1).mean())
                rec["share_to_adjacent"] = float(sh[dist <= 1].sum())
            ev.append(rec)
    E = pd.DataFrame(ev)
    E["prec_loss_abs"] = E.prec_top_na - E.prec_top
    E["prec_loss_vs_strat"] = E.prec_top_strat - E.prec_top
    E["grp"] = np.where(E.kind.str.startswith("ordinal"), "ordinal", np.where(E.kind == "nominal", "nominal", "unknown"))
    cols = ["top_share", "chance", "eff_absorbers", "n_seen", "hit_centroid", "hit_confusion", "hit_ctx", "hit_pool",
            "prec_loss_abs", "prec_loss_vs_strat", "share_rows_absorbed_top"]
    rows = []
    for (st, fm, m, grp), g in pd.concat([E, E.assign(grp="all")]).groupby(["set", "fm", "m", "grp"]):
        rec = dict(set=st, fm=fm, m=m, grp=grp, events=len(g))
        rec.update(ec.boot_frame(g, cols))
        if grp == "ordinal":
            rec.update(ec.boot_frame(g, ["hit_adjacent", "chance_adjacent", "share_to_adjacent"]))
        rows.append(rec)
    return pd.DataFrame(rows), E


# ======================================================================= H4
def h4(X: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    """Does C_eff / tail mass line the binary and multiclass gaps up at m = 1?"""
    t = T.set_index("dataset")
    Z = X[X.m == 1].copy()
    pis = {k: np.asarray(v, float) / np.sum(v) for k, v in t.counts_pool.items()}
    Z["C_simpson"] = [1 / np.sum(pis[k] ** 2) for k in Z.dataset]
    Z["C_entropy"] = [float(np.exp(-(pis[k] * np.log(pis[k])).sum())) for k in Z.dataset]
    Z["tail_mass"] = [float(pis[k][pis[k] < 0.5 / len(pis[k])].sum()) for k in Z.dataset]
    Z["mean_q"] = Z.miss_exp / Z.C
    Z["rel_obs"] = Z.obs / Z.MFS
    Z["ratio"] = Z.obs / Z.pred
    rows = []
    for fm, g in Z.groupby("fm"):
        for x in ("mean_q", "C", "C_simpson", "C_entropy", "tail_mass"):
            for target in ("rel_obs", "ratio"):
                xv = g[x].to_numpy(float)
                xv = np.log(xv) if x.startswith("C") else xv
                mc = g.mc.to_numpy(float)
                A0 = np.column_stack([np.ones(len(g)), xv])
                A1 = np.column_stack([A0, mc])
                yv = g[target].to_numpy(float)
                b0, *_ = np.linalg.lstsq(A0, yv, rcond=None)
                b1, *_ = np.linalg.lstsq(A1, yv, rcond=None)
                r2 = 1 - ((yv - A0 @ b0) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
                # bootstrap the dummy coefficient over tables
                rng = np.random.default_rng(ec.SEED)
                bs = []
                for _ in range(ec.N_BOOT):
                    i = rng.integers(0, len(g), len(g))
                    if mc[i].min() == mc[i].max():
                        continue
                    bb, *_ = np.linalg.lstsq(A1[i], yv[i], rcond=None)
                    bs.append(bb[2])
                rows.append(dict(fm=fm, x=x, target=target, tables=len(g), r2=r2, slope=b0[1],
                                 mc_offset=b1[2], mc_offset_lo=np.quantile(bs, 0.025), mc_offset_hi=np.quantile(bs, 0.975),
                                 spearman_mc=stats.spearmanr(g[x][g.mc], g[target][g.mc]).statistic))
    Z.to_csv(OUT / "e16_h4_cells.csv", index=False)
    return pd.DataFrame(rows)


# ======================================================================= H6
def h6(L: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Per-draw gap as one function of the context-share divergence, binary vs multiclass."""
    d = L[L.strategy == "random"].groupby("draw").agg(set=("set", "first"), dataset=("dataset", "first"),
                                                       fm=("fm", "first"), m=("m", "first"), C=("C", "first"),
                                                       mf=("macro_f1", "first"))
    Lr = L[L.strategy == "random"]
    P = (Lr.N / Lr.B).to_numpy()
    pi = Lr.pi.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        tu = np.where(P > 0, P * np.log(P * Lr.C.to_numpy()), 0.0)
        tp = np.where(P > 0, P * np.log(P / pi), 0.0)
    kl = pd.DataFrame(dict(draw=Lr.draw.to_numpy(), kl_u=tu, kl_pi=tp)).groupby("draw").sum()
    d = d.join(kl)
    d["kl_u_norm"] = d.kl_u / np.log(d.C)
    d = d.merge(X[["set", "dataset", "fm", "m", "MFS"]], on=["set", "dataset", "fm", "m"])
    d["gap"] = d.MFS - d.mf
    d["rel_gap"] = d.gap / d.MFS
    d["mc"] = d.C > 2
    rows = []
    for fm, g in d.groupby("fm"):
        for xcol in ("kl_u", "kl_u_norm", "kl_pi"):
            for ycol in ("gap", "rel_gap"):
                res = {}
                for fit_on in ("binary", "multiclass", "pooled"):
                    tr = g if fit_on == "pooled" else g[g.mc == (fit_on == "multiclass")]
                    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(tr[xcol], tr[ycol])
                    for ev_on in ("binary", "multiclass"):
                        te = g[g.mc == (ev_on == "multiclass")]
                        pr = iso.predict(te[xcol])
                        sse = ((te[ycol] - pr) ** 2).sum()
                        sst = ((te[ycol] - te[ycol].mean()) ** 2).sum()
                        res[(fit_on, ev_on)] = (1 - sse / sst, float((te[ycol] - pr).mean()))
                rows.append(dict(fm=fm, x=xcol, y=ycol,
                                 r2_mc_own=res[("multiclass", "multiclass")][0], r2_mc_from_bin=res[("binary", "multiclass")][0],
                                 bias_mc_from_bin=res[("binary", "multiclass")][1],
                                 r2_bin_own=res[("binary", "binary")][0], r2_bin_from_mc=res[("multiclass", "binary")][0],
                                 bias_bin_from_mc=res[("multiclass", "binary")][1],
                                 r2_mc_pooled=res[("pooled", "multiclass")][0], r2_bin_pooled=res[("pooled", "binary")][0],
                                 spearman_mc=stats.spearmanr(g[g.mc][xcol], g[g.mc][ycol]).statistic,
                                 spearman_bin=stats.spearmanr(g[~g.mc][xcol], g[~g.mc][ycol]).statistic))
    d.to_parquet(OUT / "e16_h6_draws.parquet", index=False)
    return pd.DataFrame(rows)


# ======================================================================= H7
def h7(L: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Do predicted class shares follow the context shares? (random draws, seen classes).

    Slope of (mean predicted share - true share in U) on (context share - true share), through
    the origin, pooled over draws; 1 = predictions follow the context shares fully, 0 = they track
    the true shares. ``covered``: covered random draws only (no L0 mechanics); ``all_seen``: seen
    classes of every random draw (on draws with a miss part of the slope is mechanical, because
    the seen classes must share the missed class's probability mass).
    """
    base = L[(L.strategy == "random") & L.seen].copy()
    base["ctx"] = base.N / base.B
    base["x"] = base.ctx - base.true_share
    base["y_soft"] = base.soft - base.true_share
    base["y_hard"] = base.hard - base.true_share
    base["mc"] = base.C > 2
    rows = []

    def slope(h, yc):
        return float((h.x * h[yc]).sum() / (h.x ** 2).sum())

    for variant, R in (("covered", base[base.covered]), ("all_seen", base)):
        for (st, fm, mc), g in pd.concat([R, R.assign(set="pooled")]).groupby(["set", "fm", "mc"]):
            grp = {k: v for k, v in g.groupby("dataset")}
            tabs = list(grp)
            rng = np.random.default_rng(ec.SEED)
            bs_s, bs_h = [], []
            for _ in range(ec.N_BOOT):
                h = pd.concat([grp[tabs[i]] for i in rng.integers(0, len(tabs), len(tabs))])
                bs_s.append(slope(h, "y_soft"))
                bs_h.append(slope(h, "y_hard"))
            rows.append(dict(variant=variant, set=st, fm=fm, mc=mc, tables=len(tabs), rows=len(g),
                             slope_soft=slope(g, "y_soft"), slope_soft_lo=np.quantile(bs_s, 0.025),
                             slope_soft_hi=np.quantile(bs_s, 0.975), slope_hard=slope(g, "y_hard"),
                             slope_hard_lo=np.quantile(bs_h, 0.025), slope_hard_hi=np.quantile(bs_h, 0.975)))
    return pd.DataFrame(rows)


def h7_paired(L: pd.DataFrame) -> pd.DataFrame:
    """Paired FM contrasts of the soft context-following slope (covered random draws; the labelled
    sets are identical across FMs), multiclass and binary, both sets pooled; tables resampled jointly."""
    R = L[(L.strategy == "random") & L.seen & L.covered].copy()
    R["x"] = R.N / R.B - R.true_share
    R["y"] = R.soft - R.true_share
    R["mc"] = R.C > 2
    rows = []
    for mc, g in R.groupby("mc"):
        agg = g.assign(xy=g.x * g.y, xx=g.x ** 2).groupby(["dataset", "fm"])[["xy", "xx"]].sum().unstack("fm")
        tabs = agg.index.to_numpy()
        rng = np.random.default_rng(ec.SEED)
        idx = rng.integers(0, len(tabs), size=(ec.N_BOOT, len(tabs)))
        def sl(fm, i=None):
            a = agg["xy"][fm].to_numpy(); b = agg["xx"][fm].to_numpy()
            return a.sum() / b.sum() if i is None else a[i].sum(1) / b[i].sum(1)
        for f in ("fm1", "fm3", "fm5"):
            d = sl(f) - sl("fm2")
            bd = sl(f, idx) - sl("fm2", idx)
            rows.append(dict(mc=mc, contrast=f"{f}-fm2", tables=len(tabs), diff=d,
                             lo=np.quantile(bd, 0.025), hi=np.quantile(bd, 0.975)))
    return pd.DataFrame(rows)


def h3_matched(E: pd.DataFrame) -> pd.DataFrame:
    """Ordinal vs nominal at the same number of seen classes (|S| = 2), m = 1, both sets pooled.

    Adjacent-rank test only on informative events (not every seen class is adjacent)."""
    E1 = E[(E.m == 1) & (E.n_seen == 2) & E.grp.isin(["ordinal", "nominal"])]
    rows = []
    for (fm, grp), g in E1.groupby(["fm", "grp"]):
        rec = dict(fm=fm, grp=grp, events=len(g))
        rec.update(ec.boot_frame(g, ["top_share", "eff_absorbers", "hit_centroid", "hit_confusion", "hit_ctx",
                                     "prec_loss_abs"]))
        if grp == "ordinal":
            inf = E[(E.m == 1) & (E.grp == "ordinal") & (E.chance_adjacent < 1) & (E.fm == fm)]
            rec.update({k + "_inf": v for k, v in ec.boot_frame(inf, ["hit_adjacent", "chance_adjacent",
                                                                        "share_to_adjacent"]).items()})
        rows.append(rec)
    return pd.DataFrame(rows)


def h6_bias(L: pd.DataFrame) -> pd.DataFrame:
    """Cross-group transfer of the isotonic dose curve (rel_gap on KL to the pool prior and to
    uniform): per-table mean residual of the multiclass draws under the binary-fitted curve."""
    d = pd.read_parquet(OUT / "e16_h6_draws.parquet")
    rows = []
    for fm, g in d.groupby("fm"):
        for xcol in ("kl_pi", "kl_u"):
            b, mcd = g[~g.mc], g[g.mc]
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(b[xcol], b.rel_gap)
            res = mcd.assign(res=mcd.rel_gap - iso.predict(mcd[xcol]))
            iso2 = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(mcd[xcol], mcd.rel_gap)
            res2 = b.assign(res=b.rel_gap - iso2.predict(b[xcol]))
            r1 = ec.boot_frame(res, ["res"])
            r2 = ec.boot_frame(res2, ["res"])
            rows.append(dict(fm=fm, x=xcol, mc_resid_from_bin=r1["res"], lo=r1["res_lo"], hi=r1["res_hi"],
                             bin_resid_from_mc=r2["res"], lo2=r2["res_lo"], hi2=r2["res_hi"]))
    return pd.DataFrame(rows)


# ======================================================================= summaries
def agg_decomp(X: pd.DataFrame) -> pd.DataFrame:
    cols = ["obs", "pred", "pred_own", "luck", "sel", "cov_own", "bal", "red", "red_comp", "red_lab", "absb", "spill",
            "overpred", "MFS", "miss_exp", "miss_obs", "identity_err"]
    rows = []
    for (st, mc, fm, m), g in pd.concat([X, X.assign(set="pooled")]).groupby(["set", "mc", "fm", "m"]):
        rec = dict(set=st, group="multiclass" if mc else "binary", fm=fm, m=m)
        rec.update(ec.boot_frame(g, cols))
        rows.append(rec)
    A = pd.DataFrame(rows)
    # shares of the overprediction explained (multiclass): luck, selection, -(balance), -(spillover)
    A["share_luck"] = A.luck / A.overpred
    A["share_sel"] = A.sel / A.overpred
    A["share_bal"] = -A.bal / A.overpred
    A["share_spill"] = -A.spill / A.overpred
    return A


def main():
    D, T = ec.load()
    L = classes()
    X = decompose(L, T)
    assert X.identity_err.abs().max() < 1e-12, X.identity_err.abs().max()
    X.to_csv(OUT / "e16_decomp_cells.csv", index=False)
    A = agg_decomp(X)
    A.to_csv(OUT / "e16_decomp.csv", index=False)
    SH = shapley(L, X)
    SH.to_csv(OUT / "e16_shapley.csv", index=False)
    H1, pc = h1(L)
    H1.to_csv(OUT / "e16_h1.csv", index=False)
    pc.to_csv(OUT / "e16_h1_classes.csv", index=False)
    H2 = h2(L)
    H2.to_csv(OUT / "e16_h2.csv", index=False)
    H3, E = h3(L, D, T)
    H3.to_csv(OUT / "e16_h3.csv", index=False)
    E.to_parquet(OUT / "e16_h3_events.parquet", index=False)
    H3m = h3_matched(E)
    H3m.to_csv(OUT / "e16_h3_matched.csv", index=False)
    H4 = h4(X, T)
    H4.to_csv(OUT / "e16_h4.csv", index=False)
    H6 = h6(L, X)
    H6.to_csv(OUT / "e16_h6.csv", index=False)
    H6b = h6_bias(L)
    H6b.to_csv(OUT / "e16_h6_bias.csv", index=False)
    H7 = h7(L, X)
    H7.to_csv(OUT / "e16_h7.csv", index=False)
    H7p = h7_paired(L)
    H7p.to_csv(OUT / "e16_h7_paired.csv", index=False)
    T[["set", "dataset", "name", "C", "n", "kind"]].to_csv(OUT / "e16_tables.csv", index=False)

    def tab(df, q=None, cols=None, d=3):
        df = df.query(q) if q else df
        return (df[cols] if cols else df).round(d).to_markdown(index=False)

    dc = ["set", "fm", "tables", "obs", "pred", "pred_own", "luck", "sel", "cov_own", "bal", "red", "red_comp", "red_lab", "absb",
          "spill", "overpred", "share_sel", "share_spill", "miss_exp", "miss_obs"]
    Lines = ["# E16: the multiclass protocol gap, decomposed (exploratory; stored results only)", "",
             "Inputs: Study 9 T1' (52 tables, 17 multiclass) and Study 7 Part (a) DEV (20 tables, 6 multiclass); "
             "random and stratified draws, 4 FMs. No new fits. Bootstrap over tables (2000, seed 20261027).", "",
             f"Recomputed macro-F1 from stored probabilities equals the stored value (max abs diff "
             f"{(D.macro_f1 - D.macro_f1_re).abs().max():.1e}); decomposition identity error max "
             f"{X.identity_err.abs().max():.1e}.", "",
             "## H5 exact decomposition (m = 1), multiclass", "",
             tab(A, "group == 'multiclass' and m == 1", dc), "",
             "## H5 exact decomposition (m = 1), binary", "", tab(A, "group == 'binary' and m == 1", dc), "",
             "## H5 by budget (T1', multiclass)", "", tab(A, "group == 'multiclass' and set == 't1prime'", ["fm", "m"] + dc[2:]), "",
             "## H5 Shapley coverage vs balance (model-based)", "", tab(SH), "",
             "## H1 per-class F1 vs class share (stratified, m = 1, multiclass, pooled sets)", "", tab(H1), "",
             "## H2 seen classes on draws with a miss", "",
             tab(H2, "m == 1", ["set", "fm", "group", "ref", "tables", "d_f1", "d_f1_lo", "d_f1_hi", "d_f1_na", "d_prec", "d_rec"]), "",
             tab(H2, "m > 1 and ref == 'covered_random' and set == 't1prime'", ["fm", "m", "group", "tables", "d_f1", "d_f1_lo", "d_f1_hi", "d_prec", "d_rec"]), "",
             "## H3 absorption", "", tab(H3, "m == 1"), "", tab(H3, "m > 1 and grp == 'all'"), "",
             "Matched |S| = 2 (m = 1, both sets):", "", tab(H3m), "",
             "## H4 effective number of classes (m = 1)", "", tab(H4), "",
             "## H6 one dose curve (isotonic)", "", tab(H6), "", tab(H6b), "",
             "## H7 context following", "", tab(H7), "", "Paired FM contrasts (covered draws):", "", tab(H7p), ""]
    (OUT / "E16_SUMMARY.md").write_text("\n".join(Lines))
    print("\n".join(Lines))


if __name__ == "__main__":
    main()
