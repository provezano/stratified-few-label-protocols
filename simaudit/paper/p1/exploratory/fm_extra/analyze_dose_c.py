"""EXPLORATORY analysis of Study 7 Part (c) for fm6 TabDPT / fm7 Mitra-v2 / fm8 LimiX-16M (DEV).

Plan: ``dose_plan.md`` Sec. 3 (fixed before the run). The code is the review re-analysis chain,
unchanged, applied to the new FMs as a separate Holm family:

1. ``review_reanalysis/r11_partc_common.one_file`` rescoring (complement ``comp`` and rows common to
   all compositions ``com``; asserts the stored complement metrics are reproduced, < 1e-9);
2. ``chain`` = the body of ``review_reanalysis/r14_mech_chain.main`` (per-table corrected slopes,
   one-sided Wilcoxon and sign test with Holm per scoring over the FM x arm cells, bootstrap CI of
   the mean slope, corrected/raw drop ratio with a table bootstrap, TOST, verdict rule);
3. ``raw_table`` = ``review_reanalysis/acc_imb_dose.dose_table`` (per-table raw and corrected
   slopes, bootstrap CI, two-sided Wilcoxon, Holm per metric over the cells).

``--verify`` reproduces ``r14_mech_chain.csv``, ``r14_verdicts.csv`` and ``dose_table.csv`` from the
stored pre-registered fits (fm1/fm2/fm3/fm5) and fails on any difference > 1e-12. Without it, the
script analyses ``out/dose_c/*_{fm6,fm7,fm8}.parquet``: only FMs with all 20 DEV tables enter
(stopping rule, plan Sec. 5); the Holm family is the cells of those FMs. A 14-cell Holm (with the
8 pre-registered cells of r14) is reported as a sensitivity and never used for a verdict.

Writes ``out/dose_c_chain.csv``, ``out/dose_c_verdicts.csv``, ``out/dose_c_raw.csv`` and
``out/dose_c_numbers.tex`` (macros ``\\ExDoseX...``, all EXPLORATORY).
Run from simaudit/: PYTHONPATH=src nice -n 15 .venv/bin/python
    paper/p1/exploratory/fm_extra/analyze_dose_c.py [--verify]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
RV = HERE.parent / "review_reanalysis"
sys.path.insert(0, str(RV))

import r11_partc_common as r11  # noqa: E402
import r14_mech_chain as r14  # noqa: E402
from rv_common import fmt, holm, pfmt  # noqa: E402
from simaudit.analysis.report import bootstrap_ci  # noqa: E402
from simaudit.analysis.study7 import tost_p  # noqa: E402
from simaudit.settings import settings  # noqa: E402

NEW_FMS = ("fm6", "fm7", "fm8")
FMW = dict(fm6="FMsix", fm7="FMseven", fm8="FMeight")
N_DEV = 20
IN = HERE / "out" / "dose_c"
OUT = HERE / "out"


# ------------------------------------------------------------------ r14 chain (body of r14.main)
def chain(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = d[d.arm.isin(r14.ARMS)]
    rows = []
    for (fm, arm), g in d.groupby(["fm", "arm"]):
        for sc in r14.SC:
            t = r14.per_table(g, sc)
            v = t.slope_pc.to_numpy()
            lo, hi = bootstrap_ci(v, seed=r14.SEED)
            ratio, rlo, rhi = r14.ratio_ci(t, r14.SEED)
            rows.append(dict(fm=fm, arm=arm, scoring=sc, n=len(t), slope_pc=v.mean(), slope_lo=lo, slope_hi=hi,
                             p_wil=float(stats.wilcoxon(v, alternative="less").pvalue),
                             n_neg=int((v < 0).sum()),
                             p_sign=float(stats.binomtest(int((v < 0).sum()), int((v != 0).sum()), 0.5,
                                                          alternative="greater").pvalue),
                             ratio=ratio, ratio_lo=rlo, ratio_hi=rhi,
                             p_tost=tost_p(t.pc_drop.to_numpy())))
    o = pd.DataFrame(rows)
    o["p_wil_holm"] = np.nan
    o["p_sign_holm"] = np.nan
    for sc, g in o.groupby("scoring"):
        o.loc[g.index, "p_wil_holm"] = holm(g.p_wil.to_numpy())
        o.loc[g.index, "p_sign_holm"] = holm(g.p_sign.to_numpy())
    ver = []
    for (fm, sc), g in o.groupby(["fm", "scoring"]):
        trend = bool((g.p_wil_holm < 0.05).all())
        if trend and (g.ratio_lo > r14.THR).all():
            v = "beyond"
        elif (g.p_tost < 0.05).all():
            v = "prior shift"
        elif (g.ratio_hi < r14.THR).any():
            v = "not beyond"
        else:
            v = "inconclusive"
        ver.append(dict(fm=fm, scoring=sc, trend_both=trend, verdict=v,
                        point_rule=("beyond" if (g.ratio >= r14.THR).all() else "not beyond")))
    return o, pd.DataFrame(ver)


# ------------------------------------------------------------------ acc_imb_dose.dose_table
def raw_table(c: pd.DataFrame) -> pd.DataFrame:
    c = c[c.feasible & c.arm.isin(["majority", "minority"])]
    rows = []
    for (fm, arm, did), g in c.groupby(["learner", "arm", "dataset"]):
        rec = dict(fm=fm, arm=arm, dataset=did)
        for met in ("macro_f1", "pc_macro_f1"):
            sl = [float(np.polyfit(gm.share.to_numpy(float), gm[met].to_numpy(float), 1)[0])
                  for _, gm in g.groupby("mult") if gm.share.nunique() >= 2]
            rec[met] = float(np.mean(sl)) if sl else np.nan
        rows.append(rec)
    per = pd.DataFrame(rows)
    out = []
    for (fm, arm), g in per.groupby(["fm", "arm"]):
        for met in ("macro_f1", "pc_macro_f1"):
            x = g[met].dropna().to_numpy()
            lo, hi = bootstrap_ci(x)
            out.append(dict(fm=fm, arm=arm, metric=met, n=len(x), mean=x.mean(), ci_lo=lo, ci_hi=hi,
                            p_wilcoxon=float(stats.wilcoxon(x).pvalue), share_neg=float(np.mean(x < 0))))
    tab = pd.DataFrame(out)
    tab["p_holm"] = np.nan
    for _, g in tab.groupby("metric"):
        tab.loc[g.index, "p_holm"] = holm(g.p_wilcoxon.to_numpy())
    return tab


RAW_COLS = ["learner", "dataset", "arm", "share", "mult", "rep", "feasible", "macro_f1", "pc_macro_f1"]


def rescore(files: list[str]) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=min(3, len(files))) as ex:
        res = list(ex.map(r11.one_file, files))
    err = max(e for _, e in res)
    assert err < 1e-9, err
    return pd.DataFrame([r for rows, _ in res for r in rows])


def _max_diff(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> float:
    m = a.merge(b, on=keys, suffixes=("", "_ref"))
    assert len(m) == len(a) == len(b), (len(m), len(a), len(b))
    num = [c for c in a.columns if c not in keys and pd.api.types.is_numeric_dtype(a[c])]
    return max(float(np.nanmax(np.abs(m[c].to_numpy(float) - m[c + "_ref"].to_numpy(float)))) for c in num)


def verify() -> None:
    """The chain and the raw table reproduce the stored r14 / dose_table outputs exactly."""
    d = pd.read_parquet(RV / "out" / "r11_partc_common.parquet")
    o, vv = chain(d)
    ref = pd.read_csv(RV / "out" / "r14_mech_chain.csv")
    dev = _max_diff(o, ref, ["fm", "arm", "scoring"])
    vref = pd.read_csv(RV / "out" / "r14_verdicts.csv")
    same = vv.merge(vref, on=["fm", "scoring"], suffixes=("", "_ref"))
    assert (same.verdict == same.verdict_ref).all() and (same.point_rule == same.point_rule_ref).all()
    files = sorted(p for fm in ("fm1", "fm2", "fm3", "fm5")
                   for p in (settings.results_dir / "study7" / "c").glob(f"*_{fm}.parquet"))
    c = pd.concat([pd.read_parquet(p, columns=RAW_COLS) for p in files], ignore_index=True)
    tab = raw_table(c).rename(columns=dict(p_holm="p_holm8"))
    tref = pd.read_csv(RV / "out" / "dose_table.csv")
    dev_raw = _max_diff(tab[[*tref.columns]], tref, ["fm", "arm", "metric"])
    print(f"r14 chain max |diff| = {dev:.3g}; verdicts identical; dose_table max |diff| = {dev_raw:.3g}")
    assert dev < 1e-12 and dev_raw < 1e-12


# ------------------------------------------------------------------ main
def complete_fms() -> list[str]:
    ok = []
    for fm in NEW_FMS:
        n = len(list(IN.glob(f"*_{fm}.parquet")))
        print(f"{fm}: {n} of {N_DEV} tables")
        if n == N_DEV:
            ok.append(fm)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        verify()
        return
    fms = complete_fms()
    if not fms:
        raise SystemExit("no FM with all 20 tables: nothing is analysed (plan Sec. 5)")
    files = sorted(str(p) for fm in fms for p in IN.glob(f"*_{fm}.parquet"))
    d = rescore(files)
    o, vv = chain(d)
    # sensitivity: Holm over the 14 cells (8 pre-registered of r14 + the new ones), per scoring
    ref = pd.read_csv(RV / "out" / "r14_mech_chain.csv")
    both = pd.concat([ref.assign(new=False), o.assign(new=True)], ignore_index=True)
    for sc, g in both.groupby("scoring"):
        both.loc[g.index, "p_wil_holm_all"] = holm(g.p_wil.to_numpy())
    o = o.merge(both.loc[both.new, ["fm", "arm", "scoring", "p_wil_holm_all"]], on=["fm", "arm", "scoring"])
    c = pd.concat([pd.read_parquet(f, columns=RAW_COLS) for f in files], ignore_index=True)
    tab = raw_table(c)
    o.to_csv(OUT / "dose_c_chain.csv", index=False)
    vv.to_csv(OUT / "dose_c_verdicts.csv", index=False)
    tab.to_csv(OUT / "dose_c_raw.csv", index=False)
    pd.set_option("display.width", 250)
    print(o.round(4).to_string(index=False))
    print(vv.to_string(index=False))
    print(tab.round(4).to_string(index=False))
    write_macros(o, vv, tab, fms)


def write_macros(o: pd.DataFrame, vv: pd.DataFrame, tab: pd.DataFrame, fms: list[str]) -> None:
    rows = []

    def put(name: str, value: str, src: str) -> None:
        assert name.isalpha(), name
        rows.append(f"\\newcommand{{\\ExDoseX{name}}}{{{value}}} % EXPLORATORY; source: {src}")

    src = "analyze_dose_c.py (Study 7 Part (c), DEV, fm6-fm8; r14 chain, own Holm family"
    W = dict(comp="Comp", com="Com")
    for r in o.itertuples():
        w = f"{FMW[r.fm]}{r.arm.capitalize()}{W[r.scoring]}"
        s = f"{src}; {r.fm} {r.arm} {r.scoring})"
        put(f"{w}Ratio", fmt(r.ratio, 2), s + ": corrected/raw drop")
        put(f"{w}RatioLo", fmt(r.ratio_lo, 2), s + ": bootstrap 95% CI lower, tables resampled")
        put(f"{w}RatioHi", fmt(r.ratio_hi, 2), s + ": bootstrap 95% CI upper")
        put(f"{w}PcSlope", fmt(r.slope_pc, 2, sign=True), s + ": mean per-table corrected slope")
        put(f"{w}PWilHolm", pfmt(r.p_wil_holm), s + f": one-sided Wilcoxon, Holm over {2 * len(fms)} new cells")
        put(f"{w}PTost", fmt(r.p_tost, 2), s + ": TOST p of the per-table corrected drop")
        put(f"{w}NNeg", str(r.n_neg), s + ": tables with a negative corrected slope")
    for r in vv.itertuples():
        put(f"{FMW[r.fm]}{W[r.scoring]}Verdict", r.verdict, f"{src}; {r.fm} {r.scoring}): verdict")
    MT = dict(macro_f1="Raw", pc_macro_f1="Pc")
    for r in tab.itertuples():
        stem = f"Tbl{FMW[r.fm]}{r.arm.capitalize()}{MT[r.metric]}"
        s = f"{src}; acc_imb_dose.dose_table; {r.fm} {r.arm} {r.metric})"
        put(stem + "Slope", fmt(r.mean, 2, sign=True), s + ": mean per-table slope")
        put(stem + "Lo", fmt(r.ci_lo, 2, sign=True), s + ": bootstrap 95% CI lower")
        put(stem + "Hi", fmt(r.ci_hi, 2, sign=True), s + ": bootstrap 95% CI upper")
        put(stem + "Holm", pfmt(r.p_holm), s + f": two-sided Wilcoxon, Holm over {2 * len(fms)} new cells")
    put("NFMs", str(len(fms)), f"{src}): FMs with all 20 DEV tables")
    (OUT / "dose_c_numbers.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
