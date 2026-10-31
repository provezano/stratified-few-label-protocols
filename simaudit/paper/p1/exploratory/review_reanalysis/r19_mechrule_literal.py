"""Blind review 4, M1: the pre-registered Study 7 mechanism rule, applied LITERALLY (no fits; read-only inputs).

Pre-registration (prereg/STUDY7_PROTOCOL_MECHANISM.md, Sec. 3, "Decision rule"), verbatim:
  mechanism 2 is "beyond a prior shift" for an FM if the corrected trend is still one-sided significant
  (stratified JT, p < 0.05, both arms) and its mean drop from s = 0.5 to the largest feasible level is at
  least half of the uncorrected drop; it is "a prior shift" if the corrected drop is TOST-equivalent to 0
  (margin 0.01); otherwise "unresolved".

Literal implementation (the same as the frozen reports/study7_report.py, mechanism()):
  - stratified JT as pre-registered (Sec. 4): strata (dataset, mult), groups = feasible shares in
    increasing order, the reps as observations; one-sided normal p for a decrease; on pc_macro_f1;
  - drop per (dataset, mult) = mean at the first minus mean at the last feasible share; the rule compares
    the mean corrected drop with half the mean uncorrected drop (point estimates);
  - TOST (margin 0.01) on the per-(dataset, mult) corrected drops;
  - a cell is (FM, arm); the FM verdict needs "both arms": beyond if both cells meet the beyond condition,
    a prior shift if both corrected drops are TOST-equivalent, otherwise unresolved.
  Only the three pre-registered verdict words are used.
Inputs: results/study7/c/*_{fm}.parquet (stored pre-registered outputs) for fm1/fm2/fm3/fm5, and
paper/p1/exploratory/fm_extra/out/dose_c/*_{fm}.parquet for fm6 TabDPT / fm7 Mitra-v2 / fm8 LimiX-16M.
The same rule applied to fm6-fm8 is EXPLORATORY (these FMs were not pre-registered); their macros say so.
Writes out/r19_rule.csv, out/rv_r19.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r19_mechrule_literal.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rv_common import FMS, FMW, HERE, OUT, Macros, fmt, pfmt
from simaudit.analysis import study7 as a7
from simaudit.settings import settings

XFMS = ("fm6", "fm7", "fm8")
XFMW = dict(fm6="FMsix", fm7="FMseven", fm8="FMeight")
XIN = HERE.parent / "fm_extra" / "out" / "dose_c"
WORD = {"beyond": "beyond a prior shift", "prior": "a prior shift", "unresolved": "unresolved"}


COLS = ["learner", "dataset", "arm", "share", "mult", "rep", "feasible", "macro_f1", "pc_macro_f1"]


def load() -> pd.DataFrame:
    fr = [pd.read_parquet(p, columns=COLS)
          for fm in FMS for p in sorted((settings.results_dir / "study7" / "c").glob(f"*_{fm}.parquet"))]
    for fm in XFMS:
        files = sorted(XIN.glob(f"*_{fm}.parquet"))
        assert len(files) == 20, (fm, len(files))  # stopping rule of the fm_extra plan: all 20 DEV tables
        fr += [pd.read_parquet(p, columns=COLS) for p in files]
    c = pd.concat(fr, ignore_index=True)
    return c[c.feasible]


def drops(g: pd.DataFrame, met: str) -> np.ndarray:
    out = []
    for _, h in g.groupby(["dataset", "mult"]):
        m = h.groupby("share")[met].mean().sort_index()
        if len(m) >= 2:
            out.append(m.iloc[0] - m.iloc[-1])
    return np.array(out)


def strata(g: pd.DataFrame, met: str):
    return [[gg[met].to_numpy(float) for _, gg in sorted(h.groupby("share"))]
            for _, h in g.groupby(["dataset", "mult"])]


def main():
    c = load()
    rows = []
    for fm in FMS + XFMS:
        for arm in ("majority", "minority"):
            g = c[(c.learner == fm) & (c.arm == arm)]
            raw, pc = drops(g, "macro_f1"), drops(g, "pc_macro_f1")
            jt = a7.jt_stratified(strata(g, "pc_macro_f1"))
            tost = a7.tost_p(pc)
            beyond = bool(jt["p"] < 0.05 and raw.mean() > 0 and pc.mean() >= 0.5 * raw.mean())
            rows.append(dict(fm=fm, arm=arm, raw_drop=raw.mean(), pc_drop=pc.mean(), ratio=pc.mean() / raw.mean(),
                             pc_jt_p=jt["p"], pc_jt_z=jt["z"], pc_tost_p=tost, cell_beyond=beyond,
                             cell_prior=bool(tost < 0.05)))
    r = pd.DataFrame(rows)
    ver = {}
    for fm, g in r.groupby("fm"):
        ver[fm] = "beyond" if g.cell_beyond.all() else ("prior" if g.cell_prior.all() else "unresolved")
    r["fm_verdict"] = r.fm.map(ver)
    r.to_csv(OUT / "r19_rule.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.round(4).to_string())
    print("literal per-FM verdicts:", ver)

    mac = Macros("r19")
    src = "r19_mechrule_literal.py (prereg Study 7 Sec. 3 rule, literal; stored outputs, no fits)"
    for fm in FMS:
        mac.put(f"LitRule{FMW[fm]}", WORD[ver[fm]], f"{src}: {fm} verdict over both arms")
        for a in r[r.fm == fm].itertuples():
            A = a.arm.capitalize()
            mac.put(f"LitRule{FMW[fm]}{A}Ratio", fmt(a.ratio, 2), f"{src}: {fm} {a.arm} mean corrected / uncorrected drop")
            mac.put(f"LitRule{FMW[fm]}{A}RatioThree", fmt(a.ratio, 3), f"{src}: {fm} {a.arm} same ratio, 3 decimals")
            mac.put(f"LitRule{FMW[fm]}{A}JtP", pfmt(max(a.pc_jt_p, 1e-300)) if a.pc_jt_p > 0 else "$<10^{-300}$",
                    f"{src}: {fm} {a.arm} one-sided stratified JT p on the corrected macro-F1")
    vs = [ver[fm] for fm in FMS]
    for k, w in (("beyond", "Beyond"), ("prior", "Prior"), ("unresolved", "Unresolved")):
        mac.put(f"LitRuleN{w}", str(vs.count(k)), f"{src}: number of the four FMs with verdict '{WORD[k]}'")
    mac.put("LitRuleCellsBeyond", str(int(r[r.fm.isin(FMS)].cell_beyond.sum())), f"{src}: of 8 (FM x arm) cells meeting the beyond condition")
    r4 = r[r.fm.isin(FMS)]
    mac.put("LitRuleJtPMax", pfmt(max(r4.pc_jt_p.max(), 1e-300)) if r4.pc_jt_p.max() > 0 else "$<10^{-300}$",
            f"{src}: largest corrected JT p over the 8 cells")
    xsrc = src.replace("rule, literal;", "rule, literal, applied to a NON-pre-registered FM;")
    xrows = []
    for fm in XFMS:
        xrows.append(f"\\newcommand{{\\ExRvLitRule{XFMW[fm]}}}{{{WORD[ver[fm]]}}} % EXPLORATORY; source: {xsrc}: {fm} verdict over both arms")
        for a in r[r.fm == fm].itertuples():
            A = a.arm.capitalize()
            xrows.append(f"\\newcommand{{\\ExRvLitRule{XFMW[fm]}{A}Ratio}}{{{fmt(a.ratio, 2)}}} % EXPLORATORY; source: {xsrc}: {fm} {a.arm} corrected / uncorrected drop")
    xv = [ver[fm] for fm in XFMS]
    for k, w in (("beyond", "Beyond"), ("prior", "Prior"), ("unresolved", "Unresolved")):
        xrows.append(f"\\newcommand{{\\ExRvLitRuleXN{w}}}{{{xv.count(k)}}} % EXPLORATORY; source: {xsrc}: of the three exploratory FMs, verdict '{WORD[k]}'")
    # these values apply a pre-registered rule to pre-registered outputs: label them so, not EXPLORATORY
    mac.rows = [r.replace("% EXPLORATORY; source:", "% PRE-REGISTERED rule (Study 7 Sec. 3), literal recomputation; source:")
                for r in mac.rows] + xrows
    mac.write()


if __name__ == "__main__":
    main()
