"""B1 (EXPLORATORY): accuracy vs pool imbalance with seven FMs.

Re-runs, with the original code, (i) review_reanalysis/acc_imb_dose.py ``accuracy_scoped``
(T1', Study 12 draws, S-R accuracy by TV split and Spearman rho with TV; Holm within FM over
the 4 budgets, unchanged), (ii) e6_accuracy_heterogeneity.py (DEV, Study 7 Part (a) draws; the
"mean" row becomes the mean over the seven FMs) and (iii) review_reanalysis/r4_checks.py part (3)
(T1' binary/multiclass split). All seven FMs use the same draws on each set (Study 7 Part (a)
on DEV, Study 12 on T1'). Run "4" must reproduce rv_accdose.tex / rv_r4.tex / exploratory_numbers.tex.
Usage: nice -n 15 .venv/bin/python paper/p1/exploratory/sevenfm/b1_acc_imb.py {4|7}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

S, FMS, OUT = H.setup("b1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import rv_common  # noqa: E402
import common as cm  # noqa: E402

RV_OUT = H.EXPL / "review_reanalysis" / "out"
EX_OUT = H.EXPL / "out"
rv_common.OUT = OUT
rv_common.FMS = FMS
rv_common.FMW.update(H.FMW_X)
cm.FMS = FMS
cm.FM_NAMES.update(H.NAMES_X)
cm.OUT = OUT

import acc_imb_dose as aid  # noqa: E402
import r4_checks as r4  # noqa: E402
import bc_contrasts as bc  # noqa: E402
import e6_accuracy_heterogeneity as e6  # noqa: E402
import make_exploratory_numbers as men  # noqa: E402

for mod in (aid, r4, bc):
    mod.OUT, mod.FMS = OUT, FMS
H.copy_inputs(RV_OUT, ["bc_t1_pool_tv.csv"], OUT)
(OUT.parents[1] / "out").mkdir(exist_ok=True)
H.copy_inputs(EX_OUT, ["e12_slopes.csv"], OUT.parents[1] / "out")
men.OUT = OUT
if S == "7":
    men.FT.update(H.FMW_X)


def r4_part3_macros(ab: pd.DataFrame) -> list[str]:
    """Verbatim copy of the part-(3) macro block of r4_checks.main (bin_imb, mc_imb)."""
    mac = rv_common.Macros("r4p3")
    src = "r4_checks.py"
    bi = ab[ab.group == "bin_imb"]
    mac.put("RfAccBinImbN", str(int(bi.n.iloc[0])), f"{src} (3): binary T1' tables with TV >= 0.1 (post hoc)")
    mac.put("RfAccBinImbMin", rv_common.fmt(bi["mean"].min(), 3, True), f"{src} (3): min over FMs, accuracy S-R, m-averaged")
    mac.put("RfAccBinImbMax", rv_common.fmt(bi["mean"].max(), 3, True), f"{src} (3): max over FMs")
    mac.put("RfAccBinImbPMax", rv_common.pfmt(bi.p_wilcoxon.max()), f"{src} (3): max over FMs of Wilcoxon p (unadjusted)")
    mi = ab[ab.group == "mc_imb"]
    mac.put("RfAccMcImbN", str(int(mi.n.iloc[0])), f"{src} (3): multiclass T1' tables with TV >= 0.1")
    mac.put("RfAccMcImbNSig", str(int((mi.p_wilcoxon < 0.05).sum())), f"{src} (3): FMs with Wilcoxon p < 0.05 on them")
    return mac.rows


def main() -> None:
    # (i) T1' scoped accuracy + rho (acc_imb_dose.main writes acc_imb*.csv and rv_accdose.tex)
    aid.main()
    rv_lines = (OUT / "rv_accdose.tex").read_text().splitlines()
    # (iii) r4 part (3)
    ab = r4.acc_binary()
    ab.to_csv(OUT / "r4_accbin.csv", index=False)
    rv_lines_r4 = r4_part3_macros(ab)
    # (ii) E6 on DEV
    c = e6.cells()
    c.to_csv(OUT / "e6_cells.csv", index=False)
    a = e6.assoc(c)
    a.to_csv(OUT / "e6_assoc.csv", index=False)
    th = e6.threshold(c)
    th.to_csv(OUT / "e6_threshold.csv", index=False)
    men.MAC.clear(), men.SRC.clear()
    men.e6()
    e6_lines = [f"\\newcommand{{\\{k}}}{{{v}}} % {men.SRC[k]}" for k, v in sorted(men.MAC.items())]

    if S == "4":
        bad = []
        for f in ("acc_imb.csv", "acc_imb_rho.csv", "r4_accbin.csv"):
            bad += [f] if not H.verify_csv(OUT / f, RV_OUT / f) else []
        for f in ("e6_cells.csv", "e6_assoc.csv", "e6_threshold.csv"):
            bad += [f] if not H.verify_csv(OUT / f, EX_OUT / f) else []
        acc_lines = [ln for ln in rv_lines if "TpAccImb" in ln]
        bad += H.verify_lines(acc_lines, RV_OUT / "rv_numbers.tex", "acc_imb (rv_numbers)")
        bad += H.verify_lines(rv_lines, RV_OUT / "rv_accdose.tex", "rv_accdose.tex (incl. dose part)")
        bad += H.verify_lines(rv_lines_r4, RV_OUT / "rv_numbers.tex", "r4 part 3")
        bad += H.verify_lines(e6_lines, EX_OUT / "exploratory_numbers.tex", "E6")
        print("VERIFY B1:", "PASS" if not bad else f"FAIL {bad}")
        return
    note = "B1, seven FMs (fm1-fm3, fm5 pre-registered; fm6 TabDPT, fm7 Mitra seeded, fm8 LimiX exploratory); same draws"
    keep = [ln for ln in rv_lines + rv_lines_r4 if ("TpAccImb" in ln or "RfAcc" in ln) and H.keep7(H.macro_name(ln))]
    keep = [ln.replace("cells of 16", "cells of 28") for ln in keep]
    keep += [ln for ln in e6_lines if H.keep7(H.macro_name(ln))]
    lines = [H.rename(ln, note) for ln in keep]
    H.write_tex(H.HERE / "out" / "sevenfm_b1.tex", lines, "B1 accuracy-imbalance, seven FMs")
    print(len(lines), "macros")


if __name__ == "__main__":
    main()
