"""Write study17_numbers.tex from the stored Study 17 results (pre-registered; analysed once).

Reads results/study17/study17_results.json (written once by tools/study17/analysis.py) and the frozen
frame data/pool_study17.yaml. Nothing is recomputed: the script only formats stored values, counts
verdicts and applies the frame-level rules of the pre-registration (Sec. 4.2) to the stored verdicts.

    .venv/bin/python paper/p1/make_study17_numbers.py [--src JSON] [--pool YAML] [--out TEX]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/study17/study17_results.json"
POOL = ROOT / "data/pool_study17.yaml"
OUT = Path(__file__).with_name("study17_numbers.tex")
FMS = ("fm1", "fm2", "fm3", "fm5")
NAME = {"fm1": "One", "fm2": "Two", "fm3": "Three", "fm5": "Five"}
LABEL = {"fm1": "TabPFN v2", "fm2": "TabICLv2", "fm3": "Kumo-Tabular", "fm5": "TabPFN v3.5"}
# design constants (not results): prereg Sec. 3.2 and decision O4, Sec. 10.1 (margin); Sec. 4.1 (loss limit)
MARGIN = 0.02
# JSON key -> macro tag. Confirmatory: P, E, Q, L1. Pre-declared secondary (Holm over 4 each): the rest.
FAMILIES = {
    "P": ("P",), "E": ("E",), "Q": ("Q",), "L1": ("LOne",),
    "L1_coverage_gap_S_minus_R": ("LOneCov",),
    "coverage_part_RC_minus_R": ("Cov",),
    "unconditional_S_minus_R": ("COne",),
}
SECONDARY = {"balanced_accuracy": "BA", "accuracy": "Acc"}
SHORT = {"replicated": "rep.", "reversed": "rev.", "equivalent": "eq.", "not established": "n.e."}  # table legend
SUBSETS = {"post_fm5": "Safe", "imbalanced_binary": "ImbBin", "multiclass": "Multi", "binary": "Bin",
           "sens_flag": "SensFlag"}


def num(x: float, digits: int = 3) -> str:
    """Signed number as LaTeX math (as simaudit.analysis.study15.fmt_num)."""
    return f"${float(x):+.{digits}f}$"


def pval(p: float) -> str:
    """Three decimals, or a power of ten below 0.001 (as simaudit.analysis.study15.fmt_p)."""
    p = float(p)
    if p >= 0.001:
        return f"${p:.3f}$"
    mant, exp = f"{p:.1e}".split("e")
    return f"${mant}\\times 10^{{{int(exp)}}}$"


def ci(c: list[float]) -> str:
    return f"[{num(c[0])}, {num(c[1])}]"


def pct(x: float) -> str:
    return f"{100 * float(x):.0f}\\%"


def names(fms: list[str]) -> str:
    lab = [LABEL[f] for f in fms]
    return lab[0] if len(lab) == 1 else ", ".join(lab[:-1]) + " and " + lab[-1]


def family_macros(m: dict[str, str], fam: dict, tag: str) -> None:
    for fm in FMS:
        r, k = fam[fm], f"SSvt{tag}FM{NAME[fm]}"
        assert not r["lost"], (tag, fm)
        m[k + "Mean"] = num(r["mean"])
        m[k + "CI"] = ci(r["ci"])
        m[k + "Holm"] = pval(r["p_holm"])
        m[k + "Verdict"] = r["verdict"]
        m[k + "VS"] = SHORT[r["verdict"]]
        m[k + "Win"] = pct(r["win"])
        m[k + "N"] = str(r["n"])
    verdicts = [fam[fm]["verdict"] for fm in FMS]
    m[f"SSvt{tag}NRep"] = str(verdicts.count("replicated"))
    m[f"SSvt{tag}NRev"] = str(verdicts.count("reversed"))
    m[f"SSvt{tag}NEq"] = str(verdicts.count("equivalent"))
    means = [fam[fm]["mean"] for fm in FMS]
    m[f"SSvt{tag}Min"] = num(min(means))
    m[f"SSvt{tag}Max"] = num(max(means))
    m[f"SSvt{tag}HolmMax"] = pval(max(fam[fm]["p_holm"] for fm in FMS))
    m[f"SSvt{tag}CIAbsMax"] = f"{max(abs(v) for fm in FMS for v in fam[fm]['ci']):.3f}"
    rep = [fm for fm in FMS if fam[fm]["verdict"] == "replicated"]
    m[f"SSvt{tag}RepList"] = names(rep) if rep else "none"
    m[f"SSvt{tag}NotRepList"] = names([fm for fm in FMS if fm not in rep]) if len(rep) < len(FMS) else "none"
    if rep:
        m[f"SSvt{tag}HolmMaxRep"] = pval(max(fam[fm]["p_holm"] for fm in rep))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    res = json.loads(a.src.read_text())
    pool = yaml.safe_load(a.pool.read_text())
    ds = pool["datasets"]
    m: dict[str, str] = {}

    # frame (from the frozen pool file)
    n = int(res["n_tables"])
    assert n == len(ds) == int(pool["n_tables"])
    m["SSvtNTables"] = str(n)
    m["SSvtNBinary"] = str(sum(int(d["C"]) == 2 for d in ds))
    m["SSvtNMulti"] = str(sum(int(d["C"]) > 2 for d in ds))
    m["SSvtNImbBinary"] = str(sum(int(d["C"]) == 2 and float(d["pi_min"]) < 0.25 for d in ds))
    post12 = [d for d in ds if d.get("fm1_contam") == "post" and d.get("fm2_contam") == "post"]
    m["SSvtNPostOne"] = str(len(post12))
    m["SSvtNPostFive"] = str(sum(d.get("fm5_contam") == "post" for d in ds))
    m["SSvtNFlagged"] = str(sum(bool(d.get("flag")) for d in ds))
    m["SSvtMargin"] = f"{MARGIN:.2f}"

    # tables lost per FM (complete-case rule, prereg Sec. 3.1; loss limit 10%, Sec. 4.1)
    for fm in FMS:
        m[f"SSvtNLostFM{NAME[fm]}"] = str(n - int(res["P"][fm]["n"]))
    lost = sorted({n - int(res["P"][fm]["n"]) for fm in FMS} - {0})
    m["SSvtNLostMax"] = str(max(lost) if lost else 0)
    m["SSvtNAnalysedMin"] = str(min(int(res["P"][fm]["n"]) for fm in FMS))

    for key, (tag,) in FAMILIES.items():
        family_macros(m, res[key], tag)
    for key, tag in SECONDARY.items():
        family_macros(m, res["secondary"][key], tag)

    # frame-level statements, prereg Sec. 4.2 (applied to the stored verdicts)
    def ver(fam: str) -> dict[str, str]:
        return {fm: res[fam][fm]["verdict"] for fm in FMS}
    p, e, q = ver("P"), ver("E"), ver("Q")
    rep = [fm for fm in FMS if p[fm] == "replicated"]
    rev = [fm for fm in FMS if p[fm] == "reversed"]
    eq = [fm for fm in FMS if e[fm] == "equivalent"]
    qrep = [fm for fm in FMS if q[fm] == "replicated"]
    qrev = [fm for fm in FMS if q[fm] == "reversed"]
    c_rc = len(rep) >= 3 and not rev
    c_cov = len(eq) >= 3 and not rep
    c_acq = len(qrep) >= 3 and not qrev
    m["SSvtClaimRC"] = "holds" if c_rc else "does not hold"
    m["SSvtClaimCOV"] = "holds" if c_cov else "does not hold"
    m["SSvtClaimACQ"] = "holds" if c_acq else "does not hold"
    # consistency with the claim string of the stored run (prereg Sec. 4.2; Amendment 1)
    stored = res["claim"]
    if c_rc:
        assert stored.startswith("C-RC holds"), stored
    elif c_cov:
        assert stored.startswith("C-COV holds"), stored
    elif not rev:
        assert stored == f"mixed: replicated {rep}, equivalent {eq}", stored
    m["SSvtPRepNames"] = names(rep) if rep else "none"
    m["SSvtEEqNames"] = names(eq) if eq else "none"
    m["SSvtQRepNames"] = names(qrep) if qrep else "none"
    m["SSvtQNotNames"] = names([fm for fm in FMS if fm not in qrep]) or "none"
    # per-FM reading of P and E (prereg Sec. 4.1; stored by the analysis)
    for fm in FMS:
        m[f"SSvtReadingFM{NAME[fm]}"] = res["reading"][fm].replace("+-", "$\\pm$")

    # label overhead of arm A (labels used / mC): a property of the draws, identical for FMs with the
    # same tables; reported from the FMs that kept all tables
    full = [fm for fm in FMS if int(res["P"][fm]["n"]) == n]
    assert full, "no FM kept every table"
    oh = res["A_label_overhead"]
    for mm, tag in ((1, "One"), (3, "Three"), (5, "Five"), (10, "Ten")):
        vals = {round(oh[f"{fm}_m{mm}"], 12) for fm in full}
        assert len(vals) == 1, (mm, vals)
        m[f"SSvtOverheadM{tag}"] = f"{oh[f'{full[0]}_m{mm}']:.2f}"
    m["SSvtOverheadMThreeToTen"] = f"{sum(oh[f'{full[0]}_m{x}'] for x in (3, 5, 10)) / 3:.2f}"
    for fm in FMS:
        m[f"SSvtOverheadFM{NAME[fm]}"] = f"{sum(oh[f'{fm}_m{x}'] for x in (3, 5, 10)) / 3:.2f}"

    # subsets of the P contrast (descriptive, unadjusted)
    for key, tag in SUBSETS.items():
        for fm in FMS:
            r, k = res["subsets"][key][fm], f"SSvt{tag}FM{NAME[fm]}"
            m[k + "Mean"] = num(r["mean"])
            m[k + "CI"] = ci(r["ci"])
            m[k + "P"] = pval(r["p_unadj"])
            m[k + "N"] = str(r["n"])
    # SENS-FLAG (prereg Sec. 5): does the sign or unadjusted significance of a P contrast change?
    flips = 0
    for fm in FMS:
        r = res["subsets"]["sens_flag"][fm]
        sig_sens = r["p_unadj"] < 0.05 and r["mean"] > 0
        flips += int(sig_sens != (p[fm] == "replicated"))
    m["SSvtSensFlagNFlip"] = str(flips)
    # leave-one-table-out range of the P p-values (prereg Sec. 5)
    if rep:
        m["SSvtLooMaxPRep"] = pval(max(res["loo_P"][fm][1] for fm in rep))
    notrep = [fm for fm in FMS if fm not in rep]
    if notrep:
        m["SSvtLooMinPNotRep"] = pval(min(res["loo_P"][fm][0] for fm in notrep))
    m["SSvtNInfeasible"] = str(int(res["infeasible_draws"]))
    # decomposition at m = 1 (descriptive): S - RC share of S - R
    dec = res["decomposition"]
    share = [abs(dec[f"macro_f1_{fm}_m1"]["S_RC"]) / dec[f"macro_f1_{fm}_m1"]["S_R"] for fm in FMS]
    assert all(math.isfinite(s) for s in share)
    m["SSvtLOneShareMax"] = f"{100 * max(share):.0f}\\%"

    head = [
        "% Generated by paper/p1/make_study17_numbers.py from the single Study 17 run",
        "% (results/study17/study17_results.json; tools/study17/analysis.py) and data/pool_study17.yaml.",
        "% Study 17 is pre-registered (prereg/STUDY17_COVERAGE_CONDITIONED.md, incl. Amendment 1).",
        "% Confirmatory families: P, E, Q, L1 (Holm over the four FMs each). Pre-declared secondary:",
        "% BA, Acc, Cov (RC-R), COne (S-R, m>=3), LOneCov (S-R, m=1). Subsets: descriptive, unadjusted.",
        "% Do not edit by hand.",
    ]
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    a.out.write_text("\n".join(head + lines) + "\n")
    print(f"{len(m)} macros -> {a.out}")


if __name__ == "__main__":
    main()
