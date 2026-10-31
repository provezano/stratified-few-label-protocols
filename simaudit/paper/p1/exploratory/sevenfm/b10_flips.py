"""B10 (EXPLORATORY): ranking flips / top-FM change with seven FMs.

(a) Study 12 S-3 flip rate (simaudit.analysis.study12.flip_rate, unchanged) on T1' Study 12 draws.
    The pre-declared S-3 statistic stays the six learners (four pre-registered FMs + fm1_bal,
    fm5_bal; \\StwelveFlip*). Exploratory variants: nine learners (seven FMs + the two balanced
    TabPFNs) and the seven FMs alone (analogue of the pre-declared four-FM row \\StwelveFlipFFM*).
    Plus the matched noise baseline of review_reanalysis/r6b_checks.py (reps 0-4 vs 5-19).
    Run "4" must reproduce the S-3 counts and rates of numbers_p1.tex.
(b) Per-table change of the top-ranked FM, protocol vs split-half noise (claims_audit
    make_claims_numbers.py block "ExCaTop", copied verbatim), from E17 flip cells:
    run "4" = existing e17_flips_cells.csv (four FMs; T1' = Study 9 draws); run "7" = B2's
    seven-FM cells (DEV Study 7 Part (a), T1' Study 12) and B2's four-FM cells on Study 12 (B).
    Needs b2_e17.py 7 first.
Usage: nice -n 15 .venv/bin/python paper/p1/exploratory/sevenfm/b10_flips.py {4|7}
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

S, FMS, OUT = H.setup("b10")

import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import rv_common  # noqa: E402

rv_common.OUT = OUT
import r6b_checks as r6b  # noqa: E402
from simaudit.analysis import study12 as a12  # noqa: E402
from simaudit.settings import settings  # noqa: E402

BAL = ("fm1_bal", "fm5_bal")


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}\\%"


def f2(x):
    s = f"{x:+.2f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def cf(x, nd=2):  # claims_audit f
    return f"{x:.{nd}f}".replace("-", "$-$").replace("+", "$+$")


def cpf(p):  # claims_audit pf
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def s3(runs, learners, metric):
    r = a12.flip_rate(runs, learners, metric)
    return dict(rate=r["rate"], num=r["flips"], den=r["decisive"], noise=r["noise_rate"], diff=r["diff"],
                diff_lo=r["diff_ci"][0], diff_hi=r["diff_ci"][1], ci_lo=r["ci"][0], ci_hi=r["ci"][1],
                top_strat={m: o["stratified"][0] for m, o in r["orders"].items()},
                top_rand={m: o["random"][0] for m, o in r["orders"].items()}, tau=r["tau"])


def top_change(fl: pd.DataFrame) -> dict:
    out = {}
    for st, w in [("t1prime", "Tp"), ("dev", "Dev")]:
        x = fl[(fl.set == st) & (fl.metric == "macro_f1")]
        p = x.pivot_table(index="table", columns="kind", values="top_flip", aggfunc="mean").dropna()
        out[w] = dict(prot=p.protocol.mean(), noise=p.null_split_half.mean(),
                      p=wilcoxon(p.protocol, p.null_split_half).pvalue, n=len(p))
    return out


def main() -> None:
    runs = a12.load_runs(settings.results_dir / "study12")
    variants = {"F": (FMS4 := H.FMS4) + BAL, "Acc": H.FMS4 + BAL, "FFM": H.FMS4}
    if S == "7":
        variants.update({"SevenF": H.FMS7 + BAL, "SevenAcc": H.FMS7 + BAL, "SevenFFM": H.FMS7})
    res = {}
    for k, lr in variants.items():
        res[k] = s3(runs, lr, "accuracy" if k.endswith("Acc") else "macro_f1")
        print(k, {a: b for a, b in res[k].items() if a not in ("tau",)})
    pd.DataFrame([{**{a: b for a, b in v.items() if not isinstance(b, dict)}, "variant": k} for k, v in res.items()]).to_csv(
        OUT / "s3_flips.csv", index=False)
    if S == "4":
        ref = (H.EXPL.parent / "numbers_p1.tex").read_text()
        bad = []
        for t in ("F", "Acc", "FFM"):
            r = res[t]
            mine = {"": pct(r["rate"]), "Num": str(r["num"]), "Den": str(r["den"]), "Noise": pct(r["noise"]),
                    "Diff": f2(r["diff"])}
            for k, v in mine.items():
                want = re.search(r"\\newcommand\{\\StwelveFlip" + t + k + r"\}\{([^}]*)\}", ref)
                want = want.group(1) if want else None
                ok = want == v
                bad += [] if ok else [t + k]
                print(f"[verify] StwelveFlip{t}{k}: {v} vs {want}", "OK" if ok else "MISMATCH")
        tc = top_change(pd.read_csv(H.EXPL / "e17_metrics" / "e17_flips_cells.csv"))
        ref = (H.EXPL / "claims_audit" / "out" / "claims_audit_numbers.tex").read_text()
        for w, d in tc.items():
            for k, v in (("Prot", cf(d["prot"])), ("Noise", cf(d["noise"])), ("P", cpf(d["p"]))):
                want = re.search(r"\\newcommand\{\\ExCaTop" + w + k + r"\}\{([^}]*)\}", ref).group(1)
                ok = want == v
                bad += [] if ok else [w + k]
                print(f"[verify] ExCaTop{w}{k}: {v} vs {want}", "OK" if ok else "MISMATCH")
        print("VERIFY B10:", "PASS" if not bad else f"FAIL {bad}")
        return
    # matched noise (r6b, reps 0-4 vs 5-19), seven FMs + bal
    r6b.LEARNERS = H.FMS7 + BAL
    mt = pd.DataFrame([r6b.flip_split(runs, range(0, 5), m) for m in ("macro_f1", "accuracy")])
    mt.to_csv(OUT / "r6b_matched_seven.csv", index=False)
    L = []
    src = "sevenfm/out/b10/seven/s3_flips.csv (study12.flip_rate, T1' Study 12 draws"
    names = {"SevenF": "nine learners (seven FMs + fm1_bal, fm5_bal), macro-F1",
             "SevenAcc": "nine learners, accuracy", "SevenFFM": "seven FMs, macro-F1"}
    for k, desc in names.items():
        r = res[k]
        stem = "ExSevFmStwelveFlip" + {"SevenF": "F", "SevenAcc": "Acc", "SevenFFM": "FFM"}[k]
        for sfx, v in (("", pct(r["rate"])), ("Num", str(r["num"])), ("Den", str(r["den"])), ("Noise", pct(r["noise"])),
                       ("Diff", f2(r["diff"])), ("DiffLo", f2(r["diff_lo"])), ("DiffHi", f2(r["diff_hi"]))):
            L.append(f"\\newcommand{{\\{stem}{sfx}}}{{{v}}} % {src}, {desc}; exploratory variant of pre-declared S-3)")
    for metric, w in (("macro_f1", "F"), ("accuracy", "Acc")):
        r0 = mt[mt.metric == metric].iloc[0]
        L.append(f"\\newcommand{{\\ExSevFmRvRsixbFlip{w}Matched}}{{{pct(r0.noise)}}} % r6b_matched_seven.csv, nine learners, noise reps 0-4 vs 5-19")
        L.append(f"\\newcommand{{\\ExSevFmRvRsixbFlip{w}MatchedDiff}}{{{f2(r0['diff'])}}} % r6b_matched_seven.csv, protocol minus matched noise")
        L.append(f"\\newcommand{{\\ExSevFmRvRsixbFlip{w}MatchedDiffLo}}{{{f2(r0.diff_lo)}}} % r6b_matched_seven.csv")
        L.append(f"\\newcommand{{\\ExSevFmRvRsixbFlip{w}MatchedDiffHi}}{{{f2(r0.diff_hi)}}} % r6b_matched_seven.csv")
    tops = []
    for k in ("FFM", "SevenFFM"):
        for m in sorted(res[k]["top_strat"]):
            tops.append(dict(variant=k, m=m, top_stratified=res[k]["top_strat"][m], top_random=res[k]["top_rand"][m],
                             kendall_tau=res[k]["tau"][m]))
    pd.DataFrame(tops).to_csv(OUT / "s3_top_by_m.csv", index=False)
    for tag, path, pre, desc in (
            ("A", H.HERE / "out" / "b2" / "seven" / "A" / "e17_flips_cells.csv", "ExSevFmCaTop",
             "seven FMs; DEV Study 7 Part (a), T1' Study 12 draws"),
            ("B", H.HERE / "out" / "b2" / "seven" / "B" / "e17_flips_cells.csv", "ExSevFmFourSTwelveCaTop",
             "four pre-registered FMs on T1' Study 12 draws (comparator)")):
        tc = top_change(pd.read_csv(path))
        for w, d in tc.items():
            if tag == "B" and w == "Dev":
                continue
            s = f"b2/seven/{tag}/e17_flips_cells.csv (top_flip, {w}, macro_f1, {desc}, n={d['n']} tables"
            L.append(f"\\newcommand{{\\{pre}{w}Prot}}{{{cf(d['prot'])}}} % {s}, protocol)")
            L.append(f"\\newcommand{{\\{pre}{w}Noise}}{{{cf(d['noise'])}}} % {s}, split-half noise)")
            L.append(f"\\newcommand{{\\{pre}{w}P}}{{{cpf(d['p'])}}} % {s}, paired Wilcoxon)")
    H.write_tex(H.HERE / "out" / "sevenfm_b10.tex", [ln + " [SEVEN-FM EXTENSION, exploratory: B10]" for ln in L],
                "B10 ranking flips / top-FM change, seven FMs")
    print(pd.DataFrame(tops).to_string())
    print(len(L), "macros")


if __name__ == "__main__":
    main()
