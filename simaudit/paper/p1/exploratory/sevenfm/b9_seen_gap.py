"""B9 (EXPLORATORY, new script): seen-class macro-F1 convention, stratified - random, seven FMs.

The pre-registered numbers (\\SsevSeenGap*, numbers_p1.tex) are parsed by make_numbers.py from
reports/STUDY7_REPORT.md, "Per-budget contrasts, seen_macro_f1", rows "strat - random fmX".
This script recomputes that contrast from the stored ``seen_macro_f1`` column (macro-F1 over the
classes present in the context): per DEV table and budget, mean over the 5 stratified draws minus
mean over the 20 random draws; mean over the 20 tables; Wilcoxon over tables, unadjusted (as in
the report). Run "4" checks the four FMs against the report rows (value and p, as printed);
run "7" adds fm6/fm7/fm8 (same Study 7 Part (a) draws) and, descriptively, the same contrast on
T1' with the Study 12 draws for all seven FMs (no pre-registered four-FM reference there).
Usage: nice -n 15 .venv/bin/python paper/p1/exploratory/sevenfm/b9_seen_gap.py {4|7}
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

S, FMS, OUT = H.setup("b9")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import common as cm  # noqa: E402
from simaudit.settings import settings  # noqa: E402

MW = {1: "One", 3: "Three", 5: "Five", 10: "Ten"}
FMW = dict(fm1="FMone", fm2="FMtwo", fm3="FMthree", fm5="FMfive", **H.FMW_X)


def fmt(x: float) -> str:
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.3f}"


def gaps(set_name: str) -> pd.DataFrame:
    if set_name == "dev":
        files = [settings.results_dir / "study7" / "a" / f"{d}_{fm}.parquet" for d in cm.dev_ids() for fm in FMS]
    else:
        files = [p for fm in FMS for p in sorted((settings.results_dir / "study12").glob(f"*_{fm}.parquet"))]
    fr = []
    for p in files:
        d = pd.read_parquet(p, columns=["dataset", "learner", "strategy", "mult", "rep", "seen_macro_f1"])
        fr.append(d[d.strategy.isin(["stratified", "random"])])
    d = pd.concat(fr, ignore_index=True)
    c = d.groupby(["learner", "dataset", "mult", "strategy"]).seen_macro_f1.mean().unstack("strategy")
    g = (c.stratified - c.random).rename("gap").reset_index()
    rows = []
    for (fm, m), x in g.groupby(["learner", "mult"]):
        v = x.gap.to_numpy()
        rows.append(dict(set=set_name, fm=fm, mult=int(m), n=len(v), mean=float(v.mean()),
                         p=float(wilcoxon(v).pvalue) if np.any(v != 0) else 1.0,
                         n_neg=int((v < 0).sum())))
    return pd.DataFrame(rows)


def report_rows() -> dict:
    text = (H.ROOT / "reports" / "STUDY7_REPORT.md").read_text(encoding="utf-8")
    blk = text.split("## Per-budget contrasts, seen_macro_f1")[1].split("\n## ")[0]
    out = {}
    for ln in blk.splitlines():
        m = re.match(r"\|\s*strat − random (fm\d)\s*\|(.*)", ln)
        if m:
            cells = [c.strip() for c in m.group(2).split("|") if c.strip()]
            out[m.group(1)] = [(float(re.match(r"([+\-−]?\d+\.\d+)", c).group(1).replace("−", "-")),
                                float(re.search(r"p=([\d.]+)", c).group(1))) for c in cells[:4]]
    return out


def main() -> None:
    dev = gaps("dev")
    dev.to_csv(OUT / "seen_gap_dev.csv", index=False)
    if S == "4":
        rep, bad = report_rows(), []
        for fm in FMS:
            for j, m in enumerate((1, 3, 5, 10)):
                r = dev[(dev.fm == fm) & (dev.mult == m)].iloc[0]
                v, p = rep[fm][j]
                ok = f"{r['mean']:+.3f}" == f"{v:+.3f}" and f"{r.p:.3f}" == f"{p:.3f}"
                if not ok:
                    bad.append((fm, m, r["mean"], r.p, v, p))
        print(f"[verify] {16 - len(bad)}/16 (FM x m) cells equal STUDY7_REPORT.md seen_macro_f1 rows (value 3 dp, p 3 dp)")
        for b in bad:
            print("  MISMATCH:", b)
        ref = (H.EXPL.parent / "numbers_p1.tex").read_text()
        for j, w in ((1, "One"), (3, "Three")):
            x = dev[dev.mult == j]["mean"]
            for k, v in (("Min", x.min()), ("Max", x.max())):
                want = re.search(r"\\newcommand\{\\SsevSeenGap" + w + k + r"\}\{([^}]*)\}", ref).group(1)
                print(f"[verify] SsevSeenGap{w}{k}: recomputed {fmt(v)} vs numbers_p1 {want}",
                      "OK" if fmt(v) == want else "MISMATCH")
                bad += [] if fmt(v) == want else [w + k]
        print("VERIFY B9:", "PASS" if not bad else f"FAIL {bad}")
        return
    t1 = gaps("t1prime")
    t1.to_csv(OUT / "seen_gap_t1p.csv", index=False)
    note = "B9, seen-class macro-F1 S-R, recomputed from stored seen_macro_f1"
    L = []
    for set_name, d, sw in (("dev", dev, "Dev"), ("t1prime", t1, "Tp")):
        src = f"sevenfm/out/b9/seven/seen_gap_{'dev' if set_name == 'dev' else 't1p'}.csv ({set_name}"
        drawn = "Study 7 Part (a)" if set_name == "dev" else "Study 12"
        for r in d.itertuples():
            if r.fm in H.EXTRA or set_name == "t1prime":
                st = f"ExSevFmSeenGap{sw}{FMW[r.fm]}M{MW[r.mult]}"
                L.append(f"\\newcommand{{\\{st}}}{{{fmt(r.mean)}}} % {src}, {r.fm}, m={r.mult}, {drawn} draws, n={r.n})")
                L.append(f"\\newcommand{{\\{st}P}}{{{r.p:.3f}}} % {src}, Wilcoxon over tables, unadjusted)")
        for m in (1, 3, 5, 10):
            for grp, fms, gw in (("seven", H.FMS7, "Seven"), ("four", H.FMS4, "Four"), ("extra", H.EXTRA, "Extra")):
                if set_name == "dev" and grp == "four":
                    continue  # = \SsevSeenGap* (pre-registered report)
                x = d[(d.mult == m) & d.fm.isin(fms)]
                for k, v in (("Min", x["mean"].min()), ("Max", x["mean"].max())):
                    L.append(f"\\newcommand{{\\ExSevFmSeenGap{sw}{gw}M{MW[m]}{k}}}{{{fmt(v)}}} % {src}, "
                             f"m={m}, {k.lower()} over {grp} FMs ({', '.join(fms)}), {drawn} draws)")
    H.write_tex(H.HERE / "out" / "sevenfm_b9.tex", [ln + f" [SEVEN-FM EXTENSION, exploratory: {note}]" for ln in L],
                "B9 seen-class macro-F1 convention, seven FMs")
    print(dev.round(3).to_string())
    print(t1.round(3).to_string())
    print(len(L), "macros")


if __name__ == "__main__":
    main()
