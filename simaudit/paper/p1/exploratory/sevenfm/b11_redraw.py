"""B11 (EXPLORATORY, new script around the original function): redraw comparator, seven FMs.

Original code: paper/p1/make_tables.py ``redraw_protocol`` (imported, not run as a script):
redraw = the stored random draws that cover every class (coverage == 1), stratified minus
redraw in macro-F1 per (table, FM, m), mean over tables, unadjusted Wilcoxon. Re-bound: FM list
and the source directory. Sets:
- DEV, Study 7 Part (a) draws (all seven FMs): run "4" must reproduce \\Redraw* of
  tables_numbers_p1.tex; run "7" extends to seven FMs.
- T1', Study 12 draws (run "7" only): the same covered-random definition for all seven FMs.
  Note: Study 12 also stored a true redraw arm for the four pre-registered FMs (pre-declared
  S-1); the extras have no redraw arm, so covered random draws are used for every FM here
  (like for like). The fm2 - fm5 ranking rows are unchanged by the extras and are not repeated.
Usage: nice -n 15 .venv/bin/python paper/p1/exploratory/sevenfm/b11_redraw.py {4|7}
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

S, FMS, OUT = H.setup("b11")
sys.path.insert(0, str(H.EXPL.parent))

import make_tables as mt  # noqa: E402
from simaudit.settings import settings  # noqa: E402

FMW = dict(fm1="FMone", fm2="FMtwo", fm3="FMthree", fm5="FMfive", **H.FMW_X)
MW = {1: "MOne", 3: "MThree", 5: "MFive", 10: "MTen"}


def run(src_dir: Path, tag: str):
    mt.SRC, mt.FMS = src_dir, list(FMS)
    tab, nums = mt.redraw_protocol()
    tab.to_csv(OUT / f"redraw_{tag}.csv", index=False)
    return tab, nums


def main() -> None:
    tab, nums = run(settings.results_dir / "study7" / "a", "dev")
    if S == "4":
        ref = (H.EXPL.parent / "tables_numbers_p1.tex").read_text()
        fmt = {"RedrawCoveredOne": f"{nums['RedrawCoveredOne']:.2f}", "RedrawGapMin": mt.signed(nums["RedrawGapMin"]),
               "RedrawGapMax": mt.signed(nums["RedrawGapMax"]), "RedrawRankMin": mt.signed(nums["RedrawRankMin"]),
               "RedrawRankMax": mt.signed(nums["RedrawRankMax"]), "RedrawRankPMax": f"{nums['RedrawRankPMax']:.3f}"}
        bad = []
        for k, v in fmt.items():
            want = re.search(r"\\newcommand\{\\" + k + r"\}\{([^}]*)\}", ref).group(1)
            ok = v == want
            bad += [] if ok else [k]
            print(f"[verify] {k}: {v} vs tables_numbers_p1 {want}", "OK" if ok else "MISMATCH")
        print("VERIFY B11:", "PASS" if not bad else f"FAIL {bad}")
        return
    t1, n1 = run(settings.results_dir / "study12", "t1p")
    L, note = [], "B11, redraw = covered random draws (coverage == 1)"
    for sw, t, n, drawn in (("Dev", tab, nums, "DEV Study 7 Part (a)"), ("Tp", t1, n1, "T1' Study 12")):
        src = f"sevenfm/out/b11/seven/redraw_{'dev' if sw == 'Dev' else 't1p'}.csv ({drawn} draws"
        later = t[t.m > 1]
        for grp, fms, gw in (("seven", H.FMS7, "Seven"), ("four", H.FMS4, "Four"), ("extra", H.EXTRA, "Extra")):
            if sw == "Dev" and grp == "four":
                continue  # = \RedrawGapMin/Max
            g = later[[f"gap_{fm}" for fm in fms]].to_numpy()
            L.append(f"\\newcommand{{\\ExSevFmRedraw{sw}{gw}GapMin}}{{{mt.signed(g.min())}}} % {src}, m in 3-10, min over {grp} FMs)")
            L.append(f"\\newcommand{{\\ExSevFmRedraw{sw}{gw}GapMax}}{{{mt.signed(g.max())}}} % {src}, m in 3-10, max over {grp} FMs)")
        for r in t.itertuples():
            for fm in FMS:
                if sw == "Dev" and fm not in H.EXTRA:
                    continue
                L.append(f"\\newcommand{{\\ExSevFmRedraw{sw}{FMW[fm]}{MW[int(r.m)]}Gap}}{{{mt.signed(getattr(r, 'gap_' + fm))}}}"
                         f" % {src}, {fm}, m={r.m}, n={int(r.n_datasets)})")
                L.append(f"\\newcommand{{\\ExSevFmRedraw{sw}{FMW[fm]}{MW[int(r.m)]}P}}{{{getattr(r, 'p_' + fm):.3f}}}"
                         f" % {src}, {fm}, m={r.m}, Wilcoxon unadjusted)")
        L.append(f"\\newcommand{{\\ExSevFmRedraw{sw}CoveredOne}}{{{n['RedrawCoveredOne']:.2f}}} % {src}, share of random draws covering all classes at m=1)")
    H.write_tex(H.HERE / "out" / "sevenfm_b11.tex", [ln + f" [SEVEN-FM EXTENSION, exploratory: {note}]" for ln in L],
                "B11 redraw comparator, seven FMs")
    print(tab.round(3).to_string())
    print(t1.round(3).to_string())
    print(len(L), "macros")


if __name__ == "__main__":
    main()
