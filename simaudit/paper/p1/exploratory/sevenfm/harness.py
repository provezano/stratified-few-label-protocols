"""Harness for the seven-FM extensions of the exploratory analyses (EXPLORATORY; stored fits only).

Each item script ``bN_*.py`` is run twice from simaudit/:

    nice -n 15 .venv/bin/python paper/p1/exploratory/sevenfm/bN_*.py 4   # verify: reproduce the four-FM outputs
    nice -n 15 .venv/bin/python paper/p1/exploratory/sevenfm/bN_*.py 7   # extend: fm1 fm2 fm3 fm5 fm6 fm7 fm8

The scripts reuse the original analysis code unchanged: they import the original module,
re-bind its FM list and its output directory, and call its functions. Run "4" uses the
real ``results/`` and must reproduce the existing CSVs and macros exactly (``verify``).
Run "7" points ``SIMAUDIT_RESULTS_DIR`` at a read-only symlink view (``FARM``) that adds the
stored fits of the three exploratory FMs next to the pre-registered ones:

- DEV Study 7 Part (a): ``study7/a/{did}_fm6|fm8`` -> ``fm_extra/out/s7a/``, ``{did}_fm7`` ->
  ``fm_extra/out/s7a_v2/`` (the per-fit-seeded Mitra; the unseeded ``s7a/*_fm7`` is NOT used);
- T1' Study 12 draws: ``study12/{key}_fm6|fm7`` -> ``fm_extra/out/t1p/``, ``{key}_fm8`` ->
  ``fm_extra/out/t1p_fm8/``.

The extras' fits use the same stored selections (rows and seeds) as the four pre-registered
FMs, checked in ``check_draws``. Nothing outside ``sevenfm/out/`` is written; the original
output files stay byte-identical. sealed_heldout/ is never touched.

New macros get the prefix ``\\ExSevFm`` (``rename``). In run "7" the per-FM macros of the four
pre-registered FMs are dropped (they are the existing four-FM numbers); per-FM macros of
fm6/fm7/fm8 and every pooled / min / max / count macro over the seven FMs are kept.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

HERE = Path(__file__).resolve().parent
EXPL = HERE.parent
ROOT = EXPL.parents[2]  # simaudit/
XTRA = EXPL / "fm_extra" / "out"
FARM = Path(os.environ.get("SEVENFM_FARM", "<tmp>/-Users-anonymous-ge-paper/"
                           "2d9d60ea-d077-4e58-bc60-e4ff3878dfa5/scratchpad/results7"))
FMS4 = ("fm1", "fm2", "fm3", "fm5")
FMS7 = ("fm1", "fm2", "fm3", "fm5", "fm6", "fm7", "fm8")
EXTRA = ("fm6", "fm7", "fm8")
FMW_X = dict(fm6="FMsix", fm7="FMseven", fm8="FMeight")
NAMES_X = dict(fm6="TabDPT", fm7="Mitra", fm8="LimiX")
PRE_TAGS = ("FMone", "FMtwo", "FMthree", "FMfive")


def build_farm() -> None:
    """(Re)build the symlink view of results/ with the extras' stored fits added."""
    R = ROOT / "results"
    if FARM.exists():
        shutil.rmtree(FARM)
    (FARM / "study7" / "a").mkdir(parents=True)
    (FARM / "study12").mkdir()
    for e in R.iterdir():
        if e.name not in ("study7", "study12"):
            (FARM / e.name).symlink_to(e)
    for e in (R / "study7").iterdir():
        if e.name != "a":
            (FARM / "study7" / e.name).symlink_to(e)
    for e in (R / "study7" / "a").iterdir():
        (FARM / "study7" / "a" / e.name).symlink_to(e)
    for e in (R / "study12").iterdir():
        (FARM / "study12" / e.name).symlink_to(e)
    links = [(XTRA / "s7a", "*_fm6.parquet", "study7/a"), (XTRA / "s7a", "*_fm8.parquet", "study7/a"),
             (XTRA / "s7a_v2", "*_fm7.parquet", "study7/a"), (XTRA / "t1p", "*_fm6.parquet", "study12"),
             (XTRA / "t1p", "*_fm7.parquet", "study12"), (XTRA / "t1p_fm8", "*_fm8.parquet", "study12")]
    for d, pat, dst in links:
        for p in sorted(d.glob(pat)):
            if "partial" not in p.name:
                (FARM / dst / p.name).symlink_to(p)


def setup(item: str):
    """Parse the run set from argv, set the results dir, return (set, FMS, out_dir)."""
    s = sys.argv[1] if len(sys.argv) > 1 else "7"
    assert s in ("4", "7"), s
    if s == "7":
        if not (FARM / "study12").exists():
            build_farm()
        os.environ["SIMAUDIT_RESULTS_DIR"] = str(FARM)
    out = HERE / "out" / item / ("four" if s == "4" else "seven")
    out.mkdir(parents=True, exist_ok=True)
    for p in (EXPL, EXPL / "review_reanalysis", EXPL / "claims_audit", EXPL / "e17_metrics"):
        sys.path.insert(0, str(p))
    return s, (FMS4 if s == "4" else FMS7), out


def copy_inputs(src_dir: Path, names, out: Path) -> None:
    for n in names:
        shutil.copy2(src_dir / n, out / n)


def rename(line: str, note: str) -> str:
    r"""\ExXxx -> \ExSevFmXxx; append the seven-FM provenance note to the source comment."""
    line = re.sub(r"\\newcommand\{\\Ex", r"\\newcommand{\\ExSevFm", line, count=1)
    return line.rstrip() + f" [SEVEN-FM EXTENSION, exploratory: {note}]"


def keep7(name: str) -> bool:
    """Run 7: drop per-FM macros of the four pre-registered FMs (already reported)."""
    return not any(t in name for t in PRE_TAGS)


def macro_name(line: str) -> str:
    m = re.match(r"\\newcommand\{\\(\w+)\}", line)
    return m.group(1) if m else ""


def write_tex(path: Path, lines, header: str) -> None:
    path.write_text("\n".join([f"% {header}", "% Generated by paper/p1/exploratory/sevenfm/. EXPLORATORY. Do not edit."]
                              + sorted(lines)) + "\n", encoding="utf-8")


def verify_lines(new_lines, ref_tex: Path, label: str) -> list[str]:
    """Every macro line of the four-FM re-run must occur verbatim in the reference tex."""
    ref = set(ref_tex.read_text(encoding="utf-8").splitlines())
    miss = [ln for ln in new_lines if ln not in ref]
    print(f"[verify {label}] {len(new_lines) - len(miss)}/{len(new_lines)} macro lines identical to {ref_tex.name}")
    for ln in miss[:20]:
        print("  MISMATCH:", ln)
    return miss


def verify_csv(new: Path, ref: Path) -> bool:
    import pandas as pd
    a, b = pd.read_csv(new), pd.read_csv(ref)
    ok = a.shape == b.shape and list(a.columns) == list(b.columns)
    if ok:
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)
        except AssertionError as e:
            ok = False
            print("   ", str(e)[:400])
    print(f"[verify csv] {new.name}: {'IDENTICAL' if ok else 'DIFFERS'} (vs {ref})")
    return ok
