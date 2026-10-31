"""Shared helpers for the blind-review re-analyses (EXPLORATORY; read-only inputs)."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
FMS = ("fm1", "fm2", "fm3", "fm5")
FMW = dict(fm1="FMone", fm2="FMtwo", fm3="FMthree", fm5="FMfive")
MW = {1: "MOne", 2: "MTwo", 3: "MThree", 5: "MFive", 7: "MSeven", 10: "MTen"}


def fmt(x: float, nd: int = 3, sign: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def pfmt(p: float) -> str:
    if p < 0.001:
        e = f"{p:.1e}"
        mant, ex = e.split("e")
        return f"{mant}{{\\times}}10^{{{int(ex)}}}".join(["$", "$"])
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


class Macros:
    """Collect \\ExRv... macros for one part; write them as a .tex fragment."""

    def __init__(self, part: str):
        self.part, self.rows = part, []

    def put(self, name: str, value: str, src: str) -> None:
        assert name.isalpha(), name
        self.rows.append(f"\\newcommand{{\\ExRv{name}}}{{{value}}} % EXPLORATORY; source: {src}")

    def write(self) -> None:
        (OUT / f"rv_{self.part}.tex").write_text("\n".join(self.rows) + "\n", encoding="utf-8")


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj
