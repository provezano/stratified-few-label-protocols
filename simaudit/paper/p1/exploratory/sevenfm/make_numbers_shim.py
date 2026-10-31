"""Verbatim copies of the helpers and of the THet block of paper/p1/make_numbers.py (lines 30-62,
1045-1071), wrapped in a function so they can run on another t1p_meta.csv. FM word map extended
with fm6/fm7/fm8. EXPLORATORY."""

from __future__ import annotations

import csv
import re

DIGITS = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


def macro_name(raw: str) -> str:
    name = "".join(DIGITS.get(c, c) for c in raw)
    name = re.sub(r"[^A-Za-z]", "", name)
    if not name:
        raise ValueError(raw)
    return name


def f(x: float, nd: int = 3, sign: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}\\%"


def lines(path) -> list[str]:
    OUT, SOURCES = {}, {}

    def put(name, value, src):
        name = macro_name(name)
        if name in OUT:
            raise KeyError(name)
        OUT[name], SOURCES[name] = str(value), src

    _hsrc = "analysis/out/t1p_meta.csv"
    with open(path, encoding="utf-8") as fh:
        _h = list(csv.DictReader(fh))
    _FMW = {"fm1": "FMone", "fm2": "FMtwo", "fm3": "FMthree", "fm5": "FMfive",
            "fm6": "FMsix", "fm7": "FMseven", "fm8": "FMeight"}
    put("THetK", _h[0]["k"], _hsrc)
    for _met, _stem in [("macro_f1", "F"), ("accuracy", "Acc"), ("balanced_accuracy", "BA")]:
        _rs = [r for r in _h if r["metric"] == _met]
        _ra = [r for r in _rs if r["mult"] == "all"]
        put(f"THet{_stem}IMin", f(min(float(r["I2"]) for r in _rs), 2), _hsrc + f" ({_met}, min over FM x budget)")
        put(f"THet{_stem}IMax", f(max(float(r["I2"]) for r in _rs), 2), _hsrc + f" ({_met}, max over FM x budget)")
        put(f"THet{_stem}TauMin", f(min(float(r["tau"]) for r in _ra), 3), _hsrc + f" ({_met}, mean over budgets)")
        put(f"THet{_stem}TauMax", f(max(float(r["tau"]) for r in _ra), 3), _hsrc + f" ({_met}, mean over budgets)")
        put(f"THet{_stem}ShareMin", pct(min(float(r["share_pos"]) for r in _ra)), _hsrc + f" ({_met}, all, min over FM)")
        put(f"THet{_stem}ShareMax", pct(max(float(r["share_pos"]) for r in _ra)), _hsrc + f" ({_met}, all, max over FM)")
        for r in _ra:
            _w = _FMW[r["fm"]]
            put(f"THet{_stem}{_w}AllPILo", f(float(r["pi_lo"]), 3), _hsrc + f" ({_met}, {r['fm']}, all)")
            put(f"THet{_stem}{_w}AllPIHi", f(float(r["pi_hi"]), 3), _hsrc + f" ({_met}, {r['fm']}, all)")
            put(f"THet{_stem}{_w}AllShare", pct(float(r["share_pos"])), _hsrc + f" ({_met}, {r['fm']}, all)")
        if _met == "macro_f1":
            for r in [r for r in _rs if r["mult"] == "1"]:
                _w = _FMW[r["fm"]]
                put(f"THetF{_w}MOnePILo", f(float(r["pi_lo"]), 3), _hsrc + f" (macro_f1, {r['fm']}, m=1)")
                put(f"THetF{_w}MOnePIHi", f(float(r["pi_hi"]), 3), _hsrc + f" (macro_f1, {r['fm']}, m=1)")
    return [f"\\newcommand{{\\{k}}}{{{OUT[k]}}} % {SOURCES[k]}" for k in sorted(OUT)]
