"""Round-16 descriptive check (no fits): the range of table size n and class count C on DEV and T1'.

Reads the pool summaries out/e10_pools_dev.csv and out/e10_pools_t1prime.csv (written by e10, from the
stored tables). HELD-OUT is sealed and is not read; its eligibility rule is in prereg/PILOT_I2.md Sec. 2.
Writes out/rv_r16.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r16_data_ranges.py
"""

from __future__ import annotations

import pandas as pd

from rv_common import Macros, OUT

E10 = OUT.parent.parent / "out"
M = Macros("r16")
for tag, f in (("Dev", "e10_pools_dev.csv"), ("Tp", "e10_pools_t1prime.csv")):
    d = pd.read_csv(E10 / f)
    src = f"exploratory/out/{f} (n = {len(d)} tables)"
    M.put(f"Data{tag}NMin", f"{int(d.n.min()):,}".replace(",", "{,}"), src)
    M.put(f"Data{tag}NMax", f"{int(d.n.max()):,}".replace(",", "{,}"), src)
    M.put(f"Data{tag}CMin", str(int(d.C.min())), src)
    M.put(f"Data{tag}CMax", str(int(d.C.max())), src)
M.write()
print((OUT / "rv_r16.tex").read_text())
