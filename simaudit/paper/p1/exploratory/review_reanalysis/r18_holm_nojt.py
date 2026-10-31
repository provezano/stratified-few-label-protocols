"""Round-18 sensitivity (no fits): Holm adjustment of the Study 7 family without the eight JT tests.

Reads the unadjusted and Holm-adjusted p-values of the 15 confirmatory contrasts from the confirmatory
table of reports/STUDY7_REPORT.md, recomputes Holm over the 7 non-JT contrasts and compares.
Writes out/rv_r18.tex. EXPLORATORY sensitivity analysis (MINOR-6 of blind review 2).
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r18_holm_nojt.py
"""

from __future__ import annotations

from pathlib import Path

from rv_common import Macros, OUT

REP = Path(__file__).resolve().parents[4] / "reports" / "STUDY7_REPORT.md"
rows = []
txt = REP.read_text().split("## Confirmatory contrasts", 1)[1].split("\n## ", 1)[0]
for line in txt.splitlines():
    if line.startswith("| (i"):
        c = [x.strip() for x in line.strip("|").split("|")]
        rows.append((c[0], float(c[7]), float(c[8].strip("*"))))
assert len(rows) == 15, len(rows)
keep = [r for r in rows if not r[0].startswith("(iii)")]
assert len(keep) == 7
order = sorted(range(len(keep)), key=lambda i: keep[i][1])
adj, run = [0.0] * len(keep), 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(keep) - rank) * keep[i][1]))
    adj[i] = run
# the report rounds Holm p to two significant digits, so compare at that precision
diff = max(abs(float(f"{a:.2g}") - h) / h for a, (_, _, h) in zip(adj, keep))
for (name, p, h), a in zip(keep, adj):
    print(f"{name:10s} raw {p:.3g} holm15 {h:.3g} holm7 {a:.3g}")
M = Macros("r18")
src = "reports/STUDY7_REPORT.md confirmatory table; Holm over the 7 non-JT contrasts (exploratory sensitivity)"
M.put("HolmNoJTN", str(len(keep)), src)
M.put("HolmNoJTMaxRelDiff", f"{diff:.2f}", src)
M.put("HolmNoJTSame", "unchanged" if diff < 0.05 else "changed", src)
M.write()
print((OUT / "rv_r18.tex").read_text())
