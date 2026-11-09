"""Render curation/decisions.py from tools/dec_src_*.py + curation/order.csv (positions -> (source, record))."""
import csv, glob, importlib.util, pprint, sys
from pathlib import Path
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parents[1]
order = {int(r["pos"]): r for r in csv.DictReader(open(H / "curation/order.csv"))}
for r in csv.DictReader(open(H / "curation/order_ext.csv")) if (H / "curation/order_ext.csv").exists() else []:
    order[int(r["pos"])] = r
A, R = {}, {}
for f in sorted(glob.glob(str(H / "tools/dec_src_*.py"))):
    s = importlib.util.spec_from_file_location("d", f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); A.update(m.A); R.update(m.R)
both = set(A) & set(R); assert not both, both
# Stopping point (Amendment 1): the 40th eligible table after the build is pos 546; later positions are not judged.
STOP = 546
A = {p: v for p, v in A.items() if p <= STOP}; R = {p: v for p, v in R.items() if p <= STOP}
last = max(set(A) | set(R)); missing = [p for p in range(1, last + 1) if p not in A and p not in R]; assert not missing, missing
src = lambda p: (order[p]["source"], order[p]["record"])
acc = {src(p): dict(pos=p, date=order[p]["date"], **A[p]) for p in sorted(A)}
rej = {src(p): (R[p][0], R[p][1], p, order[p]["date"]) for p in sorted(R)}
txt = '''"""Curation decisions, P1 fresh-data confirmation (rule: ../INCLUSION_RULE.txt + Amendment 1). Title/description/column names and
the per-class counts of the declared target only; no model. Processed strictly in curation/order.csv order (pos), positions 1..%d.
ACCEPT: (src, rec) -> dict(pos, date, key, file, sheet, target, drop, why, dropwhy). Drops: exact names, unique prefixes or re:<regex>.
       ACCEPT means 'passes the judgment part'; the build applies the mechanical rules, first-version and exclusion checks.
REJECT: (src, rec) -> (code, reason, pos, listing date). Codes: R0 first version < 2026-09-24 / not verifiable; R1 licence;
R2 mechanical; R3a no documented class outcome / design factor or covariate; R3d target derived from features or tool output;
R5r re-release, compilation of public sources or duplicate deposit; R5s synthetic/simulated; R6 unsuitable structure (aggregated,
long format, several rows per unit, unclear target); R7 non-tabular features; RX reserved/used elsewhere."""
ACCEPT = %s
REJECT = %s
''' % (last, pprint.pformat(acc, width=200, sort_dicts=False), pprint.pformat(rej, width=200, sort_dicts=False))
(H / "curation/decisions.py").write_text(txt)
print(len(acc), "accept;", len(rej), "reject; up to pos", last)
