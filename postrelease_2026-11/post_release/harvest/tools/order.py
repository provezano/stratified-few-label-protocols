"""INCLUSION_RULE.txt Section 2: frozen, deterministic candidate order -> curation/order.csv (one row per record).
Inputs: meta/cands_new.csv (CONF3 cands_all filtered: listing date >= cut, unreserved) + meta/pool_extra.csv (all E7/S2/CONF3 probes)."""
import csv, re, sys
from pathlib import Path
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parents[1]
csv.field_size_limit(10**8)
OUT = re.compile(r"death|dead|die|mortal|surviv|outcome|status|disease|diagnos|infect|complicat|presen|absen|recurr|relaps|positiv|negativ|case|adverse|event|"
                 r"readmi|lame|fail|success|resist|pregnan|stunt|malnut|anemi|anaemi|obes|diabet|hypert|depress|fractur|injur|sepsis|cancer|tumou?r|malign|metasta|"
                 r"occup|detect|lesion|sever|grade|stage|cure|respon|remission|delirium|aki|stroke|bleed|transfus|icu|death|mort|incid|prevalen|seroposit|"
                 r"pcr|culture|parasit|carrier|colonis|coloniz|nest|breed|predat|disper|attack|bite|frail|fall|disab|dement|cognit|preterm|caesar|cesar|"
                 r"low.?birth|apgar|nicu|sga|lga|hospitali|admission|discharge|severity|toxic|wilt|blight|rot|necros|symptom|ill|sick|pathogen", re.I)
EXC = re.compile(r"^\s*(sex|gender|age|year|date|site|region|country)", re.I)
SRC = {"fs_probe": "figshare", "mendeley_probe": "mendeley", "dataverse_probe": "dataverse", "dryad_probe": "dryad"}
def outlike(cands):
    hits = []
    for a, k, m in re.findall(r"(.*?)\((\d+):(\d+)\)\|?", cands or ""):
        a = a.strip()
        if OUT.search(a) and not EXC.search(a) and 2 <= int(k) <= 10 and int(m) >= 11:
            hits.append(f"{a[:40]}({k}:{m})")
    return hits
recs = {}
def add(src, rec, created, title, f, sheet, n, d, cands):
    try:
        n = int(float(n)); d = int(float(d))
    except (TypeError, ValueError):
        return
    if src not in ("zenodo", "figshare", "mendeley", "dataverse") or not (100 <= n <= 6000 and d >= 2):
        return
    h = outlike(cands)
    if not h:
        return
    r = recs.setdefault((src, rec), dict(date=created[:10], source=src, record=rec, title=title[:150], files=[]))
    r["date"] = min(r["date"], created[:10])
    r["files"].append(f"{f}[{sheet}] n={n} d={d}: {' '.join(h)[:300]}")
for r in csv.DictReader(open(H / "meta/cands_new.csv")):
    add(r["src"], r["rec"], r["created"], r["title"], r["file"], r["sheet"], r["n"], r["d"], r["cands"])
for r in csv.DictReader(open(H / "meta/pool_extra.csv")):
    src = next((v for k, v in SRC.items() if r["probe"].startswith(k)), "zenodo")
    add(src, r["rec"], r["created"], r["title"], r["file"], r["sheet"], r["n"], r["d"], r["cands"])
rows = sorted((r for r in recs.values() if r["date"] >= "2026-09-24"), key=lambda r: (r["date"], r["source"], r["record"]))
with open(H / "curation/order.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pos", "date", "source", "record", "title", "files"]); w.writeheader()
    for i, r in enumerate(rows, 1):
        w.writerow(dict(pos=i, **{k: r[k] for k in ("date", "source", "record", "title")}, files=" || ".join(sorted(set(r["files"])))[:1500]))
from collections import Counter
print(len(rows), Counter(r["source"] for r in rows))
