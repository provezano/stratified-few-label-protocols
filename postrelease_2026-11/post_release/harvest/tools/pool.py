"""P1 confirm: unified candidate pool from ALL existing probe files (E7, S2, CONF3; read-only), incl. files the CONF3 mech
screen dropped (balanced binary). Metadata + stored distinct-value counts only. Writes meta/pool_extra.csv (records not in cands_new)."""
import csv, glob, json, re, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
csv.field_size_limit(10**8)
G = common.G; E7 = common.E7; S2 = common.PROBES / "S2_k5_fresh_2026-11-07/harvest"; C3 = common.PROBES / "class_balancing_g4_gates_2026-11-05/conf3/harvest"
sys.path.insert(0, str(C3 / "tools"))
M = {}
for f in [common.SP / "harvest/cands_zenodo.jsonl", common.SP / "harvest/cands_zenodox.jsonl", E7 / "meta/zen_xlsx_cands.jsonl", common.SP / "harvest/cands_dataverse.jsonl",
          common.SP / "harvest/cands_mendeley.jsonl", common.SP / "harvest/cands_figshare.jsonl", E7 / "meta/fs_detail.jsonl", E7 / "meta_b/fs_detail.jsonl",
          S2 / "meta/fs_detail.jsonl", S2 / "meta/zen_new_csv.jsonl", S2 / "meta/zen_new_xlsx.jsonl", C3 / "meta/fs_detail.jsonl"]:
    if f.exists():
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            M.setdefault(str(r["id"]), r)
for f in [E7 / "meta/s3_ranked.csv", E7 / "meta/zen_sel.csv"]:
    if f.exists():
        for r in csv.DictReader(open(f)):
            M.setdefault(r["rec"], dict(id=r["rec"], title=r.get("title", ""), created=r.get("created", ""), version_index=r.get("vi"), lic=r.get("lic"), concept=r.get("concept")))
R = common.reserved_ids()
seen = {r["rec"] for r in csv.DictReader(open(Path(__file__).resolve().parents[1] / "meta/cands_new.csv"))}
probes = glob.glob(str(E7 / "meta/*probe*.csv")) + glob.glob(str(S2 / "meta/*probe*.csv")) + glob.glob(str(C3 / "meta/*probe*.csv"))
out = {}
for pf in probes:
    for r in csv.DictReader(open(pf)):
        rec = str(r.get("rec") or r.get("id") or "")
        if not rec or rec in seen:
            continue
        k = rec.replace("mnd", "").replace("dvn", "")
        if k in R or k.lower() in R or k.upper() in R:
            continue
        m = M.get(rec, {}); tl = m.get("timeline") or {}
        cr = m.get("created") or (tl.get("posted") or "")[:10] or r.get("created", "")
        if not cr or cr < common.CUT:
            continue
        try:
            n = int(float(r.get("n") or 0)); d = int(float(r.get("d") or 0))
        except ValueError:
            continue
        if not (100 <= n <= 6000 and d >= 2):
            continue
        cs = [(int(a), int(b)) for a, b in re.findall(r"\((\d+):(\d+)\)", r.get("cands") or "")]
        if not any(b >= 11 for a, b in cs):
            continue
        if m.get("concept") and str(m["concept"]) in R:
            continue
        s = common.score((m.get("title") or "") + " " + (r.get("cands") or ""))
        key = (rec, r.get("file"), r.get("sheet", ""))
        out[key] = dict(score=s, rec=rec, created=cr, vi=m.get("version_index", m.get("version")), lic=m.get("lic", ""), title=(m.get("title") or "")[:200],
                        file=r.get("file"), sheet=r.get("sheet", ""), n=n, d=d, cands=(r.get("cands") or "")[:800], cols=(r.get("cols") or "")[:2500], probe=Path(pf).name)
rows = sorted(out.values(), key=lambda x: -x["score"])
with open(Path(__file__).resolve().parents[1] / "meta/pool_extra.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "files", len({r["rec"] for r in rows}), "records")
