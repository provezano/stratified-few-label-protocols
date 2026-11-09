"""Zenodo route. modes:
  list since=YYYY-MM-DD ft=csv|xlsx|tsv  -> meta/zen_new_<ft>.jsonl (records created >= since, type dataset), ~1 req/s
  probe <listfile...> [shard=k nshard=m] -> download csv/tsv/xlsx (<= 4 MB) of unreserved post-cut records ranked by multi-class keywords,
     probe (shape, column names, distinct-value counts only), keep files with a 3..10-level column, delete others. Outcome-blind."""
import csv, datetime as dt, hashlib, json, re, sys, time, urllib.parse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, "simaudit/src")
from common import H, E7, CUT, get, reserved_ids, score, probe_csv, probe_grid, mech3
M = H / "meta"; RAW = H / "raw"; CAP = 4 * 2**20
arg = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)

def listing():
    ft = arg.get("ft", "csv"); out = M / f"zen_new_{ft}.jsonl"
    seen = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    f = open(out, "a"); day = dt.date.fromisoformat(arg.get("since", "2026-11-02")); end = dt.date.fromisoformat(arg.get("until", str(dt.date.today())))
    while day <= end:
        q = urllib.parse.quote(f"created:[{day} TO {day}]")
        for page in range(1, 200):
            try:
                d = json.loads(get(f"https://zenodo.org/api/records?q={q}&type=dataset&file_type={ft}&size=25&page={page}&sort=oldest"))
            except Exception as e:
                print("list err", day, page, type(e).__name__, flush=True); break
            hits = d["hits"]["hits"]
            for x in hits:
                if str(x["id"]) in seen:
                    continue
                md = x.get("metadata", {}); vers = (md.get("relations") or {}).get("version") or [{}]
                lic = (md.get("license") or {}).get("id", "") if isinstance(md.get("license"), dict) else str(md.get("license", ""))
                rec = dict(id=str(x["id"]), concept=str(x.get("conceptrecid", "")), created=x.get("created", "")[:10], pubdate=md.get("publication_date", ""),
                           version_index=vers[0].get("index"), access=md.get("access_right"), lic=lic, title=md.get("title", ""),
                           desc=re.sub(r"\s+", " ", re.sub("<[^>]+>", " ", md.get("description", "") or ""))[:3000], doi=x.get("doi"),
                           files=[dict(key=fi["key"], size=fi.get("size", 0), url=fi["links"]["self"]) for fi in x.get("files", [])])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); seen.add(rec["id"])
            f.flush(); time.sleep(1.1)
            if len(hits) < 25:
                break
        day += dt.timedelta(days=1)
    print("LISTDONE", ft, len(seen), flush=True)

def probe_all(lists):
    from simaudit.data.xlsx_lite import read_sheet
    sh, ns = int(arg.get("shard", 0)), int(arg.get("nshard", 1))
    pf = M / f"zen_probe_{arg.get('tag', 'x')}_{sh}.csv"; F = ["rec", "file", "sheet", "size", "n", "d", "cands", "cols", "sha256", "mech", "err"]
    done = {(r["rec"], r["file"]) for r in csv.DictReader(open(pf))} if pf.exists() else set()
    # skip files already probed by E7 (read-only)
    S2 = H.parents[2] / "S2_k5_fresh_2026-11-07/harvest"   # CONF3: also skip files probed by the S2 harvest (read-only)
    for g in list(E7.glob("meta/zen_probe_*.csv")) + list(E7.glob("meta/zen_xlsx_probe_*.csv")) + list(S2.glob("meta/zen_probe_*.csv")):
        done |= {(r["rec"], r["file"]) for r in csv.DictReader(open(g))}
    new = not pf.exists(); fh = open(pf, "a", newline=""); w = csv.DictWriter(fh, fieldnames=F)
    if new:
        w.writeheader()
    res = reserved_ids(); recs = {}
    for lf in lists:
        for l in open(lf):
            r = json.loads(l); recs.setdefault(r["id"], r)
    recs = [r for r in recs.values() if int(r["id"]) % ns == sh and r["created"] >= CUT and (r.get("pubdate") or "9") >= CUT
            and r.get("access") in (None, "open") and r["id"] not in res and r.get("concept") not in res]
    minsc = int(arg.get("minscore", 1))
    recs = [r for r in recs if score(r["title"] + " " + r.get("desc", "")) >= minsc]
    recs.sort(key=lambda r: -score(r["title"] + " " + r.get("desc", "")))
    print("records to probe", len(recs), flush=True); k = 0
    for r in recs:
        for fi in r["files"][:12]:
            ext = fi["key"].lower().rsplit(".", 1)[-1]
            if ext not in ("csv", "tsv", "xlsx") or (r["id"], fi["key"]) in done:
                continue
            row = dict(rec=r["id"], file=fi["key"], size=fi["size"], mech=0, err="")
            if fi["size"] > CAP or fi["size"] < 4000:
                row["err"] = "size"; w.writerow(row); continue
            d = RAW / f"zenodo_{r['id']}"; d.mkdir(exist_ok=True); p = d / re.sub(r"[^A-Za-z0-9._-]", "_", Path(fi["key"]).name)
            keep = False
            try:
                b = get(fi["url"], cap=CAP); p.write_bytes(b); time.sleep(1.1)
                row["sha256"] = hashlib.sha256(b).hexdigest()
                prs = []
                if ext == "xlsx":
                    for s in range(3):
                        try:
                            prs.append((s, probe_grid(read_sheet(p, s))))
                        except Exception:
                            break
                else:
                    prs.append(("", probe_csv(p)))
                for s, pr in prs:
                    k3 = any(int(m) >= 3 for m in re.findall(r"\((\d+):\d+\)", pr["cands"]))
                    rr = dict(row, sheet=s, **pr); rr["mech"] = mech3(pr)
                    keep |= bool(rr["mech"]); w.writerow(rr)
            except Exception as e:
                row["err"] = (type(e).__name__ + ":" + str(e))[:80]; w.writerow(row)
            if not keep:
                p.unlink(missing_ok=True)
                try:
                    d.rmdir()
                except OSError:
                    pass
            fh.flush(); k += 1
            if k % 100 == 0:
                print(time.strftime("%H:%M:%S"), "probed", k, flush=True)
    print("PROBEDONE", k, flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "list":
        listing()
    else:
        probe_all([a for a in sys.argv[2:] if "=" not in a])
