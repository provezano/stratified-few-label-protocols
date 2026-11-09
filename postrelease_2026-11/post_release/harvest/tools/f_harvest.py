"""figshare route. modes:
  list since=YYYY-MM-DD -> meta/fs_new_list.jsonl (item_type=3 published since)
  select -> meta/fs_sel.txt (post-cut, not E7-detailed, title score) ; detail -> meta/fs_detail.jsonl ; probe -> meta/fs_probe.csv
Outcome-blind (shape, column names, distinct-value counts). ~1 req/s to figshare."""
import csv, hashlib, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, "simaudit/src")
from common import H, E7, CUT, get, reserved_ids, score, probe_csv, probe_grid, mech3
M = H / "meta"; RAW = H / "raw"; CAP = 4 * 2**20
arg = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
DATA = re.compile(r"data ?sheet|dataset|data set|raw data|minimal data|\bS\d+ (data|file|table)|database|supplementary (data|file)|underlying data|data file|\.csv|\.xlsx", re.I)

def listing():
    f = open(M / "fs_new_list.jsonl", "w"); seen = set(); since = arg.get("since", "2026-11-03")
    for page in range(1, 30):
        d = json.loads(get(f"https://api.figshare.com/v2/articles?item_type=3&published_since={since}&order=published_date&order_direction=asc&page_size=1000&page={page}"))
        for x in d:
            if x["id"] not in seen:
                seen.add(x["id"]); f.write(json.dumps(dict(id=x["id"], title=x["title"], doi=x.get("doi"), published=x.get("published_date"), timeline=x.get("timeline")), ensure_ascii=False) + "\n")
        f.flush(); time.sleep(1.5)
        if len(d) < 1000:
            break
    print("LIST", len(seen), flush=True)

def select():
    done = set()
    S2 = H.parents[2] / "S2_k5_fresh_2026-11-07/harvest"   # CONF3: skip items detailed by E7 or S2 (their probes are mined separately)
    for p in (E7 / "meta/fs_detail.jsonl", E7 / "meta_b/fs_detail.jsonl", S2 / "meta/fs_detail.jsonl"):
        done |= {json.loads(l)["id"] for l in open(p)}
    res = reserved_ids(); X = {}
    for p in (E7 / "meta/fs_list.jsonl", H.parents[2] / "S2_k5_fresh_2026-11-07/harvest/meta/fs_new_list.jsonl", M / "fs_new_list.jsonl"):
        if p.exists():
            for l in open(p):
                x = json.loads(l); X.setdefault(x["id"], x)
    out = []
    for x in X.values():
        if (x.get("published") or "") < CUT or x["id"] in done or str(x["id"]) in res:
            continue
        t = re.sub("<[^>]+>", " ", x["title"]); s = score(t)
        if s >= 2 or (s >= 1 and DATA.search(t)):
            out.append((-s, x["id"]))
    out.sort(); (M / "fs_sel.txt").write_text("\n".join(str(i) for _, i in out))
    print("selected", len(out), flush=True)

def detail():
    out = M / "fs_detail.jsonl"
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    f = open(out, "a")
    for i in [int(x) for x in (M / "fs_sel.txt").read_text().split()]:
        if i in done:
            continue
        try:
            x = json.loads(get(f"https://api.figshare.com/v2/articles/{i}"))
            rec = dict(id=i, title=x.get("title"), doi=x.get("doi"), version=x.get("version"), timeline=x.get("timeline"), created=x.get("created_date"),
                       lic=(x.get("license") or {}).get("name"), desc=re.sub(r"\s+", " ", re.sub("<[^>]+>", " ", x.get("description") or ""))[:3000],
                       files=[dict(key=fi["name"], size=fi.get("size", 0), url=fi.get("download_url"), md5=fi.get("computed_md5")) for fi in x.get("files", [])])
        except Exception as e:
            rec = dict(id=i, err=type(e).__name__)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); time.sleep(1.0)
    print("DETAILDONE", flush=True)

def probe_all():
    from simaudit.data.xlsx_lite import read_sheet
    pf = M / "fs_probe.csv"; F = ["rec", "file", "sheet", "size", "n", "d", "cands", "cols", "sha256", "mech", "err"]
    done = {(r["rec"], r["file"]) for r in csv.DictReader(open(pf))} if pf.exists() else set()
    new = not pf.exists(); fh = open(pf, "a", newline=""); w = csv.DictWriter(fh, fieldnames=F)
    if new:
        w.writeheader()
    for l in open(M / "fs_detail.jsonl"):
        r = json.loads(l)
        tl = r.get("timeline") or {}
        if (tl.get("posted") or tl.get("firstOnline") or "9")[:10] < CUT:
            continue
        for fi in r.get("files", [])[:12]:
            ext = fi["key"].lower().rsplit(".", 1)[-1]
            if ext not in ("csv", "tsv", "xlsx", "txt") or (str(r["id"]), fi["key"]) in done:
                continue
            row = dict(rec=r["id"], file=fi["key"], size=fi["size"], mech=0, err="")
            if fi["size"] > CAP or fi["size"] < 3000:
                row["err"] = "size"; w.writerow(row); continue
            d = RAW / f"figshare_{r['id']}"; d.mkdir(exist_ok=True); p = d / re.sub(r"[^A-Za-z0-9._-]", "_", fi["key"])
            keep = False
            try:
                b = get(fi["url"], cap=CAP); p.write_bytes(b); time.sleep(1.0)
                row["sha256"] = hashlib.sha256(b).hexdigest(); prs = []
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
            fh.flush()
    print("FSPROBEDONE", flush=True)

if __name__ == "__main__":
    {"list": listing, "select": select, "detail": detail, "probe": probe_all}[sys.argv[1]]()
