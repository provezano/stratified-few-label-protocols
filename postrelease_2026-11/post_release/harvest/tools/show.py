"""show REC... : metadata (title, desc, dates, licence, files) + probed files' columns and stored candidate counts. Metadata only."""
import csv, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
exec(open(Path(__file__).resolve().parent / "pool.py").read().split("R = common.reserved_ids()")[0])   # builds M (metadata index)
H = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(H / "meta/cands_new.csv"))) + list(csv.DictReader(open(H / "meta/pool_extra.csv")))
L = int(next((a[2:] for a in sys.argv if a.startswith("L=")), 500))
for w in [a for a in sys.argv[1:] if not a.startswith("L=")]:
    m = M.get(w, {}); tl = m.get("timeline") or {}
    print(f"######## {w} concept={m.get('concept')} created={m.get('created') or tl.get('posted')} pub={m.get('pubdate')} vi={m.get('version_index', m.get('version'))} lic={m.get('lic') or (m.get('license') or {}).get('name') if isinstance(m.get('license'), dict) else m.get('lic')} doi={m.get('doi')}")
    print("TITLE:", m.get("title")); print("DESC:", (m.get("desc") or m.get("description") or "")[:L])
    print("FILES:", [f.get("key") or f.get("name") for f in m.get("files", [])][:12])
    seen = set()
    for x in rows:
        if x["rec"] == w and (x["file"], x.get("sheet")) not in seen:
            seen.add((x["file"], x.get("sheet")))
            print(f"  -> {x['file']}[{x.get('sheet','')}] n={x['n']} d={x['d']} local={x.get('local','')}\n     COLS: {x['cols'][:1500]}\n     CANDS: {x['cands'][:600]}")
