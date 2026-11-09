"""First-version date of Zenodo records (versions API, oldest first). args: record ids. Appends to meta/zen_first.json"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import H, get
out = H / "meta/zen_first.json"; D = {k: v for k, v in (json.loads(out.read_text()) if out.exists() else {}).items() if "err" not in v}
for rid in sys.argv[1:]:
    if rid in D:
        print(rid, D[rid]); continue
    try:
        d = json.loads(get(f"https://zenodo.org/api/records/{rid}/versions?size=25&sort=version"))
        hs = d["hits"]["hits"]
        D[rid] = dict(first_id=str(hs[0]["id"]), first_created=min(h["created"][:10] for h in hs),
                      first_pub=min(h["metadata"].get("publication_date", "9") for h in hs), n=len(hs))
    except Exception as e:
        D[rid] = dict(err=type(e).__name__ + str(e)[:60])
    print(rid, D[rid], flush=True); out.write_text(json.dumps(D, indent=0)); time.sleep(1.5)
