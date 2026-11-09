"""Mendeley public API: licence, version, publish date for dataset ids (no login). Appends meta/mendeley_meta.json"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import H, get
out = H / "meta/mendeley_meta.json"; D = json.loads(out.read_text()) if out.exists() else {}
for i in sys.argv[1:]:
    i = i.replace("mnd", "")
    try:
        x = json.loads(get(f"https://data.mendeley.com/public-api/datasets/{i}"))
        D[i] = dict(version=x.get("version"), licence=(x.get("data_licence") or {}).get("short_name") or (x.get("data_licence") or {}).get("full_name"),
                    publish_date=x.get("publish_date"), first=x.get("created_date") or x.get("publish_date"), keys=sorted(x)[:40])
    except Exception as e:
        D[i] = dict(err=str(e)[:80])
    print(i, {k: v for k, v in D[i].items() if k != "keys"}, flush=True); out.write_text(json.dumps(D, indent=0)); time.sleep(1.5)
