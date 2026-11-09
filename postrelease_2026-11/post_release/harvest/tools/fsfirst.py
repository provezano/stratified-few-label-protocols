"""figshare v1 posted date via versions API. args: ids. Writes meta/fs_first.json"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import H, get
out = H / "meta/fs_first.json"; D = json.loads(out.read_text()) if out.exists() else {}
for i in sys.argv[1:]:
    try:
        v = json.loads(get(f"https://api.figshare.com/v2/articles/{i}/versions")); time.sleep(1.2)
        v1 = json.loads(get(f"https://api.figshare.com/v2/articles/{i}/versions/1")); time.sleep(1.2)
        tl = v1.get("timeline") or {}
        D[i] = dict(n_versions=len(v), v1_posted=(tl.get("posted") or tl.get("firstOnline") or v1.get("published_date") or "")[:10], v1_published=(v1.get("published_date") or "")[:10])
    except Exception as e:
        D[i] = dict(err=str(e)[:80])
    print(i, D[i], flush=True); out.write_text(json.dumps(D, indent=0))
