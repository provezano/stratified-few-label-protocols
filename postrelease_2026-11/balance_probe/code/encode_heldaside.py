"""Held-aside check tables: the four post-release tables the confirmation harvest accepted on judgement but excluded at build
for capacity only (R2: n > 5000 or d > 100). Never fitted by any study. Re-encodes them from the harvest's local raw copies
(read-only, no download) with the harvest's own rule (simaudit second_frame.encode, decisions.py spec), checks that the content
hash equals the one in build_report.csv (which also records the HELD-OUT / reference exclusion check), writes heldaside/npz and
plan/frame.csv (39 confirmation tables + 4 held-aside). No model, no label statistic beyond class counts."""
import csv, importlib.util, sys, zlib
from pathlib import Path
import numpy as np
import pandas as pd
sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
C = B.parent / "confirm_2026-11-08"; H = C / "harvest"
sys.path.insert(0, str(H / "tools")); sys.path.insert(0, "simaudit/src")
import build as hb  # harvest build helpers (rawdir, safe, load_bt); main() is not called
from simaudit.data import second_frame as sf
from simaudit.data.openml_loader import content_hash
KEYS = ["z23223338_vte", "z23474900_treephyto", "m_4nr4xmbxnb_elevbp", "d_TVKRXD_hivst"]
OFF = {"zenodo": 10**10, "figshare": 2 * 10**10, "kaggle": 3 * 10**10, "dataverse": 4 * 10**10, "mendeley": 5 * 10**10}
rep = pd.read_csv(H / "curation/build_report.csv", dtype=str, keep_default_na=False).set_index("key")
s = importlib.util.spec_from_file_location("dec", H / "curation/decisions.py"); dec = importlib.util.module_from_spec(s); s.loader.exec_module(dec)
bt = hb.load_bt()
rows = []
for (src, rec), a in dec.ACCEPT.items():
    if a["key"] not in KEYS:
        continue
    r = rep.loc[a["key"]]
    assert r.excluded.startswith("R2:") and r.hash_check.startswith("no hash match"), (a["key"], r.excluded, r.hash_check)
    p = hb.rawdir(src, rec) / hb.safe(a["file"])
    assert p.exists(), p
    num = rec if rec.isdigit() else str(zlib.crc32(rec.encode()))
    base = sf.Spec(key=a["key"], source=hb.SRC[src], record=num, first_public=r.first_public, license=r.licence,
                   parts=(sf.Part(p.name, a.get("sheet"), int(a.get("header", 0))),), target=a["target"])
    header, data = sf.read_table(base, p.parent)
    tgt = bt.resolve(header, a["target"])[0]
    drop = tuple(c for d in a.get("drop", []) for c in bt.resolve(header, d) if c != tgt)
    spec = sf.Spec(key=base.key, source=base.source, record=base.record, first_public=base.first_public, license=base.license,
                   parts=base.parts, target=tgt, drop=drop)
    X, y, info = sf.encode(spec, header, data)
    h = content_hash(np.asarray(X, dtype=np.float64), np.asarray(y).astype(np.int64))
    assert h == r.sha256, (a["key"], h, r.sha256)
    sf.save_npz(B / "heldaside/npz" / f"{a['key']}.npz", X, y, a["key"])
    rows.append(dict(key=a["key"], did=OFF[src] + (int(rec) if rec.isdigit() else zlib.crc32(rec.encode())), sha256=h,
                     class_counts=r.class_counts, n=int(r.n), d=int(r.d), C=int(r.C), set="heldaside",
                     npz=str(B / "heldaside/npz" / f"{a['key']}.npz"), why_held=r.excluded))
    print(a["key"], info["n"], info["d"], info["C"], "hash OK", flush=True)
assert len(rows) == 4
fr = pd.read_csv(C / "plan/frame.csv")
res = pd.read_csv(C / "results/fresh_pertable.csv")
fr = fr[fr.key.isin(res.key)]
assert len(fr) == 39
main = fr[["key", "did", "sha256", "class_counts", "n", "d", "C"]].assign(set="confirm", npz=[str(H / "npz" / f"{k}.npz") for k in fr.key], why_held="")
out = pd.concat([main, pd.DataFrame(rows)], ignore_index=True)
assert out.did.is_unique and out.sha256.is_unique
out.to_csv(B / "plan/frame.csv", index=False)
print(len(out), "tables in plan/frame.csv")
