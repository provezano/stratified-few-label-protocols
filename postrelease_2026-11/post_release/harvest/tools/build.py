"""P1 confirm frame build (adapted from CONF3 build3.py / S2 build.py, 2026-11-07). No model; the only label statistic is the
per-class count. Rule: ../INCLUSION_RULE.txt (+ Amendment 1).
 1. reads curation/decisions.py ACCEPT (in order.csv order); raw file from a local read-only probe copy (E7/S2/CONF3/semprior)
    or fetched into raw/<src>_<id>/ (paced; URL from the listing metadata);
 2. encodes with simaudit.data.second_frame (Study 15 rule, unchanged): 2 <= C <= 10, smallest class >= 11, 100 <= n <= 5000,
    1 <= d <= 100;
 3. exclusions: E7 builder reserved() (pool_i2 DEV+HELD-OUT hash list, T1', Study 15/16/17/18, T2, semprior, posfm) + reference
    row-overlap arrays (DEV, T1') + E7 frame npz + G4 frame (all rows) and conf2 npz + S2 frame npz (data/npz) + S2 harvest npz +
    CONF3 npz: content hash, row overlap >= 0.5, signature (|n-n'| <= 2, d = d', C = C'); within-frame content/raw duplicates;
 4. writes npz/<key>.npz (second_frame.save_npz), specs.jsonl, curation/build_report.csv.
Run: OMP_NUM_THREADS=1 nice -n 15 simaudit/.venv/bin/python -I tools/build.py"""
import csv, glob, hashlib, importlib.util, json, re, shutil, sys, time, zlib
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, "simaudit/src")
from common import H, E7, SP, G, PROBES, get, CUT
from simaudit.data import second_frame as sf
from simaudit.data.openml_loader import content_hash
ROOT = G / "simaudit"; NPZ = H / "npz"; NPZ.mkdir(exist_ok=True)
S2 = PROBES / "S2_k5_fresh_2026-11-07"; G4 = PROBES / "class_balancing_g4_gates_2026-11-05"; C3 = G4 / "conf3/harvest"
sf.SOURCE_OFFSET.update({"Dataverse": 4 * 10**10, "Mendeley": 5 * 10**10})
SRC = {"zenodo": "Zenodo", "figshare": "figshare", "dataverse": "Dataverse", "mendeley": "Mendeley"}
safe = lambda s: re.sub(r"[^A-Za-z0-9._-]", "_", s)
rawdir = lambda src, rec: H / "raw" / f"{src}_{re.sub(r'[^A-Za-z0-9]', '', str(rec))}"
LOCAL_ROOTS = [E7 / "raw_zenodo", E7 / "raw_zenodo_xlsx", E7 / "raw_figshare", E7 / "tables", S2 / "harvest/raw", C3 / "raw", SP / "raw", SP / "harvest/raw"]

def load_bt():
    spec = importlib.util.spec_from_file_location("b", E7 / "tools/build_e7.py"); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m

def meta_index():
    M = {}
    for f in [SP / "harvest/cands_zenodo.jsonl", SP / "harvest/cands_zenodox.jsonl", E7 / "meta/zen_xlsx_cands.jsonl", SP / "harvest/cands_dataverse.jsonl",
              SP / "harvest/cands_mendeley.jsonl", SP / "harvest/cands_figshare.jsonl", E7 / "meta/fs_detail.jsonl", E7 / "meta_b/fs_detail.jsonl",
              S2 / "harvest/meta/fs_detail.jsonl", S2 / "harvest/meta/zen_new_csv.jsonl", S2 / "harvest/meta/zen_new_xlsx.jsonl", C3 / "meta/fs_detail.jsonl",
              H / "meta/fresh_listing.jsonl"]:
        if f.exists():
            for l in open(f):
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                M.setdefault(str(r["id"]), r)
    return M

def jload(p):
    return json.loads(p.read_text()) if p.exists() else {}

def first_public(src, rec, m):
    """(date or None, how verified)."""
    if src == "zenodo":
        f = jload(H / "meta/zen_first.json").get(rec, {})
        if f.get("first_created"):
            d = min(f["first_created"], f["first_pub"])
            return d, f"Zenodo versions API: {f['n']} version(s) of the concept, first id {f['first_id']}, earliest created {f['first_created']}, earliest publication_date {f['first_pub']}"
        return None, f"Zenodo versions API not available ({f.get('err', 'not queried')})"
    if src == "figshare":
        f = jload(H / "meta/fs_first.json").get(rec, {})
        if f.get("v1_posted"):
            return f["v1_posted"], f"figshare versions API: {f['n_versions']} version(s); v1 posted {f['v1_posted']} (published_date {f['v1_published']})"
        return None, f"figshare versions API not available ({f.get('err', 'not queried')})"
    if src == "mendeley":
        f = jload(H / "meta/mendeley_meta.json").get(rec.replace("mnd", ""), {})
        if f.get("version") == 1 and f.get("publish_date"):
            return f["publish_date"][:10], f"Mendeley public API: version 1, publish_date {f['publish_date'][:10]}"
        return None, f"Mendeley public API: version {f.get('version')} ({f.get('err', '')})"
    d = min(m.get("created") or "9", m.get("pubdate") or "9")
    return (d if d != "9" else None), f"Dataverse listing: created {m.get('created')}, published {m.get('pubdate')} (both must be >= cut)"

def licence(src, rec, m):
    if src == "mendeley":
        return jload(H / "meta/mendeley_meta.json").get(rec.replace("mnd", ""), {}).get("licence")
    return m.get("lic")

def fetch(src, rec, fname, m):
    d = rawdir(src, rec); p = d / safe(fname)
    if p.exists():
        return p, "fetched earlier"
    key = re.sub(r"^(mnd|dvn|dv_[a-z]+_)", "", rec)
    for root in LOCAL_ROOTS:
        for q in list(root.glob(f"*{key}*/{safe(fname)}")) + list(root.glob(f"*{key}*/{fname}")):
            d.mkdir(parents=True, exist_ok=True); shutil.copy2(q, p); return p, f"local copy {q}"
    url = next((f.get("url") for f in m.get("files", []) if f.get("key") == fname), None)
    if url is None:
        raise FileNotFoundError(f"no url for {fname}")
    d.mkdir(parents=True, exist_ok=True); p.write_bytes(get(url, cap=40 * 2**20)); time.sleep(1.5)
    return p, url

def npz_xy(p):
    z = np.load(p, allow_pickle=False); return np.asarray(z["X"], dtype=np.float64), np.asarray(z["y"]).astype(np.int64)

def references():
    bt = load_bt(); hashes, sigs = bt.reserved(); arrays = sf.reference_arrays(ROOT)
    def add_npz(tag, p):
        try:
            X, y = npz_xy(p)
        except Exception:
            return
        hashes[f"{tag}:{p.stem}"] = content_hash(X, y); arrays[f"{tag}:{p.stem}"] = X
        sigs.append((f"{tag}:{p.stem}", X.shape[0], X.shape[1], len(np.unique(y))))
    for tag, d in [("e7", E7 / "npz"), ("g4conf2", G4 / "conf2/data"), ("s2frame", S2 / "data/npz"), ("s2harvest", S2 / "harvest/npz"), ("conf3", C3 / "npz")]:
        for p in sorted(d.glob("*.npz")):
            add_npz(tag, p)
    for r in csv.DictReader(open(G4 / "conf2/frame_conf2.csv")):
        hashes[f"g4conf2csv:{r['key']}"] = r["sha256"]
    for r in csv.DictReader(open(G4 / "out/frame/frame.csv")):
        if r.get("chash"):
            hashes[f"g4frame:{r['key']}"] = r["chash"]
        if r["n"] and r["d"] and r["K"]:
            sigs.append((f"g4frame:{r['key']}", int(float(r["n"])), int(float(r["d"])), int(float(r["K"]))))
        if r["path"] and Path(r["path"]).is_file():
            add_npz("g4framenpz", Path(r["path"]))
    for f in [S2 / "frame.csv"]:
        for r in csv.DictReader(open(f)):
            if r.get("n") and r.get("d") and r.get("K"):
                sigs.append((f"s2framecsv:{r['key']}", int(float(r["n"])), int(float(r["d"])), int(float(r["K"]))))
    for f in [S2 / "harvest/manifest.csv", E7 / "manifest.csv", C3 / "curation/build_report.csv"]:
        for r in csv.DictReader(open(f)):
            for hk in ("content_sha256", "sha256"):
                if r.get(hk):
                    hashes[f"{f.parent.name}/{f.stem}:{r.get('id') or r.get('key')}"] = r[hk]
    return hashes, sigs, arrays, bt

def main():
    s = importlib.util.spec_from_file_location("dec", H / "curation/decisions.py"); dec = importlib.util.module_from_spec(s); s.loader.exec_module(dec)
    hashes, sigs, arrays, bt = references()
    print(len(hashes), "reference hashes;", len(sigs), "signatures;", len(arrays), "overlap arrays", flush=True)
    M = meta_index(); out, specs, seen, seen_raw = [], [], {}, {}
    for (src, rec), a in dec.ACCEPT.items():
        m = M.get(rec, {}); fp, how = first_public(src, rec, m); lic = licence(src, rec, m)
        doi = str(m.get("doi") or "").replace("doi:", "") or (f"10.5281/zenodo.{rec}" if src == "zenodo" else "")
        url = {"zenodo": f"https://zenodo.org/records/{rec}", "figshare": f"https://figshare.com/articles/dataset/{rec}",
               "mendeley": f"https://data.mendeley.com/datasets/{rec.replace('mnd', '')}/1", "dataverse": f"https://doi.org/{doi}"}[src]
        spec_out = dict(key=a["key"], pos=a.get("pos"), source=src, recid=rec, doi=doi, url=url, title=m.get("title"), first_public=fp, first_public_how=how,
                        licence=lic, file=a["file"], sheet=a.get("sheet"), target=a["target"], why_target=a["why"], drop=a.get("drop", []), drop_reasons=a.get("dropwhy", ""))
        rep = dict(key=a["key"], pos=a.get("pos"), source=src, record=rec, url=url, doi=doi, first_public=fp, first_public_how=how, licence=lic, file=a["file"], target=a["target"])
        try:
            p, got = fetch(src, rec, a["file"], m)
            num = rec if rec.isdigit() else str(zlib.crc32(rec.encode()))
            base = sf.Spec(key=a["key"], source=SRC[src], record=num, first_public=fp or "", license=lic or "", parts=(sf.Part(p.name, a.get("sheet"), int(a.get("header", 0))),), target=a["target"])
            header, rows = sf.read_table(base, p.parent)
            tgt = bt.resolve(header, a["target"])[0]
            drop = tuple(c for d in a.get("drop", []) for c in bt.resolve(header, d) if c != tgt)
            spec = sf.Spec(key=base.key, source=base.source, record=base.record, first_public=base.first_public, license=base.license, parts=base.parts, target=tgt, drop=drop)
            X, y, info = sf.encode(spec, header, rows)
        except Exception as exc:
            out.append(dict(rep, excluded=f"RU: unreadable: {exc}"[:200])); specs.append(spec_out); print(a["key"], "UNREADABLE", exc, flush=True); continue
        chk = sf.overlap_checks(X, y, hashes, arrays)
        cc = info["class_counts"]; why = None
        if fp is None or fp < CUT:
            why = f"R0: first public {fp} < {CUT} or not verifiable as first version ({how})"
        elif not (lic and re.search(r"cc|odc|etalab|licence ouverte|open|public domain", lic, re.I)) or re.search(r"-nd|nd-|noderiv", lic or "", re.I):
            why = f"R1: licence {lic!r} not verified open"
        elif sf.ineligible(info):
            why = "R2: " + sf.ineligible(info)
        elif info["n"] > 5000:
            why = f"R2: n = {info['n']} > 5000"
        elif info["d"] > 100:
            why = f"R2: d = {info['d']} > 100"
        elif chk["hash_match"]:
            why = f"RX: content hash equals {chk['hash_match']}"
        elif chk["max_overlap"] >= sf.OVERLAP_LIMIT:
            why = f"RX: row overlap {chk['max_overlap']:.2f} with {chk['max_overlap_with']}"
        raw_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if why is None and chk["sha256"] in seen:
            why = f"R5r: within-frame duplicate of {seen[chk['sha256']]}"
        if why is None and raw_sha in seen_raw:
            why = f"R5r: within-frame raw duplicate of {seen_raw[raw_sha]}"
        sig = [k for k, n, d, c in sigs if abs(n - info["n"]) <= 2 and d == info["d"] and c in (info["C"], -1)]
        if why is None and sig:
            why = f"RX: signature match {'|'.join(sig)[:120]}"
        if why is None:
            sf.save_npz(NPZ / f"{a['key']}.npz", X, y, a["key"]); seen[chk["sha256"]] = a["key"]; seen_raw[raw_sha] = a["key"]
        kinds = list(info["features"].values())
        spec_out.update(drop_resolved=list(drop), class_mapping=info["labels"], n=info["n"], d=info["d"], C=info["C"], class_counts=cc, raw_source=got)
        spec_out.update(excluded=why or "")
        if why is None:
            specs.append(spec_out)  # specs.jsonl = eligible tables only; every judged ACCEPT is in build_report.csv
        hc = (f"no hash match among {len(hashes)} reference hashes; max row overlap {chk['max_overlap']:.3f} ({chk['max_overlap_with'] or '-'}); "
              f"{len(sig)} signature hit(s)") if not chk["hash_match"] else f"hash match {chk['hash_match']}"
        out.append(dict(rep, n=info["n"], d=info["d"], C=info["C"], class_counts=";".join(map(str, cc)), n_numeric=kinds.count("numeric"),
                        n_nominal=kinds.count("nominal"), dropped=";".join(f"{k}:{v}" for k, v in info["dropped"].items())[:800],
                        sha256=chk["sha256"], raw_sha256=raw_sha, max_overlap=round(chk["max_overlap"], 3), max_overlap_with=chk["max_overlap_with"],
                        signature_flags="|".join(sig)[:300], hash_check=hc, excluded=why or ""))
        print(a.get("pos"), a["key"], fp, lic, info["n"], info["d"], info["C"], cc, why or "OK", flush=True)
    F = ["pos", "key", "source", "record", "url", "doi", "first_public", "first_public_how", "licence", "file", "target", "n", "d", "C", "class_counts",
         "n_numeric", "n_nominal", "dropped", "sha256", "raw_sha256", "max_overlap", "max_overlap_with", "signature_flags", "hash_check", "excluded"]
    with open(H / "curation/build_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=F); w.writeheader(); w.writerows(out)
    (H / "specs.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False, default=str) for s in specs) + "\n")
    print("ELIGIBLE", sum(1 for r in out if not r["excluded"]))

if __name__ == "__main__":
    main()
