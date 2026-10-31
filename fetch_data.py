"""Fetch the public tables that the analyses read, and check each against its pinned hash.

Nothing third-party is shipped in this package; this script rebuilds the local data cache instead.

Run from simaudit/ (after `uv sync --frozen`):
    uv run python ../fetch_data.py                 # DEV (OpenML) and T1' (PMLB, UCI) tables
    uv run python ../fetch_data.py --case-study    # also the five OpenML ARFF files of the TabEBM case study

Sources and pins
- DEV: 20 OpenML datasets by id (an OpenML id names one dataset version); ids, versions and content
  hashes in ../datasets/dev.csv. Written to data/cache/npz/<id>.npz by simaudit.data.openml_loader.
- T1' PMLB tables: the PMLB repository at the commit pinned in simaudit.runner.build_pool_pmlb
  (PMLB_COMMIT; also in ../datasets/sources.json), Git LFS content checked against its oid.
- T1' UCI tables: the official UCI API (per-dataset metadata, then the data file under
  archive.ics.uci.edu/static/public/).
  Every T1' table is checked against its file SHA-256 and content SHA-256 in ../datasets/t1prime.csv.
- Case study: OpenML ARFF files by file id, checked against the SHA-256 below (the same files as
  paper/p1/exploratory/case_study/PREREG.md, Sec. 8). Table 841 (stock) is a DEV table (npz above).

A mismatch is reported and the script exits with status 1; a changed upstream file is never used
silently.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from openml.exceptions import OpenMLServerException

from simaudit.data.openml_loader import load
from simaudit.runner import build_pool_pmlb as bpp
from simaudit.runner import build_pool_uci as bpu
from simaudit.settings import settings

ROOT = Path(__file__).resolve().parent
CASE = Path("paper/p1/exploratory/case_study/third_party")
# OpenML did -> (file id, file name, SHA-256 of the ARFF file)
ARFF = {
    40966: (17928620, "MiceProtein", "f48ced98ecfe8ff056d8ea3507e067258c82f29199a1af2261b516010827cc4d"),
    14: (14, "mfeat-fourier", "bc5dd56b0cf2ee0ced33c0cc17d82b855c3eef43e1644b3969e9b44d2b2e3933"),
    1472: (1587929, "energy-efficiency", "8e2ca7aabe95e1f29deb432ff8c52f27c23a090172859fafad535e18ecd82fa4"),
    40971: (17953251, "collins", "05fc0a3f30d2c938f72100281abfee9276c582e2ef4b3cc32524305eec64de6a"),
    40499: (4535764, "texture", "9143690415c61a8d9c33ededf947126fa71cced8f91d50642f06621f84e3e9bd"),
}


def transient(e: Exception) -> bool:
    """Network errors, and OpenML's "database connection error" (code 107) under high server load."""
    if isinstance(e, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return True
    return isinstance(e, OpenMLServerException) and getattr(e, "code", None) == 107


def retry(f, tries: int = 8, wait: float = 30.0):
    """Retry a network call on transient errors, waiting longer each time."""
    for i in range(tries):
        try:
            return f()
        except Exception as e:  # noqa: BLE001 - re-raised unless transient
            if not transient(e) or i == tries - 1:
                raise
            print(f"  retry {i + 1} after {e}")
            time.sleep(wait * (i + 1))


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-study", action="store_true")
    args = ap.parse_args()
    cache = settings.data_cache_dir
    bad: list[str] = []

    dev = pd.read_csv(ROOT / "datasets" / "dev.csv")
    for r in dev.itertuples():
        d = retry(lambda: load(int(r.openml_did), cache))
        ok = d.sha256 == r.content_sha256
        print(f"DEV {r.openml_did:>6} {d.name:<30} {'ok' if ok else 'HASH MISMATCH'}")
        ok or bad.append(f"DEV {r.openml_did}")

    t1 = pd.read_csv(ROOT / "datasets" / "t1prime.csv")
    pmlb_fetch, uci_fetch = bpp.fetcher(cache), bpu.fetcher()
    for r in t1.itertuples():
        if r.source == "pmlb":
            d, file_sha = retry(lambda: bpp.load_pmlb(r.key.split(":", 1)[1], cache, pmlb_fetch))
        else:
            npz = cache / "npz" / f"{bpu.uci_did(int(r.uci_id))}.npz"
            meta = {"uci_id": int(r.uci_id), "name": r.name}  # enough when the table is cached
            if not npz.exists():
                meta = dict(retry(lambda: bpu.http_json(bpu.META_URL.format(uid=int(r.uci_id))))["data"])
            d, file_sha = retry(lambda: bpu.load_uci(meta, cache, uci_fetch))
        ok = d.sha256 == r.content_sha256 and file_sha == r.file_sha256 and d.did == r.internal_did
        print(f"T1' {r.key:<45} {'ok' if ok else 'HASH OR ID MISMATCH'}")
        ok or bad.append(f"T1' {r.key}")

    if args.case_study:
        for did, (fid, name, want) in ARFF.items():
            path = CASE / f"openml_{did}" / f"{name}.arff"
            url = f"https://www.openml.org/data/v1/download/{fid}"
            # OpenML can answer 200 with an error text under load: only a file with the pinned hash is kept
            for attempt in range(6):
                if path.exists() and sha256(path.read_bytes()) == want:
                    break
                raw = retry(lambda: urllib.request.urlopen(url, timeout=120).read())
                if sha256(raw) == want:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
                    break
                print(f"  ARFF {did}: got {len(raw)} bytes that do not match the pinned hash; retrying")
                time.sleep(15 * (attempt + 1))
            ok = path.exists() and sha256(path.read_bytes()) == want
            print(f"ARFF {did:>6} {name:<20} {'ok' if ok else 'HASH MISMATCH'}")
            ok or bad.append(f"ARFF {did}")

    print("all tables match their pinned hashes" if not bad else f"MISMATCHES: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
