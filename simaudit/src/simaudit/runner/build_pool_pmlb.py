"""Study 8 Amendment 1: PMLB tiers 2a and 3 (prereg/STUDY8_EXPANSION.md, Sec. 10).

PMLB files are fetched by plain HTTPS at a pinned commit, hashed, parsed with
pandas and encoded exactly like OpenML data; nothing downloaded is executed.
Each dataset is cached as ``data/cache/npz/{pmlb_id}.npz``, so the standard
``openml_loader.load`` reads it back. PMLB IDs are negative, never an OpenML ID.
"""

import gzip
import hashlib
import io
import os
import urllib.request
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from simaudit.data.openml_loader import Dataset, content_hash, encode
from simaudit.runner import build_pool_ext as ext

PMLB_COMMIT = "7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68"
SUMMARY_SHA256 = "d658820a844a555776273a2c1b6766eba983bb078e3bcc539141b1ac2cc33960"
RAW = f"https://raw.githubusercontent.com/EpistasisLab/pmlb/{PMLB_COMMIT}"
MEDIA = f"https://media.githubusercontent.com/media/EpistasisLab/pmlb/{PMLB_COMMIT}"
SUMMARY_URL = f"{RAW}/pmlb/all_summary_stats.tsv"
TIERS = ("2a", "3")

Fetch = Callable[[str], bytes]
DevLoader = Callable[[int], Dataset]


def sha256_bytes(raw: bytes) -> str:
    """SHA-256 hex digest of a byte string."""
    return hashlib.sha256(raw).hexdigest()


def pmlb_id(name: str) -> int:
    """Stable negative integer ID of a PMLB dataset (never an OpenML ID)."""
    return -(zlib.crc32(name.encode()) % 1_000_000_000) - 1


def _http_get(url: str) -> bytes:  # pragma: no cover - network call
    with urllib.request.urlopen(url, timeout=60) as response:
        return bytes(response.read())


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)


def fetch_summary(cache_dir: Path) -> pd.DataFrame:
    """Return PMLB's summary table at the pinned commit, checked against its hash."""
    path = cache_dir / "pmlb" / "all_summary_stats.tsv"
    raw = path.read_bytes() if path.exists() else _http_get(SUMMARY_URL)
    if sha256_bytes(raw) != SUMMARY_SHA256:
        raise ValueError(f"summary SHA-256 {sha256_bytes(raw)} != {SUMMARY_SHA256}")
    _atomic_bytes(path, raw)
    return pd.read_csv(io.BytesIO(raw), sep="\t")


def lfs_oid(pointer: bytes) -> str | None:
    """SHA-256 named by a Git LFS pointer file, or None if ``pointer`` is not one."""
    text = pointer[:512].decode("ascii", errors="replace")
    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    oid = next((ln for ln in text.splitlines() if ln.startswith("oid sha256:")), "")
    return oid.removeprefix("oid sha256:") or None


def fetcher(cache_dir: Path) -> Fetch:
    """Fetch ``<name>.tsv.gz`` at the pinned commit, cached under data/cache/pmlb.

    PMLB stores the files in Git LFS: the pointer at the commit names the
    SHA-256 of the content, which is fetched from GitHub's media host and must
    match it.
    """

    def fetch(name: str) -> bytes:
        path = cache_dir / "pmlb" / f"{name}.tsv.gz"
        if path.exists():
            return path.read_bytes()
        rel = f"datasets/{name}/{name}.tsv.gz"
        raw = _http_get(f"{RAW}/{rel}")
        oid = lfs_oid(raw)
        if oid is not None:
            raw = _http_get(f"{MEDIA}/{rel}")
            if sha256_bytes(raw) != oid:
                raise ValueError(f"{name}: content does not match LFS oid {oid}")
        _atomic_bytes(path, raw)
        return raw

    return fetch


def _between(value: Any, bounds: tuple[int, int]) -> bool:
    return bool(bounds[0] <= value <= bounds[1])


def metadata_reason(row: pd.Series, prot: ext.Protected) -> str | None:
    """First failing Amendment 1 rule (A1.2, A1.3 by name) or None."""
    name = str(row.dataset)
    rules: list[tuple[str, bool]] = [
        ("not classification", row.task != "classification"),
        ("deprecated", name.startswith("_deprecated")),
        ("C outside 2-10", not _between(row.n_classes, ext.C_RANGE)),
        ("n outside 100-5000", not _between(row.n_instances, ext.N_RANGE)),
        ("d outside 2-100", not _between(row.n_features, ext.D_RANGE)),
        ("DEV/HELD-OUT name", ext.name_key(name) in prot.names),
        ("DEV/HELD-OUT family", ext.family_key(name) in prot.families),
    ]
    return next((reason for reason, failed in rules if failed), None)


def tier_of(row: pd.Series) -> str:
    """Tier 2a if PMLB flags no categorical feature, else Tier 3."""
    return "2a" if int(row.n_categorical_features) == 0 else "3"


def parse_tsv(raw: bytes) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Features and encoded target of a PMLB ``.tsv.gz`` (target column ``target``)."""
    frame = pd.read_csv(io.BytesIO(gzip.decompress(raw)), sep="\t")
    return encode(frame.drop(columns="target"), frame["target"])


def load_pmlb(name: str, cache_dir: Path, fetch: Fetch) -> tuple[Dataset, str]:
    """Load (and cache as npz) one PMLB dataset; returns it and its file hash."""
    did = pmlb_id(name)
    path = cache_dir / "npz" / f"{did}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        data = Dataset(did, name, 0, z["X"], z["y"], str(z["sha"]))
        return data, str(z["file_sha"])
    raw = fetch(name)
    X, y = parse_tsv(raw)
    sha, file_sha = content_hash(X, y), sha256_bytes(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, X=X, y=y, name=name, version=0, sha=sha, file_sha=file_sha)
    tmp.replace(path)
    return Dataset(did, name, 0, X, y, sha), file_sha


def data_signature(data: Dataset) -> ext.Signature:
    """(n, C, majority size, minority size) of loaded data."""
    counts = np.bincount(data.y)
    return (len(data.y), int(counts.size), int(counts.max()), int(counts.min()))


def loaded_reason(data: Dataset, prot: ext.Protected) -> str | None:
    """Post-load checks (ranges, minority >= 11) and DEV/HELD-OUT content checks."""
    n, d = data.X.shape
    counts = np.bincount(data.y)
    if (
        counts.min() < ext.MIN_CLASS_SIZE
        or not _between(counts.size, ext.C_RANGE)
        or not _between(n, ext.N_RANGE)
        or not _between(d, ext.D_RANGE)
    ):
        return f"post-load check: n={n}, d={d}, C={counts.size}, min={counts.min()}"
    if data.sha256 in prot.hashes:
        return "DEV/HELD-OUT data hash"
    if data_signature(data) in prot.sigs:
        return "DEV/HELD-OUT near-duplicate"
    return None


def dedupe(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """One dataset per name key, family key, then signature; 2a first, then name."""
    ranked = rows.assign(
        nkey=rows["name"].map(ext.name_key), fkey=rows["name"].map(ext.family_key)
    ).sort_values(["tier", "name"])
    dropped: dict[str, str] = {}
    steps = (
        ("nkey", "older/newer version of the same name"),
        ("fkey", "same generator family"),
        ("sig", "near-duplicate (n, C, class sizes)"),
    )
    for key, reason in steps:
        dup = ranked.duplicated(key)
        dropped.update({str(n): reason for n in ranked[dup]["name"]})
        ranked = ranked[~dup]
    return ranked.drop(columns=["nkey", "fkey"]), dropped


def protected_without_heldout_sigs(
    pool_path: Path, dev_loader: DevLoader
) -> ext.Protected:
    """DEV/HELD-OUT names, families and hashes; signatures of DEV only (A1.3)."""
    pool = yaml.safe_load(pool_path.read_text())
    entries = pool["dev"] + pool["heldout"]
    return ext.Protected(
        dids={int(e["did"]) for e in entries},
        names={ext.name_key(e["name"]) for e in entries},
        families={ext.family_key(e["name"]) for e in entries},
        hashes={e["sha256"] for e in entries},
        sigs={data_signature(dev_loader(int(e["did"]))) for e in pool["dev"]},
    )


def _entry(data: Dataset, tier: str, file_sha: str) -> dict[str, Any]:
    n, C, big, small = data_signature(data)
    return dict(
        key=f"pmlb:{data.name}",
        did=data.did,
        name=data.name,
        version=0,
        sha256=data.sha256,
        file_sha256=file_sha,
        stratum=ext.stratum(C, big / small),
        n=n,
        d=int(data.X.shape[1]),
        C=C,
        majority=big,
        minority=small,
        ir=round(big / small, 4),
        tier=2 if tier == "2a" else 3,
        subtier=tier,
        source="pmlb",
        suites=[],
        priorsel_overlap=False,
    )


def _screen_and_load(
    summary: pd.DataFrame, prot: ext.Protected, cache_dir: Path, fetch: Fetch
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Metadata screen, download and post-load checks; rows kept and audit rows."""
    kept: list[dict[str, Any]] = []
    audit: list[dict[str, str]] = []
    for _, row in summary.sort_values("dataset").iterrows():
        name = str(row.dataset)
        reason = metadata_reason(row, prot)
        stage = "metadata"
        if reason is None:
            stage = "post-load"
            try:
                data, file_sha = load_pmlb(name, cache_dir, fetch)
                reason = loaded_reason(data, prot)
            except Exception as exc:  # noqa: BLE001 -- any failure excludes it
                reason = f"load: {exc}"[:200]
        if reason is None:
            entry = _entry(data, tier_of(row), file_sha)
            kept.append(dict(entry, sig=data_signature(data)))
        else:
            audit.append(dict(name=name, stage=stage, reason=reason))
    return kept, audit


def _counts(datasets: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Number of datasets per subtier and stratum."""
    return {
        t: {
            s: sum(d["subtier"] == t and d["stratum"] == s for d in datasets)
            for s in ext.STRATA
        }
        for t in TIERS
    }


def build_pool_pmlb(
    cache_dir: Path,
    pool_path: Path,
    out_path: Path,
    audit_path: Path,
    *,
    summary: pd.DataFrame,
    fetch: Fetch,
    dev_loader: DevLoader,
) -> dict[str, Any]:
    """Apply Amendment 1 to PMLB and freeze Tiers 2a and 3 (HELD-OUT sig pending)."""
    prot = protected_without_heldout_sigs(pool_path, dev_loader)
    kept, audit = _screen_and_load(summary, prot, cache_dir, fetch)
    frame = pd.DataFrame(kept, columns=[*_entry_keys(), "sig"])
    frame, dropped = dedupe(frame)
    audit += [dict(name=n, stage="dedup", reason=r) for n, r in dropped.items()]
    datasets = [
        {str(k): v for k, v in r.items() if k != "sig"}
        for r in frame.sort_values(["tier", "stratum", "name"]).to_dict("records")
    ]
    pool = dict(
        source="pmlb",
        commit=PMLB_COMMIT,
        summary_sha256=SUMMARY_SHA256,
        heldout_signature_check="pending",
        tier1_duplicate_check="pending",
        n_summary=int(len(summary)),
        counts=_counts(datasets),
        datasets=datasets,
        audit_file=audit_path.name,
    )
    pd.DataFrame(audit, columns=["name", "stage", "reason"]).to_csv(
        audit_path, index=False
    )
    out_path.write_text(yaml.safe_dump(_plain(pool), sort_keys=False))
    return cast(dict[str, Any], yaml.safe_load(out_path.read_text()))


def _entry_keys() -> list[str]:
    probe = Dataset(-1, "x", 0, np.zeros((2, 1)), np.array([0, 1]), "")
    return list(_entry(probe, "2a", ""))


def _plain(obj: Any) -> Any:
    """Convert numpy scalars to Python types for YAML."""
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
