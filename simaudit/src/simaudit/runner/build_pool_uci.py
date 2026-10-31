"""Study 8 Amendment 2: Tier 4 from the UCI ML Repository (prereg/STUDY8_EXPANSION.md).

Candidates come from a frozen snapshot of UCI's official API metadata (the
datasets the API serves as ``data.csv``). Data files are fetched by plain HTTPS
from ``archive.ics.uci.edu/static/public``, hashed, parsed with pandas and
encoded exactly like OpenML data; nothing downloaded is executed. Each dataset
is cached as ``data/cache/npz/{uci_did}.npz`` so the standard
``openml_loader.load`` reads it back.
"""

import gzip
import hashlib
import io
import json
import re
import time
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from simaudit.data.openml_loader import Dataset, content_hash, encode
from simaudit.runner import build_pool_ext as ext
from simaudit.runner import build_pool_pmlb as pm

API = "https://archive.ics.uci.edu/api"
LIST_URL = f"{API}/datasets/list?filter=python"
META_URL = f"{API}/dataset?id={{uid}}"
PAGE_URL = "https://archive.ics.uci.edu/dataset/{uid}"
DATA_PREFIX = "https://archive.ics.uci.edu/static/public/"
NUMERIC_TYPES = frozenset({"Continuous", "Integer", "Binary"})
# Words too generic to identify a dataset in the name-clash rule (A2.3).
GENERIC = frozenset({"data", "dataset", "database", "test", "clean", "numeric"})
MIN_KEY = 4
UCI_OFFSET = 1_000_000_000
# Frozen snapshot of the API metadata (data/uci_api_snapshot.json.gz, A2.1).
SNAPSHOT_SHA256 = "1f523469b00b0eaec11695f8b437131fcb638eb0c955dd8f194ec0bd5683de46"

Fetch = Callable[[str], bytes]
GetJson = Callable[[str], dict[str, Any]]
GetText = Callable[[str], str]
DidLoader = Callable[[int], Dataset]
Meta = Mapping[str, Any]
data_signature = pm.data_signature


def uci_did(uid: int) -> int:
    """Negative dataset ID below every PMLB ID, so never an OpenML or PMLB ID."""
    return -(UCI_OFFSET + int(uid))


def licence_of(html: str) -> str | None:
    """Creative Commons licence linked from a UCI dataset page, or None."""
    m = re.search(r"creativecommons\.org/licenses/([a-z-]+)/(\d\.\d)", html)
    if m:
        return f"CC {m.group(1).upper()} {m.group(2)}"
    m = re.search(r"creativecommons\.org/publicdomain/zero/(\d\.\d)", html)
    return f"CC0 {m.group(1)}" if m else None


def name_tokens(name: str) -> list[str]:
    """Lower-case alphanumeric runs of a dataset name."""
    return re.findall(r"[a-z0-9]+", name.lower())


def head_token(name: str) -> str | None:
    """First token of at least MIN_KEY characters that is not a generic word."""
    return next(
        (t for t in name_tokens(name) if len(t) >= MIN_KEY and t not in GENERIC),
        None,
    )


def _contains(a: str, b: str) -> bool:
    """True if one key (of at least MIN_KEY characters) lies inside the other."""
    return (len(a) >= MIN_KEY and a in b) or (len(b) >= MIN_KEY and b in a)


def _clashes(candidate: str, other: str) -> bool:
    cand_keys = {ext.name_key(candidate), ext.family_key(candidate)}
    other_keys = {ext.name_key(other), ext.family_key(other)}
    tokens = set(name_tokens(candidate))
    return (
        bool(cand_keys & other_keys)
        or any(_contains(a, b) for a in cand_keys for b in other_keys)
        or head_token(other) in tokens
        or ext.name_key(other) in tokens
    )


def name_clash(candidate: str, protected: Iterable[str]) -> str | None:
    """First protected name that may denote the same data as ``candidate`` (A2.3)."""
    return next((p for p in protected if _clashes(candidate, p)), None)


def feature_columns(meta: Meta) -> list[str]:
    """Names of the variables whose role is Feature, in metadata order."""
    return [str(v["name"]) for v in meta["variables"] if v["role"] == "Feature"]


def target_column(meta: Meta) -> str:
    """The single target column named by the metadata."""
    return str(meta["target_col"][0])


class Screen(NamedTuple):
    """Names the metadata screen protects (DEV/HELD-OUT fuzzily, PMLB exactly)."""

    protected_names: list[str]
    pmlb_names: list[str]


def _feature_types(meta: Meta) -> set[str]:
    return {str(v["type"]) for v in meta["variables"] if v["role"] == "Feature"}


def _official(meta: Meta) -> bool:
    return str(meta.get("data_url") or "").startswith(DATA_PREFIX)


def _pmlb_key_hit(name: str, pmlb: list[str], key: Callable[[str], str]) -> bool:
    return key(name) in {key(p) for p in pmlb}


def metadata_reason(meta: Meta, screen: Screen) -> str | None:
    """First failing Amendment 2 metadata rule (A2.2, A2.3), or None."""
    name = str(meta["name"])
    clash = name_clash(name, screen.protected_names)
    rules: list[tuple[str, bool]] = [
        ("no official data file", not _official(meta)),
        ("not classification", "Classification" not in (meta.get("tasks") or [])),
        ("licence not Creative Commons", not meta.get("licence")),
        ("not one target", len(meta.get("target_col") or []) != 1),
        (
            "n outside 100-5000",
            not ext.N_RANGE[0] <= int(meta["num_instances"] or 0) <= ext.N_RANGE[1],
        ),
        (
            "d outside 2-100",
            not ext.D_RANGE[0] <= len(feature_columns(meta)) <= ext.D_RANGE[1],
        ),
        ("non-numeric features", not _feature_types(meta) <= NUMERIC_TYPES),
        ("missing values", meta.get("has_missing_values") != "no"),
        (f"DEV/HELD-OUT name ({clash})", clash is not None),
        ("PMLB name", _pmlb_key_hit(name, screen.pmlb_names, ext.name_key)),
        ("PMLB family", _pmlb_key_hit(name, screen.pmlb_names, ext.family_key)),
    ]
    return next((reason for reason, failed in rules if failed), None)


def parse_csv(raw: bytes, meta: Meta) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Feature-role columns and the encoded target of a UCI ``data.csv``."""
    frame = pd.read_csv(io.BytesIO(raw))
    feats, target = feature_columns(meta), target_column(meta)
    absent = [c for c in [*feats, target] if c not in frame.columns]
    if absent:
        raise ValueError(f"columns not in file: {absent[:5]}")
    return encode(frame[feats], frame[target])


def _fetch_cached(meta: Meta, cache_dir: Path, fetch: Fetch) -> bytes:
    path = cache_dir / "uci" / f"{int(meta['uci_id'])}.csv"
    if path.exists():
        return path.read_bytes()
    raw = fetch(str(meta["data_url"]))
    pm._atomic_bytes(path, raw)
    return raw


def load_uci(meta: Meta, cache_dir: Path, fetch: Fetch) -> tuple[Dataset, str]:
    """Load (and cache as npz) one UCI dataset; returns it and its file hash."""
    did, name = uci_did(int(meta["uci_id"])), str(meta["name"])
    path = cache_dir / "npz" / f"{did}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        data = Dataset(did, name, 0, z["X"], z["y"], str(z["sha"]))
        return data, str(z["file_sha"])
    raw = _fetch_cached(meta, cache_dir, fetch)
    X, y = parse_csv(raw, meta)
    sha, file_sha = content_hash(X, y), pm.sha256_bytes(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, X=X, y=y, name=name, version=0, sha=sha, file_sha=file_sha)
    tmp.replace(path)
    return Dataset(did, name, 0, X, y, sha), file_sha


def feature_hash(X: NDArray[np.float64]) -> str:
    """SHA-256 of the row-sorted X alone: same rows under another target."""
    rows = X[np.lexsort(X.T[::-1])]
    return hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()


class ContentRefs(NamedTuple):
    """Content identifiers of the frozen PMLB tiers (A2.3)."""

    hashes: set[str]
    feature_hashes: set[str]
    sigs: set[ext.Signature]


def pmlb_refs(pmlb_pool: Path, loader: DidLoader) -> ContentRefs:
    """Hashes, feature hashes and signatures of every PMLB pool dataset."""
    entries = yaml.safe_load(pmlb_pool.read_text())["datasets"]
    return ContentRefs(
        hashes={str(e["sha256"]) for e in entries},
        feature_hashes={feature_hash(loader(int(e["did"])).X) for e in entries},
        sigs={(e["n"], e["C"], e["majority"], e["minority"]) for e in entries},
    )


def content_reason(data: Dataset, refs: ContentRefs) -> str | None:
    """Exclusion of a loaded UCI dataset that repeats PMLB content, or None."""
    if data.sha256 in refs.hashes:
        return "PMLB data hash"
    if feature_hash(data.X) in refs.feature_hashes:
        return "PMLB feature hash"
    if data_signature(data) in refs.sigs:
        return "PMLB near-duplicate (n, C, class sizes)"
    return None


def dedupe(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    """One dataset per name, family, signature and feature hash; lowest UCI ID wins."""
    ranked = rows.assign(
        nkey=rows["name"].map(ext.name_key), fkey=rows["name"].map(ext.family_key)
    ).sort_values("uci_id")
    dropped: dict[int, str] = {}
    steps = (
        ("nkey", "older/newer version of the same name"),
        ("fkey", "same generator family"),
        ("sig", "near-duplicate (n, C, class sizes)"),
        ("fx", "same feature matrix"),
    )
    for key, reason in steps:
        dup = ranked.duplicated(key)
        dropped.update({int(u): reason for u in ranked[dup]["uci_id"]})
        ranked = ranked[~dup]
    return ranked.drop(columns=["nkey", "fkey"]), dropped


def _http_get(url: str) -> bytes:  # pragma: no cover - network call
    with urllib.request.urlopen(url, timeout=60) as response:
        return bytes(response.read())


def http_json(url: str) -> dict[str, Any]:  # pragma: no cover - network call
    """GET a UCI API URL and parse the JSON body (data only, never executed)."""
    return cast(dict[str, Any], json.loads(_http_get(url)))


def http_text(url: str) -> str:  # pragma: no cover - network call
    """GET a page as text (only searched for a licence link)."""
    return _http_get(url).decode("utf-8", errors="replace")


def build_snapshot(
    get_json: GetJson, get_text: GetText, sleep: Callable[[float], None] = time.sleep
) -> dict[str, Any]:
    """Metadata of every dataset the API serves as data.csv, with its licence."""
    listing = get_json(LIST_URL)["data"]
    datasets: list[dict[str, Any]] = []
    missing: list[int] = []
    for uid in sorted(int(d["id"]) for d in listing):
        body = get_json(META_URL.format(uid=uid))
        sleep(0.3)
        if body.get("status") != 200 or "data" not in body:
            missing.append(uid)
            continue
        meta = dict(body["data"])
        meta["licence"] = licence_of(get_text(PAGE_URL.format(uid=uid)))
        datasets.append(meta)
    return dict(list_url=LIST_URL, datasets=datasets, missing=missing)


def save_snapshot(snapshot: Mapping[str, Any], path: Path) -> str:
    """Write the snapshot as canonical gzipped JSON; return the file's SHA-256."""
    text = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(text.encode())
    pm._atomic_bytes(path, buf.getvalue())
    return pm.sha256_bytes(buf.getvalue())


def load_snapshot(path: Path, sha: str) -> dict[str, Any]:
    """Read the frozen snapshot, refusing any file whose hash is not ``sha``."""
    raw = path.read_bytes()
    if pm.sha256_bytes(raw) != sha:
        raise ValueError(f"snapshot SHA-256 {pm.sha256_bytes(raw)} != {sha}")
    return cast(dict[str, Any], json.loads(gzip.decompress(raw)))


def fetcher() -> Fetch:  # pragma: no cover - network call
    """Fetch an official UCI data file; any other host is refused."""

    def fetch(url: str) -> bytes:
        if not url.startswith(DATA_PREFIX):
            raise ValueError(f"not an official UCI data URL: {url}")
        return _http_get(url)

    return fetch


def _entry(data: Dataset, meta: Meta, file_sha: str) -> dict[str, Any]:
    n, C, big, small = data_signature(data)
    return dict(
        key=f"uci:{int(meta['uci_id'])}",
        did=data.did,
        uci_id=int(meta["uci_id"]),
        name=data.name,
        version=0,
        doi=meta.get("dataset_doi"),
        last_updated=meta.get("last_updated"),
        licence=meta.get("licence"),
        sha256=data.sha256,
        file_sha256=file_sha,
        stratum=ext.stratum(C, big / small),
        n=n,
        d=int(data.X.shape[1]),
        C=C,
        majority=big,
        minority=small,
        ir=round(big / small, 4),
        tier=4,
        subtier="4",
        source="uci",
        suites=[],
        priorsel_overlap=False,
    )


class _Ctx(NamedTuple):
    screen: Screen
    prot: ext.Protected
    refs: ContentRefs
    cache_dir: Path
    fetch: Fetch


def _audit(meta: Meta, stage: str, reason: str) -> dict[str, Any]:
    return dict(
        uci_id=int(meta["uci_id"]), name=meta["name"], stage=stage, reason=reason
    )


def _screen_one(
    meta: Meta, ctx: _Ctx
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Metadata rules, download and post-load checks for one candidate."""
    reason = metadata_reason(meta, ctx.screen)
    if reason is not None:
        return None, _audit(meta, "metadata", reason)
    try:
        data, file_sha = load_uci(meta, ctx.cache_dir, ctx.fetch)
        reason = pm.loaded_reason(data, ctx.prot) or content_reason(data, ctx.refs)
    except Exception as exc:  # noqa: BLE001 -- any failure excludes it
        reason = f"load: {exc}"[:200]
    if reason is not None:
        return None, _audit(meta, "post-load", reason)
    row = _entry(data, meta, file_sha)
    return dict(row, sig=data_signature(data), fx=feature_hash(data.X)), None


def _counts(datasets: list[dict[str, Any]]) -> dict[str, int]:
    return {s: sum(d["stratum"] == s for d in datasets) for s in ext.STRATA}


def build_pool_uci(
    cache_dir: Path,
    pool_path: Path,
    pmlb_pool: Path,
    out_path: Path,
    audit_path: Path,
    *,
    snapshot: Mapping[str, Any],
    snapshot_sha: str,
    fetch: Fetch,
    loader: DidLoader,
) -> dict[str, Any]:
    """Apply Amendment 2 to the UCI snapshot; freeze Tier 4 (HELD-OUT sig pending)."""
    prot = pm.protected_without_heldout_sigs(pool_path, loader)
    i2 = yaml.safe_load(pool_path.read_text())
    pmlb_names = [
        str(e["name"]) for e in yaml.safe_load(pmlb_pool.read_text())["datasets"]
    ]
    screen = Screen([str(e["name"]) for e in i2["dev"] + i2["heldout"]], pmlb_names)
    ctx = _Ctx(screen, prot, pmlb_refs(pmlb_pool, loader), cache_dir, fetch)
    kept: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for meta in sorted(snapshot["datasets"], key=lambda m: int(m["uci_id"])):
        row, rejected = _screen_one(meta, ctx)
        kept += [row] if row else []
        audit += [rejected] if rejected else []
    frame = pd.DataFrame(kept, columns=[*_entry_keys(), "sig", "fx"])
    frame, dropped = dedupe(frame)
    names = {int(r["uci_id"]): r["name"] for r in kept}
    audit += [
        dict(uci_id=u, name=names[u], stage="dedup", reason=r)
        for u, r in dropped.items()
    ]
    datasets = [
        {str(k): v for k, v in r.items() if k not in ("sig", "fx")}
        for r in frame.sort_values(["stratum", "uci_id"]).to_dict("records")
    ]
    pool = dict(
        source="uci",
        api=API,
        snapshot_sha256=snapshot_sha,
        n_snapshot=len(snapshot["datasets"]),
        heldout_signature_check="pending",
        tier1_duplicate_check="pending",
        counts=_counts(datasets),
        datasets=datasets,
        audit_file=audit_path.name,
    )
    pd.DataFrame(audit, columns=["uci_id", "name", "stage", "reason"]).sort_values(
        "uci_id"
    ).to_csv(audit_path, index=False)
    out_path.write_text(yaml.safe_dump(pm._plain(pool), sort_keys=False))
    return cast(dict[str, Any], yaml.safe_load(out_path.read_text()))


def _entry_keys() -> list[str]:
    probe = Dataset(-1, "x", 0, np.zeros((2, 1)), np.array([0, 1]), "")
    return list(_entry(probe, dict(uci_id=1), ""))
