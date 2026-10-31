"""Study 8: build the stratified expansion pool (prereg/STUDY8_EXPANSION.md)."""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import openml
import pandas as pd
import yaml

from simaudit.data.openml_loader import Dataset, load

SEED = (20261020, 8)
N_RANGE = (100, 5000)
C_RANGE = (2, 10)
D_RANGE = (2, 100)
MIN_CLASS_SIZE = 11
QUOTA = 8
C_BINS = ("2", "3-4", "5-10")
IR_BINS = ("<2", "2-5", ">5")
STRATA = tuple(f"C{c}|IR{i}" for c in C_BINS for i in IR_BINS)
# Shortfall goes first to the strata where protocol bias should be largest.
REALLOC_ORDER = tuple(f"C{c}|IR{i}" for c in C_BINS[::-1] for i in IR_BINS[::-1])
SUITES = {"cc18": 99, "amlb_classification": 271, "tabarena_v0.1": 457}
# Amendment 1: |Tier 1| + |Tier 2| <= TOTAL_CAP; Tier 2b aims at STRATUM_TARGET.
TOTAL_CAP = 200
STRATUM_TARGET = 22

Loader = Callable[[int, Path], Dataset]
Accept = Callable[[int, str], str | None]
Signature = tuple[int, int, int, int]


@dataclass(frozen=True)
class Protected:
    """Identifiers of DEV and HELD-OUT datasets that the new pool must avoid."""

    dids: set[int]
    names: set[str]
    families: set[str]
    hashes: set[str]
    sigs: set[Signature]


def c_bin(c: int) -> str:
    """Return the class-count bin of ``c`` (2, 3-4 or 5-10)."""
    if not C_RANGE[0] <= c <= C_RANGE[1]:
        raise ValueError(f"C={c} outside {C_RANGE}")
    return "2" if c == 2 else "3-4" if c <= 4 else "5-10"


def ir_bin(ir: float) -> str:
    """Return the imbalance-ratio bin: < 2, 2-5 (inclusive) or > 5."""
    return "<2" if ir < 2 else "2-5" if ir <= 5 else ">5"


def stratum(c: int, ir: float) -> str:
    """Return the stratum label, e.g. ``C3-4|IR>5``."""
    return f"C{c_bin(c)}|IR{ir_bin(ir)}"


def name_key(name: str) -> str:
    """Lower-case the name and drop every non-alphanumeric character."""
    return re.sub(r"[^0-9a-z]", "", name.lower())


def family_key(name: str) -> str:
    """Return the generator family: the name key truncated at its first digit."""
    key = name_key(name)
    head = re.match(r"[a-z]*", key)
    return head.group(0) if head and head.group(0) else key


def signature(row: pd.Series) -> Signature:
    """Near-duplicate signature: (n, C, majority size, minority size)."""
    return (
        int(row.NumberOfInstances),
        int(row.NumberOfClasses),
        int(row.MajorityClassSize),
        int(row.MinorityClassSize),
    )


def _between(value: Any, bounds: tuple[int, int]) -> bool:
    return bool(bounds[0] <= value <= bounds[1])


Rule = tuple[str, Callable[[pd.Series], bool]]


def _qualities_missing(row: pd.Series) -> bool:
    cols = ["NumberOfClasses", "MajorityClassSize", "MinorityClassSize"]
    return bool(row[cols].isna().any())


def _rules(prot: Protected) -> list[Rule]:
    """(reason, failed) checks in the preregistered order (STUDY8 Sec. 3)."""
    return [
        ("not active", lambda r: r.status != "active"),
        ("sparse format", lambda r: str(r.format).lower() == "sparse_arff"),
        ("not classification", _qualities_missing),
        ("C outside 2-10", lambda r: not _between(r.NumberOfClasses, C_RANGE)),
        ("n outside 100-5000", lambda r: not _between(r.NumberOfInstances, N_RANGE)),
        ("d outside 2-100", lambda r: not _between(r.NumberOfFeatures - 1, D_RANGE)),
        ("non-numeric features", lambda r: r.NumberOfSymbolicFeatures != 1),
        ("missing values", lambda r: r.NumberOfMissingValues != 0),
        ("minority class < 11", lambda r: r.MinorityClassSize < MIN_CLASS_SIZE),
        ("DEV/HELD-OUT id", lambda r: int(r.did) in prot.dids),
        ("DEV/HELD-OUT name", lambda r: name_key(r["name"]) in prot.names),
        ("DEV/HELD-OUT family", lambda r: family_key(r["name"]) in prot.families),
        ("DEV/HELD-OUT near-duplicate", lambda r: signature(r) in prot.sigs),
    ]


def metadata_reason(row: pd.Series, prot: Protected) -> str | None:
    """Return the first eligibility rule ``row`` fails, or None if it passes."""
    for reason, failed in _rules(prot):
        if failed(row):
            return reason
    return None


def _dedupe(
    frame: pd.DataFrame, key: str, order: list[str], reason: str
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Keep the first row per ``key`` after sorting by ``order``."""
    ranked = frame.sort_values(order)
    dup = ranked.duplicated(key)
    return ranked[~dup], {int(d): reason for d in ranked[dup].did}


def screen(
    meta: pd.DataFrame, prot: Protected, suite_ids: set[int]
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Apply the metadata rules and the within-pool de-duplication.

    Returns the eligible frame (with ``stratum``, ``tier``, ``n``, ``d``, ``C``
    and ``ir``) and the reason for every excluded dataset ID.
    """
    reasons = {int(r.did): metadata_reason(r, prot) for _, r in meta.iterrows()}
    excluded = {d: r for d, r in reasons.items() if r is not None}
    ok = meta[~meta.did.astype(int).isin(excluded)].copy()
    ok["tier"] = np.where(ok.did.astype(int).isin(suite_ids), 1, 2)
    ok["nkey"] = ok["name"].map(name_key)
    ok["fkey"] = ok["name"].map(family_key)
    ok["sig"] = [signature(r) for _, r in ok.iterrows()]
    steps = [
        (
            "nkey",
            ["nkey", "tier", "version", "did"],
            "older/newer version of the same name",
        ),
        ("fkey", ["fkey", "tier", "version", "did"], "same generator family"),
        ("sig", ["tier", "version", "did"], "near-duplicate (n, C, class sizes)"),
    ]
    for key, order, reason in steps:
        ok, dropped = _dedupe(ok, key, order, reason)
        excluded.update(dropped)
    ok = ok.assign(
        n=ok.NumberOfInstances.astype(int),
        d=(ok.NumberOfFeatures - 1).astype(int),
        C=ok.NumberOfClasses.astype(int),
        ir=ok.MajorityClassSize / ok.MinorityClassSize,
    )
    ok["stratum"] = [stratum(c, ir) for c, ir in zip(ok.C, ok.ir)]
    keep = ["did", "name", "version", "n", "d", "C", "ir", "stratum", "tier"]
    return ok.sort_values("did")[keep].reset_index(drop=True), excluded


def build_queues(
    eligible: pd.DataFrame, seed: tuple[int, ...] = SEED
) -> dict[str, list[int]]:
    """Per stratum: suite members in seeded random order, then all others."""
    rng = np.random.default_rng(list(seed))
    queues: dict[str, list[int]] = {}
    for s in STRATA:
        queues[s] = []
        for tier in (1, 2):
            ids = sorted(
                eligible.did[(eligible.stratum == s) & (eligible.tier == tier)]
            )
            queues[s] += [int(i) for i in rng.permutation(np.array(ids, dtype=int))]
    return queues


def _take(
    queue: list[int], s: str, accept: Accept, rejected: dict[int, str]
) -> int | None:
    """Pop from ``queue`` until a dataset passes ``accept``; None if exhausted."""
    while queue:
        did = queue.pop(0)
        reason = accept(did, s)
        if reason is None:
            return did
        rejected[did] = reason
    return None


def draw(
    queues: Mapping[str, list[int]], quota: int, accept: Accept
) -> tuple[dict[str, list[int]], dict[int, str]]:
    """Fill each stratum's quota, then reallocate any shortfall (Sec. 5).

    The shortfall is filled one dataset at a time, cycling over the strata in
    ``REALLOC_ORDER`` that still have candidates, until the total reaches
    ``quota * 9`` or every queue is empty. ``queues`` is consumed in place.
    """
    return draw_quotas(queues, {s: quota for s in STRATA}, quota * len(STRATA), accept)


def draw_quotas(
    queues: Mapping[str, list[int]],
    quotas: Mapping[str, int],
    target: int,
    accept: Accept,
) -> tuple[dict[str, list[int]], dict[int, str]]:
    """Fill per-stratum ``quotas``, then reallocate up to ``target`` in total."""
    picked: dict[str, list[int]] = {s: [] for s in STRATA}
    rejected: dict[int, str] = {}
    for s in STRATA:
        while len(picked[s]) < quotas[s] and sum(map(len, picked.values())) < target:
            did = _take(queues[s], s, accept, rejected)
            if did is None:
                break
            picked[s].append(did)
    _reallocate(queues, picked, target, accept, rejected)
    return picked, rejected


def _reallocate(
    queues: Mapping[str, list[int]],
    picked: dict[str, list[int]],
    target: int,
    accept: Accept,
    rejected: dict[int, str],
) -> None:
    """Round-robin over ``REALLOC_ORDER`` until ``target`` or no candidates left."""
    total = sum(map(len, picked.values()))
    while total < target and any(queues.values()):
        for s in REALLOC_ORDER:
            if total >= target:
                break
            did = _take(queues[s], s, accept, rejected)
            if did is not None:
                picked[s].append(did)
                total += 1


def loaded_reason(data: Dataset, expected: str, seen: set[str]) -> str | None:
    """Post-load checks on the downloaded data; None if the dataset is valid."""
    counts = np.bincount(data.y)
    n, d = data.X.shape
    if (
        counts.min() < MIN_CLASS_SIZE
        or not _between(counts.size, C_RANGE)
        or not _between(n, N_RANGE)
        or not _between(d, D_RANGE)
    ):
        return f"post-load check: n={n}, d={d}, C={counts.size}, min={counts.min()}"
    if data.sha256 in seen:
        return "duplicate data hash"
    actual = stratum(int(counts.size), float(counts.max() / counts.min()))
    if actual != expected:
        return f"stratum mismatch: metadata {expected}, data {actual}"
    return None


def protected_from_pool(pool_path: Path, meta: pd.DataFrame) -> Protected:
    """Collect DEV and HELD-OUT identifiers from the non-sealed pool file."""
    pool = yaml.safe_load(pool_path.read_text())
    entries = pool["dev"] + pool["heldout"]
    dids = {int(e["did"]) for e in entries}
    rows = meta[meta.did.astype(int).isin(dids)]
    return Protected(
        dids=dids,
        names={name_key(e["name"]) for e in entries},
        families={family_key(e["name"]) for e in entries},
        hashes={e["sha256"] for e in entries},
        sigs={signature(r) for _, r in rows.iterrows()},
    )


def fetch_suite_ids(suites: Mapping[str, int | str] = SUITES) -> dict[str, list[int]]:
    """Dataset IDs of each OpenML benchmark suite."""
    return {
        k: sorted(int(d) for d in openml.study.get_suite(v).data or [])
        for k, v in suites.items()
    }


def fetch_metadata() -> pd.DataFrame:  # pragma: no cover - network call
    """All active OpenML datasets with their qualities."""
    meta = openml.datasets.list_datasets(output_format="dataframe", status="active")
    return meta


@dataclass
class _Acceptor:
    """Downloads a candidate with the normal loader and validates it."""

    cache_dir: Path
    loader: Loader
    seen: set[str]
    loaded: dict[int, Dataset]

    def __call__(self, did: int, s: str) -> str | None:
        try:
            data = self.loader(did, self.cache_dir)
        except Exception as exc:  # noqa: BLE001 -- any failure excludes it
            return f"load: {exc}"[:200]
        reason = loaded_reason(data, s, self.seen)
        if reason is None:
            self.seen.add(data.sha256)
            self.loaded[did] = data
        return reason


def _entry(
    data: Dataset, s: str, subtier: str, suites: list[str], priorsel: bool
) -> dict[str, Any]:
    counts = np.bincount(data.y)
    return dict(
        key=f"openml:{data.did}",
        did=data.did,
        name=data.name,
        version=data.version,
        sha256=data.sha256,
        stratum=s,
        n=int(data.X.shape[0]),
        d=int(data.X.shape[1]),
        C=int(counts.size),
        majority=int(counts.max()),
        minority=int(counts.min()),
        ir=round(float(counts.max() / counts.min()), 4),
        tier=1 if subtier == "1" else 2,
        subtier=subtier,
        source="suite" if suites else "openml",
        suites=suites,
        priorsel_overlap=priorsel,
    )


def _audit(
    path: Path, meta: pd.DataFrame, stages: list[tuple[str, Mapping[int, str]]]
) -> None:
    names = dict(zip(meta.did.astype(int), meta["name"]))
    rows = [
        dict(did=d, name=names.get(d, ""), stage=stage, reason=r)
        for stage, reasons in stages
        for d, r in sorted(reasons.items())
    ]
    pd.DataFrame(rows, columns=["did", "name", "stage", "reason"]).to_csv(
        path, index=False
    )


def _strata_summary(
    eligible: pd.DataFrame,
    picked: Mapping[str, list[int]],
    picked_2b: Mapping[str, list[int]],
    pmlb: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Eligible, suite-member and sampled counts per stratum and tier."""
    return {
        s: dict(
            eligible=int((eligible.stratum == s).sum()),
            suite_members=int(((eligible.stratum == s) & (eligible.tier == 1)).sum()),
            sampled_tier1=len(picked[s]),
            sampled_tier2a=sum(
                e["subtier"] == "2a" and e["stratum"] == s for e in pmlb
            ),
            sampled_tier2b=len(picked_2b[s]),
            sampled_tier3=sum(e["subtier"] == "3" and e["stratum"] == s for e in pmlb),
        )
        for s in STRATA
    }


def tier2b_quotas(
    tier1: Mapping[str, int], tier2a: Mapping[str, int], target: int = STRATUM_TARGET
) -> dict[str, int]:
    """Tier 2b quota per stratum: what is left of ``target`` (Amendment 1, A1.4)."""
    return {s: max(0, target - tier1.get(s, 0) - tier2a.get(s, 0)) for s in STRATA}


def _pmlb_sig(entry: Mapping[str, Any]) -> Signature:
    return (
        int(entry["n"]),
        int(entry["C"]),
        int(entry["majority"]),
        int(entry["minority"]),
    )


def pmlb_conflicts(
    pmlb: list[dict[str, Any]],
    tier1: list[dict[str, Any]],
    heldout_sigs: set[Signature],
) -> dict[str, str]:
    """PMLB entries that duplicate a Tier 1 dataset or HELD-OUT (Tier 1 wins)."""
    names = {name_key(t["name"]) for t in tier1}
    families = {family_key(t["name"]) for t in tier1}
    hashes = {t["sha256"] for t in tier1}
    sigs = {tuple(t["sig"]) for t in tier1}
    out: dict[str, str] = {}
    for e in pmlb:
        hit = (
            name_key(e["name"]) in names
            or family_key(e["name"]) in families
            or e["sha256"] in hashes
            or _pmlb_sig(e) in sigs
        )
        if hit:
            out[e["name"]] = "duplicate of Tier 1"
        elif _pmlb_sig(e) in heldout_sigs:
            out[e["name"]] = "DEV/HELD-OUT near-duplicate"
    return out


def drop_pmlb_duplicates(
    queues: Mapping[str, list[int]], meta: pd.DataFrame, pmlb: list[dict[str, Any]]
) -> dict[int, str]:
    """Remove OpenML candidates that duplicate a PMLB dataset (by metadata)."""
    names = {name_key(e["name"]) for e in pmlb}
    families = {family_key(e["name"]) for e in pmlb}
    sigs = {_pmlb_sig(e) for e in pmlb}
    rows = meta.set_index(meta.did.astype(int))
    dropped: dict[int, str] = {}
    for queue in queues.values():
        for did in list(queue):
            row = cast(pd.Series, rows.loc[did])
            if (
                name_key(str(row["name"])) in names
                or family_key(str(row["name"])) in families
                or signature(row) in sigs
            ):
                queue.remove(did)
                dropped[did] = "duplicate of PMLB"
    return dropped


def _reason_counts(reasons: Mapping[int, str]) -> dict[str, int]:
    """Number of excluded datasets per reason, most frequent first."""
    counts: dict[str, int] = {}
    for r in reasons.values():
        counts[r] = counts.get(r, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _read_pmlb(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return list(yaml.safe_load(path.read_text())["datasets"])


def _openml_entries(
    picked: Mapping[str, list[int]],
    subtier: str,
    loaded: Mapping[int, Dataset],
    suites: Mapping[str, list[int]],
    priorsel: set[int],
) -> list[dict[str, Any]]:
    return [
        _entry(
            loaded[d],
            s,
            subtier,
            [k for k, ids in suites.items() if d in ids],
            d in priorsel,
        )
        for s in STRATA
        for d in picked[s]
    ]


def _count(entries: list[dict[str, Any]], subtier: str) -> dict[str, int]:
    return {
        s: sum(e["subtier"] == subtier and e["stratum"] == s for e in entries)
        for s in STRATA
    }


def build_pool_ext(
    cache_dir: Path,
    pool_path: Path,
    priorsel_path: Path,
    out_path: Path,
    audit_path: Path,
    *,
    meta: pd.DataFrame,
    suites: Mapping[str, list[int]],
    loader: Loader = load,
    quota: int = QUOTA,
    pmlb_path: Path | None = None,
    total_cap: int = TOTAL_CAP,
    stratum_target: int = STRATUM_TARGET,
) -> dict[str, Any]:
    """Apply Study 8 (Tier 1) and Amendment 1 (Tiers 2a, 2b, 3); freeze the pool."""
    prot = protected_from_pool(pool_path, meta)
    suite_ids = {d for ids in suites.values() for d in ids}
    eligible, meta_excluded = screen(meta, prot, suite_ids)
    queues = build_queues(eligible)
    acceptor = _Acceptor(cache_dir, loader, set(prot.hashes), {})
    picked, rejected = draw(queues, quota, acceptor)
    priorsel = {int(i) for i in priorsel_path.read_text().split()}
    tier1 = _openml_entries(picked, "1", acceptor.loaded, suites, priorsel)
    t1_sigs = [
        dict(e, sig=(e["n"], e["C"], e["majority"], e["minority"])) for e in tier1
    ]
    pmlb_all = _read_pmlb(pmlb_path)
    removed = pmlb_conflicts(pmlb_all, t1_sigs, prot.sigs)
    pmlb = [e for e in pmlb_all if e["name"] not in removed]
    dup_pmlb = drop_pmlb_duplicates(queues, meta, pmlb)
    acceptor.seen.update(e["sha256"] for e in pmlb)
    tier2a = [e for e in pmlb if e["subtier"] == "2a"]
    quotas = tier2b_quotas(_count(tier1, "1"), _count(tier2a, "2a"), stratum_target)
    cap = max(0, total_cap - len(tier1) - len(tier2a))
    picked_2b, rejected_2b = draw_quotas(queues, quotas, cap, acceptor)
    tier2b = _openml_entries(picked_2b, "2b", acceptor.loaded, suites, priorsel)
    undrawn = {d: "eligible, not drawn (cap met)" for q in queues.values() for d in q}
    pool = dict(
        seed=list(SEED),
        quota_per_stratum=quota,
        total_cap=total_cap,
        stratum_target_tier2=stratum_target,
        tier2b_quotas=quotas,
        suites={k: len(v) for k, v in suites.items()},
        n_metadata_listed=int(len(meta)),
        n_eligible=int(len(eligible)),
        heldout_signature_check="passed",
        strata=_strata_summary(eligible, picked, picked_2b, pmlb),
        datasets=tier1 + tier2a + tier2b + [e for e in pmlb if e["subtier"] == "3"],
        pmlb_removed=removed,
        post_load_rejected={**rejected, **rejected_2b},
        metadata_exclusions=_reason_counts(meta_excluded),
        audit_file=audit_path.name,
    )
    _audit(
        audit_path,
        meta,
        [
            ("metadata", meta_excluded),
            ("post-load", {**rejected, **rejected_2b}),
            ("duplicate of PMLB", dup_pmlb),
            ("tier-2b not drawn", undrawn),
        ],
    )
    out_path.write_text(yaml.safe_dump(pool, sort_keys=False))
    return cast(dict[str, Any], yaml.safe_load(out_path.read_text()))
