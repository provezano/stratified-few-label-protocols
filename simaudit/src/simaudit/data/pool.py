"""Outcome-blind construction of the pilot dataset pool (PILOT_I2.md, Sec. 2)."""

import re

import numpy as np
import pandas as pd

N_TOTAL = 60
N_DEV = 20


def family(name: str) -> str:
    """Return the generator family: the name truncated at its first digit."""
    match = re.match(r"[^0-9]*", name)
    return match.group(0) if match else name


def eligible(meta: pd.DataFrame, excluded: set[int]) -> pd.DataFrame:
    """Apply the eligibility criteria, exclusions and per-name/family caps."""
    keep = (
        (meta.status == "active")
        & (meta.format.str.lower() != "sparse_arff")
        & meta.NumberOfInstances.between(100, 1000)
        & meta.NumberOfClasses.between(2, 10)
        & (meta.NumberOfSymbolicFeatures == 1)
        & (meta.NumberOfMissingValues == 0)
        & (meta.NumberOfFeatures - 1).between(2, 100)
        & (meta.MinorityClassSize >= 11)
        & ~meta.did.isin(excluded)
    )
    kept = meta[keep].sort_values(["name", "version"]).drop_duplicates("name")
    kept = kept.assign(family=kept.name.map(family)).sort_values("did")
    return kept.drop_duplicates("family").drop(columns="family")


def dedupe_by_hash(hashes: dict[int, str]) -> list[int]:
    """Keep the lowest dataset ID among datasets with identical content hashes."""
    first: dict[str, int] = {}
    for did in sorted(hashes):
        first.setdefault(hashes[did], did)
    return sorted(first.values())


def sample_pool(
    ids: list[int],
    seed: int,
    n_total: int = N_TOTAL,
    n_dev: int = N_DEV,
) -> tuple[list[int], list[int]]:
    """Sample the pool; the first ``n_dev`` sampled form DEV, the rest HELD-OUT."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(sorted(ids))[: min(n_total, len(ids))]
    sampled = [int(i) for i in order]
    return sampled[:n_dev], sampled[n_dev:]
