"""Build and freeze the pilot pool (PILOT_I2.md, Sec. 2)."""

from pathlib import Path
from typing import Any

import numpy as np
import openml
import yaml

from simaudit.data.openml_loader import load
from simaudit.data.pool import dedupe_by_hash, eligible, sample_pool

MIN_CLASS_SIZE = 11


def build_pool(
    cache_dir: Path, excluded_path: Path, out_path: Path, seed: int
) -> dict[str, Any]:
    """Query OpenML metadata, validate each candidate and write the frozen pool."""
    excluded = {int(i) for i in excluded_path.read_text().split()}
    meta = openml.datasets.list_datasets(output_format="dataframe", status="active")
    candidates = eligible(meta, excluded)
    hashes, info, rejected = {}, {}, {}
    for did in candidates.did.astype(int):
        try:
            data = load(did, cache_dir)
        except Exception as exc:  # noqa: BLE001 -- any failure excludes the dataset
            rejected[did] = f"load: {exc}"[:200]
            continue
        counts = np.bincount(data.y)
        if (
            counts.min() < MIN_CLASS_SIZE
            or counts.size > 10
            or not 100 <= len(data.y) <= 1000
        ):
            rejected[did] = (
                f"post-load check: n={len(data.y)}, min class={counts.min()}"
            )
            continue
        hashes[did] = data.sha256
        info[did] = dict(
            did=did, name=data.name, version=data.version, sha256=data.sha256
        )
    kept = dedupe_by_hash(hashes)
    dev, held = sample_pool(kept, seed)
    pool = dict(
        seed=seed,
        n_metadata_eligible=int(len(candidates)),
        n_valid_unique=len(kept),
        dev=[info[d] for d in dev],
        heldout=[info[d] for d in held],
        rejected=rejected,
        duplicates=sorted(set(hashes) - set(kept)),
    )
    out_path.write_text(yaml.safe_dump(pool, sort_keys=False))
    return pool
