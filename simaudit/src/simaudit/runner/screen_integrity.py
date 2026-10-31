"""Study 8 Amendment 2 (A2.6): run the integrity screens on every pooled dataset."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from simaudit.data.integrity import OVERLAP, overlap_share, screen
from simaudit.data.openml_loader import Dataset

DidLoader = Callable[[int], Dataset]
# A2.3a priority among the expansion tiers (earlier wins).
TIER_RANK = {"2a": 0, "3": 1, "4": 2}


def _overlap(candidate: Dataset, kept: Dataset) -> float:
    """Share of the candidate's rows found in a kept dataset (A2.3b)."""
    return overlap_share(candidate.X, kept.X)


def overlap_exclusions(
    entries: Sequence[dict[str, Any]],
    loader: DidLoader,
    protected: Mapping[str, Dataset],
) -> dict[str, str]:
    """Datasets sharing >= OVERLAP of rows with DEV or a higher-priority dataset."""
    ranked = sorted(
        entries, key=lambda e: (TIER_RANK[str(e["subtier"])], str(e["key"]))
    )
    kept: dict[str, Dataset] = dict(protected)
    out: dict[str, str] = {}
    for e in ranked:
        data = loader(int(e["did"]))
        hit = next(
            ((k, v) for k, d in kept.items() if (v := _overlap(data, d)) >= OVERLAP),
            None,
        )
        if hit is None:
            kept[str(e["key"])] = data
        else:
            out[str(e["key"])] = f"exclude: row overlap {hit[1]:.2f} with {hit[0]}"
    return out


def screen_pools(
    pools: Sequence[Path],
    loader: DidLoader,
    out_path: Path,
    protected: Mapping[str, Dataset] | None = None,
) -> list[dict[str, Any]]:
    """Screen every dataset of ``pools`` in stored row order; write the audit CSV.

    Row-overlap exclusions (A2.3a) replace a non-excluding screen decision.
    """
    entries = [
        e for path in pools for e in yaml.safe_load(path.read_text())["datasets"]
    ]
    overlap = overlap_exclusions(entries, loader, protected or {})
    rows = []
    for e in entries:
        data = loader(int(e["did"]))
        result = screen(data.X, data.y).as_row()
        key = str(e["key"])
        if key in overlap and not str(result["decision"]).startswith("exclude"):
            result["decision"] = overlap[key]
        rows.append(dict(key=key, did=e["did"], subtier=e["subtier"], **result))
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return rows


def read_screens(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Decision and dropped column indices per dataset key."""
    frame = pd.read_csv(path, keep_default_na=False, dtype={"drop_columns": str})
    return {
        str(r.key): (
            str(r.decision),
            tuple(int(c) for c in str(r.drop_columns).split(";") if c),
        )
        for r in frame.itertuples()
    }
