"""Download, validate and cache OpenML datasets as float64 X and int64 y."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import openml
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class Dataset:
    """A validated dataset with label-encoded target (classes 0..C-1)."""

    did: int
    name: str
    version: int
    X: NDArray[np.float64]
    y: NDArray[np.int64]
    sha256: str


def encode(
    X_frame: pd.DataFrame, y_series: pd.Series
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return float64 features and labels encoded in sorted order of their strings.

    Raises:
        ValueError: if a feature is not numeric or any value is missing.
    """
    non_numeric = [c for c in X_frame if not pd.api.types.is_numeric_dtype(X_frame[c])]
    if non_numeric:
        raise ValueError(f"non-numeric features: {non_numeric[:5]}")
    if X_frame.isna().any().any() or y_series.isna().any():
        raise ValueError("missing values")
    _, y = np.unique(y_series.astype(str).to_numpy(), return_inverse=True)
    return X_frame.to_numpy(dtype=np.float64), y.astype(np.int64)


def content_hash(X: NDArray[np.float64], y: NDArray[np.int64]) -> str:
    """SHA-256 of the row-sorted (X, y); invariant to row order."""
    rows = np.column_stack([X, y.astype(np.float64)])
    rows = rows[np.lexsort(rows.T[::-1])]
    return hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()


def load(did: int, cache_dir: Path) -> Dataset:
    """Load a dataset from the local cache, downloading it from OpenML if needed."""
    path = cache_dir / "npz" / f"{did}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return Dataset(
            did, str(z["name"]), int(z["version"]), z["X"], z["y"], str(z["sha"])
        )
    openml.config.set_root_cache_directory(str(cache_dir / "openml"))
    ds = openml.datasets.get_dataset(
        did,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    target = ds.default_target_attribute
    if not target or "," in target:
        raise ValueError(f"no single default target: {target!r}")
    X_frame, y_series, _, _ = ds.get_data(target=target, dataset_format="dataframe")
    X, y = encode(cast(pd.DataFrame, X_frame), cast(pd.Series, y_series))
    sha = content_hash(X, y)
    version = int(ds.version or 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, X=X, y=y, name=ds.name, version=version, sha=sha)
    tmp.replace(path)
    return Dataset(did, ds.name, version, X, y, sha)
