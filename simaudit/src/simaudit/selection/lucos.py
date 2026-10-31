"""S-LUCoS (Ipas et al. 2026): k-medoids on TabClustPFN row embeddings.

Study 4 Amendment 2 (prereg/STUDY4_LABEL_SELECTION.md, Sec. 7). The embedding
mirrors LUCoS's ``TabClustPFNEncoder.fit_transform``: TabClustPFN's own
``DataPreprocessor`` with LUCoS's defaults (NaN -> 0, z-norm, zero-variance
columns removed), then ``col_embedder`` -> ``row_interactor`` on CPU. The
selection mirrors ``KMedoidsUnderSampler`` (K = budget, Euclidean, k-medoids++,
one initialisation per repetition); LUCoS's ``one_hot_encode`` is the identity
on a float array, so nothing is encoded. Label-free: only X is ever used.

The TabClustPFN code and checkpoint are research-only and live outside git
(``settings.tabclustpfn_root``, ``settings.tabclustpfn_checkpoint``). They are
imported lazily, the checkpoint is loaded only with ``weights_only=True`` after
its SHA-256 is verified, and embeddings are cached on disk by a hash of X.
"""

import hashlib
import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from simaudit.selection.cache import cache_key
from simaudit.selection.strategies import _check_budget, _kmedoids, _kmedoids_inits
from simaudit.settings import settings

CHECKPOINT_SHA256 = "feb7ab7ad21cd4702b57ce92338ace65f0d852ef1bc00653f2d7049a235095a2"
# Top-level package names of the TabClustPFN repository; generic enough to
# shadow other modules, so they are removed from sys.modules after the import.
TABCLUSTPFN_PACKAGES = ("model", "utils", "configs")
# TabClustPFNEncoder defaults in LUCoS (encoders.py).
PREPROCESSING = dict(
    nan_strategy="zero", normalization="z-norm", remove_zero_variance=True
)


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _import_tabclustpfn(root: Path) -> tuple[Any, Any]:
    """Import ``TabClusterIMAB`` and ``DataPreprocessor`` from the pinned code.

    ``root`` is on sys.path only during the import, and the repository's
    generic top-level packages are dropped from sys.modules afterwards, so no
    later ``import utils`` (say) can resolve to TabClustPFN.
    """
    clash = [name for name in TABCLUSTPFN_PACKAGES if name in sys.modules]
    if clash:
        raise RuntimeError(f"modules {clash} already imported; TabClustPFN would clash")
    sys.path.insert(0, str(root))
    try:
        tabcluster = importlib.import_module("model.tabcluster")
        preprocess = importlib.import_module("utils.preprocess")
    finally:
        sys.path.remove(str(root))
        for name in [m for m in sys.modules if m.split(".")[0] in TABCLUSTPFN_PACKAGES]:
            del sys.modules[name]
    return tabcluster.TabClusterIMAB, preprocess.DataPreprocessor


@lru_cache(maxsize=2)
def _load_model(root: Path, checkpoint: Path, sha256: str) -> tuple[Any, Any]:
    """The TabClustPFN model in eval mode on CPU, and its preprocessor class.

    Raises if the checkpoint hash differs or if ``weights_only`` loading fails;
    there is deliberately no fallback to unrestricted unpickling.
    """
    import torch

    found = file_sha256(checkpoint)
    if found != sha256:
        raise ValueError(f"checkpoint {checkpoint} has SHA-256 {found}, not {sha256}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_cls, preprocessor_cls = _import_tabclustpfn(root)
    model = model_cls(**state["config"])
    model.load_state_dict(state["state_dict"], strict=True)
    model.to("cpu")
    model.eval()
    return model, preprocessor_cls


def _encode(X: NDArray[np.float64]) -> NDArray[np.float32]:
    """Row representations as in LUCoS's ``TabClustPFNEncoder.fit_transform``."""
    import pandas as pd
    import torch

    model, preprocessor_cls = _load_model(
        settings.tabclustpfn_root, settings.tabclustpfn_checkpoint, CHECKPOINT_SHA256
    )
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    prepared = preprocessor_cls(**PREPROCESSING).fit_transform(frame)
    tensor = torch.from_numpy(prepared.astype(np.float64)).float().unsqueeze(0)
    with torch.no_grad():
        rows = model.row_interactor(model.col_embedder(tensor))
    return np.asarray(rows.squeeze(0).detach().cpu().numpy(), dtype=np.float32)


def _cache_dir() -> Path:
    return settings.data_cache_dir / "tabclustpfn"


def embed(X: NDArray[np.float64], cache_dir: Path | None = None) -> NDArray[np.float32]:
    """Label-free TabClustPFN row embeddings of X (n x 512), disk-cached by X."""
    directory = cache_dir or _cache_dir()
    path = directory / f"{cache_key(X)}.npy"
    if path.exists():
        return np.asarray(np.load(path), dtype=np.float32)
    Z = _encode(np.asarray(X, dtype=np.float64))
    directory.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, Z)
    tmp.replace(path)
    return Z


def select_lucos(
    X: NDArray[np.float64], budget: int, seed: int, n_init: int = 1
) -> NDArray[np.int64]:
    """S-LUCoS: k-medoids (K = budget, Euclidean) on the TabClustPFN embeddings.

    Flagged adaptation (Amendment 2): LUCoS fits sklearn-extra's ``KMedoids``
    (method "alternate", k-medoids++, max_iter 300, float32 distances); this
    reuses our ``_kmedoids`` with ``n_init=1`` instead: alternating updates,
    k-medoids++ by plain D^2 sampling with numpy's RNG, at most 100 iterations,
    float64 distances. If that single initialisation ends with fewer than K
    distinct medoids (possible only with duplicate embeddings), the 10-init
    routine of S-KM is applied to the same embeddings. ``n_init`` > 1 keeps the
    lowest-cost of that many initialisations (Study 10 sensitivity arm LUCoS-10).
    """
    _check_budget(len(X), budget)
    Z = embed(X).astype(np.float64)
    chosen = _kmedoids_inits(Z, budget, seed, n_init=n_init)
    return _kmedoids(Z, budget, seed, n_init=10) if chosen is None else chosen
