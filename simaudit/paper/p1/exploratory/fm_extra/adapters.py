"""Thin adapters for extra tabular FMs (P1 exploratory, fm_extra feasibility).

Each predictor has the project's FM interface (``harness.fm_tabicl`` and
``runner.study4._predictor``)::

    predict(X_fit, y_fit, X_query, seed) -> proba  # (len(X_query), y_fit.max() + 1)

``y_fit`` is already re-encoded to 0..K-1 over the seen classes by
``selection.strategies.predict_with_budget``, which also scatters the columns back
to all C classes (unseen classes get 0) and skips the FM for one-class contexts.
So these adapters never see a missing class, exactly like fm1-fm5.

The models live in separate virtual environments (never the project .venv):

- fm6 TabDPT      -> .venv_fm_extra        (tabdpt, Apache-2.0 code + weights)
- fm7 Mitra-2     -> .venv_fm_extra_mitra  (autogluon.tabular[mitra], Apache-2.0)
- fm8 LimiX-16M   -> .venv_fm_extra_limix  (LimiX repo, Stable AI License v1.0)

Settings: CPU, one torch thread, in-context only (no fine-tuning), the package's
default ensemble size. Model objects are cached per process (weights load once).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

Predictor = Callable[..., NDArray[np.float64]]

# Pinned releases (latest as of 2026-10-27); hashes = HF LFS sha256 of the weight file.
TABDPT_FILE = "tabdpt1_3.safetensors"  # Layer6/TabDPT @ a5ca6e01c0fa
MITRA_REPO = "autogluon/mitra-classifier-2"  # Mitra-v2 @ edada0d20759
MITRA_REVISION = "edada0d20759c58ada8c8605c25f22f6e98ea5f0"
LIMIX_REPO, LIMIX_FILE = "stable-ai/LimiX-1_16M", "LimiX-1_16M.ckpt"  # @ 3a7c18ac7876
LIMIX_REVISION = "3a7c18ac7876fa0f4418cc077abc12630c4fefbf"
TABDPT_ENSEMBLES = 8  # TabDPTClassifier.predict default


def _threads() -> None:
    import torch

    torch.set_num_threads(1)


def _columns(proba: Any, k: int) -> NDArray[np.float64]:
    p = np.asarray(proba, dtype=np.float64)
    if p.shape[1] != k:
        raise ValueError(f"expected {k} columns, got {p.shape[1]}")
    return p


# ---------------------------------------------------------------- fm6 TabDPT
@lru_cache(maxsize=1)
def _tabdpt() -> Any:
    _threads()
    from tabdpt import TabDPTClassifier

    return TabDPTClassifier(device="cpu", compile=False, use_flash=False, verbose=False)


def tabdpt_predict(X_fit: Any, y_fit: Any, X_query: Any, seed: int) -> NDArray[np.float64]:
    """TabDPT v1.3 (Layer6; pretrained on real tables), 8-member ensemble, full context."""
    model = _tabdpt()
    model.fit(np.asarray(X_fit, np.float64), np.asarray(y_fit, np.int64))
    proba = model.ensemble_predict_proba(
        np.asarray(X_query, np.float64), n_ensembles=TABDPT_ENSEMBLES, seed=int(seed)
    )
    return _columns(proba, int(y_fit.max()) + 1)


# ---------------------------------------------------------------- fm7 Mitra-2
@lru_cache(maxsize=1)
def _mitra_dir() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(MITRA_REPO, revision=MITRA_REVISION)


@lru_cache(maxsize=1)
def _mitra_weights() -> Any:
    """Load Tab2D once; MitraClassifier would reload 300 MB on every fit."""
    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

    return Tab2D.from_pretrained(_mitra_dir(), device="cpu")


@lru_cache(maxsize=1)
def _mitra_classifier(si: Any) -> Any:
    """MitraClassifier computing in float32 instead of bfloat16 autocast.

    The package hard-codes ``precision="bfloat16"``; on this CPU a bf16-autocast
    matmul is ~300x slower than fp32 (0.17 s vs 53 s for 20 1024x512x2048
    products), giving 166 s per fit. fp32 is the more exact arithmetic of the same
    weights; only rounding differs from the GPU default.
    """

    class MitraFP32(si.MitraClassifier):
        def _create_config(self, task, dim_output, time_limit=None):  # type: ignore[no-untyped-def]
            cfg, model = super()._create_config(task, dim_output, time_limit)
            cfg.hyperparams["precision"] = "float32"
            return cfg, model

    return MitraFP32


def mitra_predict(X_fit: Any, y_fit: Any, X_query: Any, seed: int) -> NDArray[np.float64]:
    """Mitra-v2 (AutoGluon), in-context only (fine_tune=False), one estimator (AG default)."""
    import copy

    _threads()
    from autogluon.tabular.models.mitra import sklearn_interface as si

    base = _mitra_weights()
    original = si.Tab2D.from_pretrained
    si.Tab2D.from_pretrained = classmethod(lambda cls, *a, **k: copy.deepcopy(base))
    try:
        clf = _mitra_classifier(si)(
            fine_tune=False, device="cpu", n_estimators=1, seed=int(seed),
            hf_model=_mitra_dir(), verbose=False,
        )
        X = np.asarray(X_fit, np.float64)
        y = np.asarray(y_fit, np.int64)
        # Pass the context as its own validation set: skips the 80/20 split, which
        # cannot be stratified at one row per class and would drop context rows.
        clf.fit(X, y, X_val=X[:1], y_val=y[:1])
    finally:
        si.Tab2D.from_pretrained = original
    return _columns(clf.predict_proba(np.asarray(X_query, np.float64)), int(y.max()) + 1)


# ---------------------------------------------------------------- fm8 LimiX-16M
@lru_cache(maxsize=1)
def _limix_ckpt() -> tuple[str, Any]:
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(LIMIX_REPO, LIMIX_FILE, revision=LIMIX_REVISION)
    return path, torch.load(path, map_location="cpu", weights_only=False)


def _stub_triton() -> None:
    """LimiX v1 imports triton at module level for a CUDA RMSNorm kernel, but its
    ``TritonRMSNorm.forward`` falls back to ``F.rms_norm`` when the input is not on
    CUDA, so the kernel is never called on CPU. Triton has no macOS wheel; a stub
    module (``jit`` = identity, ``tl.*`` placeholders for annotations) lets the
    import succeed. Any real kernel call would fail loudly, not silently."""
    import sys
    import types

    if "triton" in sys.modules:
        return

    class _Any(types.ModuleType):
        def __getattr__(self, name: str) -> Any:
            if name.startswith("__"):
                raise AttributeError(name)
            return object

    tl = _Any("triton.language")
    triton = _Any("triton")
    triton.jit = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)  # type: ignore[attr-defined]
    triton.language = tl  # type: ignore[attr-defined]
    sys.modules["triton"], sys.modules["triton.language"] = triton, tl


def _limix_config() -> str:
    import config  # LimiX ships its JSON configs as the ``config`` package

    return str(Path(config.__file__).parent / "cls_default_noretrieval.json")


def limix_predict(X_fit: Any, y_fit: Any, X_query: Any, seed: int) -> NDArray[np.float64]:
    """LimiX-16M, default no-retrieval classification config, fp32 on CPU."""
    import torch

    import tempfile

    _threads()
    _stub_triton()
    from inference.predictor import LimiXPredictor
    from inference.v1_0 import predictor as v1

    # The package hard-codes cache_dir="/mnt/public/infe_cache" (created even with
    # use_data_cache=False); point it at a temp dir. Nothing is cached (flag off).
    cache = os.path.join(tempfile.gettempdir(), "limix_infe_cache")
    v1.LimiXPredictor.CacheManager.__init__.__defaults__ = (cache,)
    path, ckpt = _limix_ckpt()
    clf = LimiXPredictor(
        device=torch.device("cpu"), model_path=path, inference_config=_limix_config(),
        mix_precision=False, seed=int(seed), ckpt=ckpt,
    )
    proba = clf.predict(
        np.asarray(X_fit, np.float64), np.asarray(y_fit, np.int64),
        np.asarray(X_query, np.float64), task_type="Classification",
    )
    return _columns(proba, int(y_fit.max()) + 1)


PREDICTORS: dict[str, tuple[str, Predictor]] = dict(
    fm6=("TabDPT v1.3", tabdpt_predict),
    fm7=("Mitra-v2", mitra_predict),
    fm8=("LimiX-16M", limix_predict),
)
VENVS = dict(fm6=".venv_fm_extra", fm7=".venv_fm_extra_mitra", fm8=".venv_fm_extra_limix")
