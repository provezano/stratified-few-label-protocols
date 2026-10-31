"""Subprocess worker for the Study 13 secondary FMs (Amendments 1 (B) and 3).

fm6 = TabDPT v1.3, fm7 = Mitra-v2 and fm8 = LimiX-16M (Amendment 3) need
packages that are not in the project ``.venv``; each runs in its own pinned
venv (``.venv_fm_extra``, ``.venv_fm_extra_mitra``,
``.venv_fm_extra_limix``). The Study 13 runner starts this file there with
``{venv}/bin/python -I fm_extra_worker.py {fm}``. It imports only numpy, the
standard library and the model's own packages.

Before the first fit the worker checks the pins of
``paper/p1/exploratory/fm_extra/out/pins.json`` (package versions, Hugging
Face revision and SHA-256 of the weight file, offline only) and fails if any
differs. The model settings are those of the exploratory adapters
(``paper/p1/exploratory/fm_extra/adapters.py``) used for the DEV Part (a) runs:
CPU, one thread, in-context only; TabDPT with its default 8-member ensemble;
Mitra with ``fine_tune=False``, one estimator and float32 instead of the
package's bf16 autocast. One addition for Mitra: the package's own
``set_seed(seed)`` is called with the cell seed before every fit, because
AutoGluon 1.6 no longer seeds per fit and repeated identical Mitra fits
otherwise differ (found on DEV, 2026-10-28).

fm8 (Amendment 3) follows ``adapters.limix_predict``: the default
no-retrieval classification config (``cls_default_noretrieval.json``, its
SHA-256 pinned here), ``mix_precision=False``, the cell seed, one predictor
per fit built from the checkpoint loaded once. The checkpoint bytes are read
once, hashed against the pinned SHA-256 and only then loaded with
``torch.load(..., weights_only=True)`` (no pickle code); a mismatch refuses
to run. A checkpoint that the package routes to anything but the LimiX v1
line (``v1_0``) is refused: LimiX-2 is forbidden.

Protocol, on the original stdout (fd 1 is pointed at stderr so that library
prints cannot corrupt it): frames of an 8-byte big-endian length followed by
``np.savez`` bytes (loaded with ``allow_pickle=False``).

- worker -> parent, once: ``hello`` (JSON of the verified pins) or ``error``;
- parent -> worker: ``X_fit``, ``y_fit``, ``X_query``, ``seed``;
- worker -> parent: ``proba`` or ``error``;
- a zero-length frame from the parent ends the worker.

``uniform`` is a test learner (no model, no pins): seeded random rows.
"""

import hashlib
import io
import json
import os
import struct
import sys
from collections.abc import Callable
from importlib import import_module, metadata
from pathlib import Path
from typing import IO, Any, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Predict = Callable[[FloatArray, NDArray[np.int64], FloatArray, int], FloatArray]

# paper/p1/exploratory/fm_extra/out/pins.json (fm6, fm7, fm8); a unit test checks it.
PINS: dict[str, dict[str, Any]] = {
    "fm6": dict(
        model="TabDPT v1.3",
        venv=".venv_fm_extra",
        packages={
            "tabdpt": "1.3.1",
            "torch": "2.14.1",
            "scikit-learn": "1.9.1",
            "numpy": "2.5.4",
        },
        hf_repo="Layer6/TabDPT",
        hf_revision="a5ca6e01c0fa09ec68c73e958e5199d1932abb3a",
        file="tabdpt1_3.safetensors",
        sha256="97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb",
    ),
    "fm7": dict(
        model="Mitra-v2 (mitra-classifier-2)",
        venv=".venv_fm_extra_mitra",
        packages={"autogluon.tabular": "1.6.3", "torch": "2.10.0", "numpy": "2.5.4"},
        hf_repo="autogluon/mitra-classifier-2",
        hf_revision="edada0d20759c58ada8c8605c25f22f6e98ea5f0",
        file="model.safetensors",
        sha256="5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09",
    ),
    "fm8": dict(
        model="LimiX-16M (LimiX-1_16M)",
        venv=".venv_fm_extra_limix",
        packages={"limix": "2.0.0", "torch": "2.14.1", "numpy": "2.5.4"},
        code_repo="github.com/limix-ldm/LimiX",
        code_commit="516bf396333feb3198cf7aff8a6c10421f218e24",
        hf_repo="stable-ai/LimiX-1_16M",
        hf_revision="3a7c18ac7876fa0f4418cc077abc12630c4fefbf",
        file="LimiX-1_16M.ckpt",
        sha256="0862b8aacc4669ead009ad418c83d33c143396b7aeec069e7d0d667aa70a168a",
    ),
}
TABDPT_ENSEMBLES = 8
LIMIX_CONFIG = "cls_default_noretrieval.json"
LIMIX_CONFIG_SHA256 = "9ad6baac6d4cb8d7f218b76078dbdc801ce9b253f5b7868342df0c2e0b0a1e2f"
LIMIX_LINE = "v1_0"  # LimiX v1 only; LimiX-2 (v2_0) is forbidden
HEADER = struct.Struct(">Q")


# ---------------------------------------------------------------- framing
def write_frame(stream: IO[bytes], **arrays: Any) -> None:
    """One frame: length, then the npz bytes of ``arrays``."""
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    data = buffer.getvalue()
    stream.write(HEADER.pack(len(data)) + data)
    stream.flush()


def read_frame(stream: IO[bytes]) -> dict[str, Any] | None:
    """The next frame as a dict of arrays; None on a zero-length frame or EOF."""
    head = stream.read(HEADER.size)
    if len(head) < HEADER.size:
        return None
    (size,) = HEADER.unpack(head)
    if size == 0:
        return None
    data = stream.read(size)
    if len(data) != size:
        raise EOFError("truncated frame")
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- pins
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pins(fm: str) -> tuple[Path, dict[str, Any]]:
    """Check package versions and the weight file offline; (weights, pins)."""
    pin = PINS[fm]
    for package, want in pin["packages"].items():
        got = metadata.version(package)
        if got != want:
            raise RuntimeError(f"{fm}: {package} {got} != pinned {want}")
    hub = import_module("huggingface_hub")
    folder = hub.snapshot_download(
        pin["hf_repo"],
        revision=pin["hf_revision"],
        local_files_only=True,
    )
    weights = Path(folder) / pin["file"]
    got_sha = _sha256(weights)
    if got_sha != pin["sha256"]:
        raise RuntimeError(f"{fm}: weight SHA-256 {got_sha} != pinned {pin['sha256']}")
    return weights, pin


def read_pinned(path: Path, sha256: str) -> bytes:
    """The file's bytes, read once; refuses if their SHA-256 is not the pin."""
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise RuntimeError(f"{path.name}: SHA-256 {got} != pinned {sha256}")
    return data


# ---------------------------------------------------------------- models
def _threads() -> None:
    import_module("torch").set_num_threads(1)


def _tabdpt(weights: Path) -> Predict:
    _threads()
    model = import_module("tabdpt").TabDPTClassifier(
        device="cpu",
        compile=False,
        use_flash=False,
        verbose=False,
        model_weight_path=str(weights),
    )

    def predict(
        X_fit: FloatArray, y_fit: NDArray[np.int64], X_query: FloatArray, seed: int
    ) -> FloatArray:
        model.fit(X_fit, y_fit)
        proba = model.ensemble_predict_proba(
            X_query, n_ensembles=TABDPT_ENSEMBLES, seed=int(seed)
        )
        return np.asarray(proba, dtype=np.float64)

    return predict


def _mitra(weights: Path) -> Predict:
    import copy

    _threads()
    si: Any = import_module("autogluon.tabular.models.mitra.sklearn_interface")
    tab2d: Any = import_module("autogluon.tabular.models.mitra._internal.models.tab2d")
    seeding: Any = import_module(
        "autogluon.tabular.models.mitra._internal.utils.set_seed"
    )
    parent: Any = si.MitraClassifier
    folder = str(weights.parent)
    base = tab2d.Tab2D.from_pretrained(folder, device="cpu")

    class MitraFP32(parent):  # type: ignore[misc]
        def _create_config(
            self, task: Any, dim_output: Any, time_limit: Any = None
        ) -> Any:
            cfg, model = super()._create_config(task, dim_output, time_limit)
            cfg.hyperparams["precision"] = "float32"
            return cfg, model

    def predict(
        X_fit: FloatArray, y_fit: NDArray[np.int64], X_query: FloatArray, seed: int
    ) -> FloatArray:
        # AutoGluon 1.6 no longer seeds per fit, so Mitra's random preprocessing
        # (mirroring, class shuffle, feature order) drew from the unseeded global
        # numpy RNG and repeated fits differed. Seed it with the cell seed.
        seeding.set_seed(int(seed))
        original = si.Tab2D.from_pretrained
        si.Tab2D.from_pretrained = classmethod(lambda cls, *a, **k: copy.deepcopy(base))
        try:
            clf = MitraFP32(
                fine_tune=False,
                device="cpu",
                n_estimators=1,
                seed=int(seed),
                hf_model=folder,
                verbose=False,
            )
            # The context is its own validation set: no 80/20 split of the context.
            clf.fit(X_fit, y_fit, X_val=X_fit[:1], y_val=y_fit[:1])
        finally:
            si.Tab2D.from_pretrained = original
        return np.asarray(clf.predict_proba(X_query), dtype=np.float64)

    return predict


def _stub_triton() -> None:
    """Install a placeholder ``triton`` module, as ``adapters._stub_triton``.

    LimiX v1 imports triton for a CUDA RMSNorm kernel that its forward never
    calls on CPU; triton has no macOS wheel, so a placeholder module lets the
    import succeed (a real kernel call would fail).
    """
    import types

    if "triton" in sys.modules:
        return

    class _Any(types.ModuleType):
        def __getattr__(self, name: str) -> Any:
            if name.startswith("__"):
                raise AttributeError(name)
            return object

    tl = _Any("triton.language")
    triton: Any = _Any("triton")
    triton.jit = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)
    triton.language = tl
    sys.modules["triton"], sys.modules["triton.language"] = triton, tl


def limix_checkpoint(weights: Path, torch: Any) -> dict[str, Any]:
    """Pinned bytes, then ``torch.load(weights_only=True)`` of exactly those."""
    data = read_pinned(weights, PINS["fm8"]["sha256"])
    ckpt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or not {"state_dict", "config"} <= set(ckpt):
        raise RuntimeError("fm8: checkpoint is not a {state_dict, config} dict")
    return cast(dict[str, Any], ckpt)


def _limix(weights: Path) -> Predict:
    import tempfile

    _threads()
    torch: Any = import_module("torch")
    ckpt = limix_checkpoint(weights, torch)
    version: Any = import_module("model.version")
    line = version.resolve_arch_line(version.resolve_arch_version(ckpt))
    if line != LIMIX_LINE:
        raise RuntimeError(
            f"fm8: checkpoint routes to {line}; only LimiX v1 is allowed"
        )
    config = Path(str(import_module("config").__file__)).parent / LIMIX_CONFIG
    read_pinned(config, LIMIX_CONFIG_SHA256)
    _stub_triton()
    router: Any = import_module("inference.predictor")
    v1: Any = import_module("inference.v1_0.predictor")
    # As the adapter: the package hard-codes cache_dir="/mnt/public/infe_cache"
    # (created even with use_data_cache=False); nothing is cached (flag off).
    cache = os.path.join(tempfile.gettempdir(), "limix_infe_cache")
    v1.LimiXPredictor.CacheManager.__init__.__defaults__ = (cache,)

    def predict(
        X_fit: FloatArray, y_fit: NDArray[np.int64], X_query: FloatArray, seed: int
    ) -> FloatArray:
        clf = router.LimiXPredictor(
            device=torch.device("cpu"),
            model_path=str(weights),
            inference_config=str(config),
            mix_precision=False,
            seed=int(seed),
            ckpt=ckpt,
        )
        proba = clf.predict(X_fit, y_fit, X_query, task_type="Classification")
        return np.asarray(proba, dtype=np.float64)

    return predict


def _uniform() -> Predict:
    def predict(
        X_fit: FloatArray, y_fit: NDArray[np.int64], X_query: FloatArray, seed: int
    ) -> FloatArray:
        p = np.random.default_rng(int(seed)).random(
            (len(X_query), int(y_fit.max()) + 1)
        )
        return np.asarray(p / p.sum(axis=1, keepdims=True), dtype=np.float64)

    return predict


def build(fm: str) -> tuple[Predict, dict[str, Any]]:
    """The predictor of ``fm`` and the pins it was verified against."""
    if fm == "uniform":
        return _uniform(), {}
    weights, pin = verify_pins(fm)
    model = {"fm6": _tabdpt, "fm7": _mitra, "fm8": _limix}[fm]
    return model(weights), pin


def checked(predict: Predict, request: dict[str, Any]) -> FloatArray:
    """Run one fit with the project's input and output contract."""
    X_fit = np.asarray(request["X_fit"], dtype=np.float64)
    y_fit = np.asarray(request["y_fit"], dtype=np.int64)
    X_query = np.asarray(request["X_query"], dtype=np.float64)
    k = int(y_fit.max()) + 1
    if not np.array_equal(np.unique(y_fit), np.arange(k)):
        raise ValueError("y_fit must be coded 0..K-1 with every code present")
    proba = predict(X_fit, y_fit, X_query, int(request["seed"]))
    if proba.shape != (len(X_query), k) or not np.isfinite(proba).all():
        raise ValueError(
            f"bad probabilities: shape {proba.shape}, expected {k} columns"
        )
    return proba


# ---------------------------------------------------------------- main
def serve(fm: str, inbox: IO[bytes], outbox: IO[bytes]) -> None:
    """Hello, then answer requests until a zero-length frame."""
    try:
        predict, pin = build(fm)
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        write_frame(outbox, error=np.array(repr(exc)))
        return
    write_frame(outbox, hello=np.array(json.dumps(pin, sort_keys=True)))
    while (request := read_frame(inbox)) is not None:
        try:
            write_frame(outbox, proba=checked(predict, request))
        except Exception as exc:  # noqa: BLE001 - reported to the parent
            write_frame(outbox, error=np.array(repr(exc)))


def main() -> None:  # pragma: no cover - runs in the FM venv
    """``python -I fm_extra_worker.py {fm6,fm7,fm8,uniform}``."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.environ["HF_HUB_OFFLINE"] = "1"
    protocol = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # library prints go to stderr
    serve(sys.argv[1], sys.stdin.buffer, protocol)


if __name__ == "__main__":  # pragma: no cover
    main()
