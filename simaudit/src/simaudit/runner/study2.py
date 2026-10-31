"""Study 2, Stage A: every SSL method x 16 configurations on the pilot splits.

One job = one (dataset, method). Label-free pretraining (VIME, SCARF) runs once
per (repetition, configuration) and is shared across m; FM base predictions for
C&S / LLGC-prior are computed once per cell and shared across configurations.
Each job writes ``{did}_{method}.parquet`` atomically and is skipped if present
(prereg/STUDY2_BASELINES.md, Sec. 9).
"""

import itertools
import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from simaudit.arms import fm
from simaudit.arms import graph_nn as gn
from simaudit.arms import ssl_deep as sd
from simaudit.arms import stunt as st
from simaudit.data.openml_loader import load
from simaudit.data.splits import draw_split
from simaudit.graph.knn import mutual_knn_graph, zscore
from simaudit.graph.propagation import llgc
from simaudit.runner.pilot import M_VALUES, _provenance_name, cell_seed, source_digest

STAGE_A_REPS = 5
GRAPH_K = 10
TORCH_THREADS = 1


@dataclass(frozen=True)
class LLGCPriorConfig:
    """LLGC with an FM prior on the mutual kNN graph of z-scored features."""

    alpha: float = 0.9
    k: int = 10


def llgc_prior_grid() -> list[LLGCPriorConfig]:
    """16 configurations, default (0.9, 10) first."""
    rest = [
        LLGCPriorConfig(a, k)
        for a, k in itertools.product((0.5, 0.7, 0.9, 0.99), (5, 10, 20, 40))
        if (a, k) != (0.9, 10)
    ]
    return [LLGCPriorConfig(), *rest]


def _default_first(grid: list[Any], default: Any) -> list[Any]:
    return [default, *[c for c in grid if c != default]]


def _fm_base(version: str) -> Callable[..., NDArray[np.float64]]:
    def predict(X_fit: Any, y_fit: Any, X_query: Any, seed: int) -> NDArray[np.float64]:
        model = fm.make_tabpfn(seed) if version == "v2" else fm.make_tabpfn_v35(seed)
        n_classes = int(y_fit.max()) + 1
        proba = np.zeros((len(X_query), n_classes))
        model.fit(X_fit, y_fit)
        proba[:, model.classes_] = model.predict_proba(X_query)
        return proba

    return predict


GRIDS: dict[str, list[Any]] = {
    "knn1": [None],
    "ncm": [None],
    "vime": _default_first(sd.grid("vime"), sd.VimeConfig()),
    "scarf": _default_first(sd.grid("scarf"), sd.ScarfConfig()),
    "scarf_st": [sd.ScarfConfig(self_training=True)],
    "gcn": gn.grid("gcn"),
    "appnp": gn.grid("appnp"),
    "cs_logreg": gn.grid("cs"),
    "cs_fm1": gn.grid("cs"),
    "cs_fm5": gn.grid("cs"),
    "llgc_fm1": llgc_prior_grid(),
    "llgc_fm5": llgc_prior_grid(),
    "stunt": st.grid(),
}
BASES: dict[str, Any] = {
    "cs_logreg": "logreg",
    "cs_fm1": _fm_base("v2"),
    "cs_fm5": _fm_base("v35"),
    "llgc_fm1": _fm_base("v2"),
    "llgc_fm5": _fm_base("v35"),
}


def pretrain_seed(base_seed: int, did: int, rep: int, config_index: int) -> int:
    """Seed of label-free pretraining: independent of m, so encoders are shared."""
    state = np.random.SeedSequence([base_seed, did, rep, config_index, 7])
    return int(state.generate_state(1)[0] % (2**31 - 1))


class _Cell:
    """Lazy per-split state shared by every configuration of one method."""

    def __init__(self, X: NDArray[np.float64], labeled: Any, y_lab: Any, seed: int):
        self.X, self.labeled, self.y_lab, self.seed = X, labeled, y_lab, seed
        self._base: NDArray[np.float64] | None = None
        self._graphs: dict[int, sparse.csr_matrix] = {}

    def base(self, base: Any) -> NDArray[np.float64]:
        if self._base is None:
            self._base = gn.base_predictions(
                self.X, self.labeled, self.y_lab, self.seed, base
            )
        return self._base

    def graph(self, k: int) -> sparse.csr_matrix:
        if k not in self._graphs:
            self._graphs[k] = mutual_knn_graph(zscore(self.X), k)
        return self._graphs[k]


def _predict(
    method: str,
    config: Any,
    cell: _Cell,
    encoder: Any,
) -> NDArray[np.float64]:
    X, lab, y_lab, seed = cell.X, cell.labeled, cell.y_lab, cell.seed
    if method in ("knn1", "ncm"):
        return sd.fit_predict(method, X, lab, y_lab, seed, None)
    if method == "stunt":
        return st.predict(encoder, X, lab, y_lab)
    if method in ("vime", "scarf", "scarf_st"):
        name = "vime" if method == "vime" else "scarf"
        return sd.finetune(name, encoder, X, lab, y_lab, seed, config)
    if method in ("gcn", "appnp"):
        return gn.fit_predict(method, X, cell.graph(GRAPH_K), lab, y_lab, seed, config)
    Z = cell.base(BASES[method])
    if method.startswith("cs_"):
        return gn.cs_apply(cell.graph(GRAPH_K), Z, lab, y_lab, config)
    prior = Z.copy()
    prior[lab] = np.eye(Z.shape[1])[y_lab]
    return np.asarray(llgc(cell.graph(config.k), prior, config.alpha), dtype=np.float64)


def _macro_f1(y: NDArray[np.int64], pred: NDArray[np.int64], n_classes: int) -> float:
    from sklearn.metrics import f1_score

    return float(
        f1_score(
            y, pred, labels=list(range(n_classes)), average="macro", zero_division=0
        )
    )


def run_job(
    did: int,
    method: str,
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    grids: dict[str, list[Any]] | None = None,
    reps: int = STAGE_A_REPS,
    m_values: tuple[int, ...] = M_VALUES,
) -> Path:
    """All repetitions, configurations and m for one (dataset, method)."""
    out = out_dir / f"{did}_{method}.parquet"
    if out.exists():
        return out
    grid = (grids or GRIDS)[method]
    data = load(did, cache_dir)
    n_classes = int(data.y.max()) + 1
    rows = []
    shared: dict[int, Any] = {}  # STUNT: one pretraining per configuration
    for rep in range(reps):
        encoders: dict[int, Any] = shared if method == "stunt" else {}
        for m in m_values:
            seed = cell_seed(base_seed, did, m, rep)
            lab, unl = draw_split(data.y, m, seed)
            cell = _Cell(data.X, lab, data.y[lab], seed)
            for index, config in enumerate(grid):
                start = time.perf_counter()
                encoder = _encoder(
                    method,
                    data.X,
                    config,
                    index,
                    encoders,
                    base_seed,
                    did,
                    rep,
                    n_classes,
                )
                proba = _predict(method, config, cell, encoder)
                pred = proba[unl].argmax(axis=1)
                rows.append(
                    dict(
                        dataset=did,
                        m=m,
                        rep=rep,
                        seed=seed,
                        method=method,
                        config=index,
                        is_default=index == 0,
                        macro_f1=_macro_f1(data.y[unl], pred, n_classes),
                        accuracy=float(np.mean(pred == data.y[unl])),
                        seconds=time.perf_counter() - start,
                        pv_score=getattr(encoder, "pv_accuracy", float("nan")),
                    ),
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(out)
    return out


def _encoder(
    method: str,
    X: NDArray[np.float64],
    config: Any,
    index: int,
    encoders: dict[int, Any],
    base_seed: int,
    did: int,
    rep: int,
    n_classes: int = 0,
) -> Any:
    if method == "stunt":
        if index not in encoders:  # label-free; shared across reps and m (Sec. 9)
            seed = pretrain_seed(base_seed, did, 0, index)
            encoders[index] = st.pretrain(X, n_classes, seed, config)
        return encoders[index]
    if method not in ("vime", "scarf", "scarf_st"):
        return None
    if index not in encoders:
        name = "vime" if method == "vime" else "scarf"
        seed = pretrain_seed(base_seed, did, rep, index)
        encoders[index] = sd.pretrain(name, X, seed, config)
    return encoders[index]


def _init_worker() -> None:
    import torch

    torch.set_num_threads(TORCH_THREADS)


def run_stage_a(
    dataset_ids: list[int],
    methods: list[str],
    cache_dir: Path,
    out_dir: Path,
    base_seed: int,
    workers: int,
) -> list[Path]:
    """Run every (dataset, method) job in parallel, slowest methods first."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _provenance_name()).write_text(
        json.dumps(
            dict(
                base_seed=base_seed,
                reps=STAGE_A_REPS,
                m_values=M_VALUES,
                graph_k=GRAPH_K,
                torch_threads=TORCH_THREADS,
                methods={m: len(GRIDS[m]) for m in methods},
                datasets=dataset_ids,
                source_sha256=source_digest(Path(__file__).resolve().parents[1]),
            ),
            indent=2,
        ),
    )
    os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_THREADS))
    jobs = [(did, m) for m in methods for did in dataset_ids]
    done = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {
            pool.submit(run_job, did, m, cache_dir, out_dir, base_seed): (did, m)
            for did, m in jobs
        }
        for future in as_completed(futures):
            did, method = futures[future]
            try:
                done.append(future.result())
            except Exception as exc:  # noqa: BLE001 - logged; other jobs go on
                _record_failure(out_dir, did, method, exc)
                print(f"FAILED {(did, method)}: {exc!r}", flush=True)
                continue
            print(f"done {(did, method)} ({len(done)}/{len(jobs)})", flush=True)
    return done


def _record_failure(out_dir: Path, did: int, method: str, exc: Exception) -> None:
    line = dict(dataset=did, method=method, error=repr(exc))
    with (out_dir / "failures.jsonl").open("a") as handle:
        handle.write(json.dumps(line) + "\n")
