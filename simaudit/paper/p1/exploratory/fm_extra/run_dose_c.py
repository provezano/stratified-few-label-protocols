"""EXPLORATORY: Study 7 Part (c) (dose-response) on DEV for fm6 TabDPT / fm7 Mitra-v2 / fm8 LimiX-16M.

Plan and value gate: ``dose_plan.md`` (committed before any fit). Design identical to the
pre-registered FMs: the designs and contexts are read, read-only, from the stored
``results/study7/c/{did}_fm2.parquet`` (learner-independent), every context is re-derived with
``runner.study7.draw_context`` (copied below; the runner pulls in fm1-fm5's packages) and asserted
equal, designs that share a seed are fitted once, and each fit is ``runner.study7._fit`` +
``_dose_cells`` re-implemented: ``predict_with_budget`` on all rows, ``complement_metrics`` raw and
after ``prior_correct`` (``pc_*``). Output schema = ``results/study7/c``.

Learners (``--impl``):
- ``ref``: the adapters exactly as in the DEV Part (a) runs (``adapters.py``; Mitra seeded per fit
  as ``run_s7a_v2.seeded_mitra``); LimiX checkpoint with ``weights_only=True`` as ``run_t1p_fm8``.
- ``opt`` (default): the same models with per-fit overheads removed, checked against ``ref`` on the
  pilot (``check_dose_opt.py``): Mitra reuses one Tab2D instead of a 300 MB deepcopy per fit (the
  in-context fit never changes the weights: ``max_epochs = 0``); LimiX builds the network from the
  checkpoint once per process instead of once per fit. TabDPT is unchanged.

Jobs: one job = (FM, table, mult); parts ``{did}_{fm}.m{mult}.parquet`` are merged into
``{did}_{fm}.parquet`` when the three mults are done. Longest jobs first (estimated from the stored
DEV Part (a) seconds of the same FM). Resumable (parts and ``*.partial.parquet`` checkpoints every
10 fits), clean stop with ``{out}/STOP`` or ``{out}/STOP_{fm}``, memory guard (``--min-free-gb``),
``{out}/DONE_{fm}`` at the end. Nothing is written outside ``fm_extra/out/``.

Usage (from simaudit/, in the model's venv; see ``adapters.VENVS``):
    PYTHONPATH=src nice -n 15 .venv_fm_extra_limix/bin/python \
      paper/p1/exploratory/fm_extra/run_dose_c.py fm8 --workers 2 [--datasets 46562,40496] \
      [--reps 0,1] [--out out/dose_c_pilot] [--impl opt]
"""

from __future__ import annotations

import os

# DOSE_C_HISTORICAL_THREADS=1 reproduces the thread settings of the DEV Part (a) runs (OMP/OPENBLAS/
# MKL = 1, torch intra-op = 1; Accelerate (numpy's BLAS/LAPACK here) and torch inter-op left at their
# defaults); used only by the ref side of the optimization check.
HISTORICAL_THREADS = os.environ.get("DOSE_C_HISTORICAL_THREADS") == "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS") + (
        () if HISTORICAL_THREADS else ("VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")):
    os.environ[_v] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights from the local cache only

import argparse  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import adapters as ad  # noqa: E402
from simaudit.analysis.study7 import complement_metrics, prior_correct  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.selection import strategies as sel  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

S7C = settings.results_dir / "study7" / "c"
REFERENCE_FM = "fm2"
DOSE_MULTS = (3, 5, 10)
CHECKPOINT_EVERY = 10
DESIGN_COLS = ["dataset", "cell", "arm", "share", "mult", "rep", "upweighted", "seed", "feasible",
               "counts", "budget", "share_real", "labeled"]
PART_A = dict(fm6="s7a", fm7="s7a_v2", fm8="s7a")  # stored DEV Part (a) runs, for job ordering


# ------------------------------------------------------------------ threads
def _threads() -> None:
    import torch

    torch.set_num_threads(1)
    if HISTORICAL_THREADS:
        return
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # already set in this process
        pass


# ------------------------------------------------------------------ learners
@lru_cache(maxsize=1)
def _limix_ckpt_safe() -> tuple[str, Any]:
    """As ``run_t1p_fm8``: cached weights only, ``weights_only=True``."""
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(ad.LIMIX_REPO, ad.LIMIX_FILE, revision=ad.LIMIX_REVISION,
                           local_files_only=True)
    return path, torch.load(path, map_location="cpu", weights_only=True)


ad._limix_ckpt = _limix_ckpt_safe


def mitra_ref(X_fit: Any, y_fit: Any, X_query: Any, seed: int) -> np.ndarray:
    """``run_s7a_v2.seeded_mitra``: AutoGluon ``set_seed`` then ``adapters.mitra_predict``."""
    from autogluon.tabular.models.mitra._internal.utils.set_seed import set_seed

    set_seed(int(seed))
    return ad.mitra_predict(X_fit, y_fit, X_query, seed)


def mitra_opt(X_fit: Any, y_fit: Any, X_query: Any, seed: int) -> np.ndarray:
    """``mitra_ref`` without the per-fit deepcopy of the 300 MB Tab2D.

    With ``fine_tune=False`` the package sets ``max_epochs = 0``: the trainer runs one validation
    forward, stores ``state_dict()`` references and loads them back (a self-copy), so the weights
    are never changed and one shared instance gives the same predictions (checked on the pilot).
    """
    from autogluon.tabular.models.mitra import sklearn_interface as si
    from autogluon.tabular.models.mitra._internal.utils.set_seed import set_seed

    _threads()
    set_seed(int(seed))
    base = ad._mitra_weights()
    original = si.Tab2D.from_pretrained
    si.Tab2D.from_pretrained = classmethod(lambda cls, *a, **k: base)
    try:
        clf = ad._mitra_classifier(si)(
            fine_tune=False, device="cpu", n_estimators=1, seed=int(seed),
            hf_model=ad._mitra_dir(), verbose=False,
        )
        X = np.asarray(X_fit, np.float64)
        y = np.asarray(y_fit, np.int64)
        clf.fit(X, y, X_val=X[:1], y_val=y[:1])
    finally:
        si.Tab2D.from_pretrained = original
    return ad._columns(clf.predict_proba(np.asarray(X_query, np.float64)), int(y.max()) + 1)


@lru_cache(maxsize=1)
def _limix_network() -> tuple[Any, Any]:
    """The LimiX network built from the checkpoint once per process (``load_from_checkpoint``)."""
    from utils.loading import load_from_checkpoint

    _, ckpt = _limix_ckpt_safe()
    return load_from_checkpoint(ckpt, mask_prediction=False, deterministic=False)


def limix_opt(X_fit: Any, y_fit: Any, X_query: Any, seed: int) -> np.ndarray:
    """``adapters.limix_predict`` with the network built once per process, not once per fit.

    The predictor object (config, seeded preprocessing pipelines, class shuffles) is still
    created per fit with the cell seed; only ``load_from_checkpoint`` is served from a cache.
    The network runs under ``torch.inference_mode`` in eval mode and keeps no state between
    calls (checked on the pilot).
    """
    ad._stub_triton()
    from inference.v1_0 import predictor as v1

    if not getattr(v1, "_dose_c_cached", False):
        v1.load_from_checkpoint = lambda *a, **k: _limix_network()
        v1._dose_c_cached = True
    return ad.limix_predict(X_fit, y_fit, X_query, seed)


def predictor(fm: str, impl: str) -> Callable[..., np.ndarray]:
    _threads()
    if fm == "fm6":
        return ad.tabdpt_predict
    if fm == "fm7":
        return mitra_opt if impl == "opt" else mitra_ref
    if fm == "fm8":
        return limix_opt if impl == "opt" else ad.limix_predict
    raise ValueError(fm)


# ------------------------------------------------------------------ design (runner.study7 copies)
def draw_context(y: np.ndarray, counts: np.ndarray, seed: int) -> np.ndarray:
    """``runner.study7.draw_context``: rows uniformly at random within each class."""
    rng = np.random.default_rng(seed)
    chosen = [rng.choice(np.flatnonzero(y == c), size=int(k), replace=False)
              for c, k in enumerate(counts)]
    return np.sort(np.concatenate(chosen)).astype(np.int64)


def designs(did: int, mult: int, reps: tuple[int, ...] | None) -> pd.DataFrame:
    """Stored Part (c) designs of one table and mult (all arms, shares, reps), in stored order."""
    d = pd.read_parquet(S7C / f"{did}_{REFERENCE_FM}.parquet", columns=DESIGN_COLS)
    d = d[d["mult"] == mult]
    if reps is not None:
        d = d[d["rep"].isin(reps)]
    return d.reset_index(drop=True)


def fit_group(predict: Callable[..., np.ndarray], X: np.ndarray, y: np.ndarray,
              group: pd.DataFrame) -> list[dict[str, Any]]:
    """One fit for every design sharing a seed (``runner.study7._dose_cells.run``)."""
    first = group.iloc[0]
    counts = np.asarray(first["counts"], dtype=np.int64)
    seed = int(first["seed"])
    labeled = draw_context(y, counts, seed)
    for stored in group["labeled"]:
        assert np.array_equal(labeled, np.asarray(stored, dtype=np.int64)), "context differs"
    n_classes = int(y.max()) + 1
    start = time.perf_counter()
    proba = sel.predict_with_budget(predict, X, labeled, y[labeled], n_classes, seed)
    seconds = time.perf_counter() - start
    record = dict(
        n_classes=n_classes, labeled=labeled.tolist(), proba=proba.astype(np.float32).ravel(),
        coverage=sel.class_coverage(y, labeled), seen=np.unique(y[labeled]).tolist(),
        seconds=seconds, **complement_metrics(y, proba, labeled),
    )
    corrected = complement_metrics(y, prior_correct(proba, y[labeled]), labeled)
    record.update({f"pc_{k}": v for k, v in corrected.items()})
    meta_cols = [c for c in DESIGN_COLS if c != "labeled"]
    return [dict({k: r[k] for k in meta_cols}, shared=len(group) > 1, **record)
            for _, r in group.iterrows()]


def infeasible_row(r: pd.Series) -> dict[str, Any]:
    return {k: r[k] for k in DESIGN_COLS if k != "labeled"}


# ------------------------------------------------------------------ guards, io
def available_gb() -> float:
    import subprocess

    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split(" ")[0])
    pages = {k.strip(): int(v.strip(" .")) for k, v in
             (line.split(":") for line in out.splitlines()[1:] if ":" in line)}
    free = pages["Pages free"] + pages["Pages inactive"] + pages["Pages speculative"]
    return free * page / 2**30


def stop_requested(out_dir: Path, fm: str) -> bool:
    return (out_dir / "STOP").exists() or (out_dir / f"STOP_{fm}").exists()


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS


# ------------------------------------------------------------------ jobs
def part_path(out_dir: Path, did: int, fm: str, mult: int) -> Path:
    return out_dir / "parts" / f"{did}_{fm}.m{mult}.parquet"


def run_part(fm: str, did: int, mult: int, out_dir: Path, impl: str,
             reps: tuple[int, ...] | None, min_free_gb: float) -> dict[str, Any]:
    """One (FM, table, mult) job; returns timing and peak RSS of the worker."""
    out = part_path(out_dir, did, fm, mult)
    if out.exists():
        return dict(status="skip", fm=fm, did=did, mult=mult)
    partial = out.with_name(out.name.replace(".parquet", ".partial.parquet"))
    rows: list[dict[str, Any]] = []
    if partial.exists():
        rows = pd.read_parquet(partial).to_dict("records")
    done = {r["_key"] for r in rows}
    t0 = time.perf_counter()
    predict = predictor(fm, impl)
    data = load(did, settings.data_cache_dir)
    d = designs(did, mult, reps)
    for _, r in d[~d["feasible"]].iterrows():
        if r["cell"] not in done:
            rows.append(dict(infeasible_row(r), _key=r["cell"]))
    fresh, fits, fit_s = 0, 0, 0.0
    feasible = d[d["feasible"]]
    for _, group in feasible.groupby("seed", sort=False):
        key = "+".join(group["cell"])
        if key in done:
            continue
        while available_gb() < min_free_gb and not stop_requested(out_dir, fm):
            time.sleep(30)
        if stop_requested(out_dir, fm):
            if rows:
                _write(pd.DataFrame(rows), partial)
            return dict(status="stopped", fm=fm, did=did, mult=mult, fits=fits)
        new = fit_group(predict, data.X, data.y, group)
        rows.extend(dict(x, _key=key) for x in new)
        fits += 1
        fit_s += new[0]["seconds"]
        fresh += 1
        if fresh % CHECKPOINT_EVERY == 0:
            _write(pd.DataFrame(rows), partial)
    frame = pd.DataFrame(rows).drop(columns="_key")
    _write(frame, out)
    partial.unlink(missing_ok=True)
    return dict(status="done", fm=fm, did=did, mult=mult, fits=fits, fit_s=fit_s,
                wall_s=time.perf_counter() - t0, peak_rss_mb=peak_rss_mb(), pid=os.getpid())


def merge(out_dir: Path, did: int, fm: str) -> Path | None:
    """``{did}_{fm}.parquet`` from the three mult parts, in the stored row order."""
    parts = [part_path(out_dir, did, fm, m) for m in DOSE_MULTS]
    if not all(p.exists() for p in parts):
        return None
    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    order = pd.read_parquet(S7C / f"{did}_{REFERENCE_FM}.parquet", columns=["cell"])["cell"]
    rank = {c: i for i, c in enumerate(order)}
    frame = frame.sort_values("cell", key=lambda s: s.map(rank), kind="stable").reset_index(drop=True)
    frame["n_classes"] = int(load(did, settings.data_cache_dir).y.max()) + 1
    frame.insert(0, "learner", fm)
    out = out_dir / f"{did}_{fm}.parquet"
    _write(frame, out)
    return out


def dev_ids() -> list[int]:
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_i2.yaml").read_text())
    return [int(d["did"]) for d in pool["dev"]]


def job_cost(fm: str, did: int, mult: int, reps: tuple[int, ...] | None) -> float:
    """Estimated seconds: unique fits x stored DEV Part (a) mean seconds of this FM at mult."""
    d = designs(did, mult, reps)
    fits = d.loc[d["feasible"], "seed"].nunique()
    a = pd.read_parquet(HERE / "out" / PART_A[fm] / f"{did}_{fm}.parquet", columns=["mult", "seconds"])
    return float(fits * a.loc[a["mult"] == mult, "seconds"].mean())


def init_worker() -> None:
    _threads()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fm", choices=("fm6", "fm7", "fm8"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--datasets", default="", help="comma-separated DEV ids (default: all 20)")
    ap.add_argument("--reps", default="", help="comma-separated reps (default: all 10; pilot only)")
    ap.add_argument("--out", default=str(HERE / "out" / "dose_c"))
    ap.add_argument("--impl", choices=("opt", "ref"), default="opt")
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"DONE_{a.fm}").unlink(missing_ok=True)
    dids = [int(x) for x in a.datasets.split(",")] if a.datasets else dev_ids()
    reps = tuple(int(x) for x in a.reps.split(",")) if a.reps else None
    jobs = sorted(((did, m) for did in dids for m in DOSE_MULTS),
                  key=lambda j: -job_cost(a.fm, j[0], j[1], reps))
    log = out_dir / f"jobs_{a.fm}.jsonl"
    with ProcessPoolExecutor(max_workers=a.workers, initializer=init_worker) as pool:
        futures = {pool.submit(run_part, a.fm, did, m, out_dir, a.impl, reps, a.min_free_gb): (did, m)
                   for did, m in jobs}
        for future in as_completed(futures):
            did, m = futures[future]
            try:
                res = future.result()
            except Exception as exc:  # noqa: BLE001 - logged; other jobs go on
                res = dict(status="failed", fm=a.fm, did=did, mult=m, error=repr(exc))
            res["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with log.open("a") as handle:
                handle.write(pd.Series(res).to_json() + "\n")
            print(res, flush=True)
            merge(out_dir, did, a.fm)
    (out_dir / f"DONE_{a.fm}").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
