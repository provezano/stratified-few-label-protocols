"""Independent learner seed for Study 7 Part (a) (DEV) and Study 13 arm T (HELD-OUT fm2).

EXPLORATORY; see PLAN.md. Same stored labelled sets, only the learner seed changes.

    python run_seed_indep.py dev fm2,fm1 --workers 3
    python run_seed_indep.py dev classical --workers 3
    python run_seed_indep.py repro fm2,fm1          # original-seed reproduction control
    P1_HELDOUT_GO=1 python run_seed_indep.py heldout fm2 --workers 3
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # paper/p1/exploratory (common, e1_classical)

import common as cm  # noqa: E402
from simaudit.analysis.study7 import complement_metrics  # noqa: E402
from simaudit.runner import study4, study7  # noqa: E402
from simaudit.runner.study9 import seed_did  # noqa: E402
from simaudit.selection.strategies import predict_with_budget  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

NAMESPACE = 0xF00D
STRATS = ("stratified", "random")
FMS = ("fm2", "fm1", "fm5", "fm3")
CLASSICAL = ("lr", "knn1", "knn3", "gnb", "lda", "hgb")
OUT = HERE / "out"
DEV_OUT = OUT / "dev"
HO_OUT = settings.results_dir / "seed_indep" / "heldout"  # git-ignored
METRICS = ("macro_f1", "accuracy", "balanced_accuracy")


def fit_seed(base_seed: int, did: int, mult: int, rep: int, strategy: str) -> int:
    """Learner seed in namespace 0xF00D (never used for selection), spawned child."""
    parent = np.random.SeedSequence(
        [base_seed, seed_did(did), mult, rep, study4.ORDER.index(strategy), NAMESPACE]
    )
    child = parent.spawn(1)[0]
    return int(child.generate_state(1)[0] % (2**31 - 1))


def check_streams(sel_seeds: list[int], new_seeds: list[int], all_sel: set[int]) -> None:
    """Assert the fit seeds are different streams from the selection seeds."""
    for s, f in zip(sel_seeds, new_seeds):
        assert s != f, (s, f)
        a = np.random.default_rng(s).integers(0, 2**62, 64)
        b = np.random.default_rng(f).integers(0, 2**62, 64)
        assert not np.array_equal(a, b)
    clash = set(new_seeds) & all_sel
    assert not clash, f"fit seeds coincide with selection seeds: {sorted(clash)[:5]}"


def _init() -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover
        pass


def _atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


# ------------------------------------------------------------------ DEV
def _dev_selections(did: int) -> pd.DataFrame:
    frames = []
    for name in STRATS:
        sel = pd.read_parquet(study7.selection_path(cm.S7, did, name))
        frames.append(sel.assign(strategy=name))
    return pd.concat(frames, ignore_index=True)


def _all_selection_seeds(did: int) -> set[int]:
    seeds: set[int] = set()
    for p in (cm.S7 / "selections").glob(f"{did}_*.parquet"):
        seeds |= set(pd.read_parquet(p, columns=["seed"]).seed.astype(int))
    return seeds


def _check_against_stored(did: int, fm: str, sel: pd.DataFrame) -> pd.DataFrame:
    stored = pd.read_parquet(cm.S7 / "a" / f"{did}_{fm}.parquet")
    stored = stored[stored.strategy.isin(STRATS)]
    m = sel.merge(stored, on=["strategy", "mult", "rep"], suffixes=("", "_st"))
    assert len(m) == len(sel) == len(stored), (did, fm, len(m), len(sel), len(stored))
    assert (m.seed == m.seed_st).all()
    for a, b in zip(m.labeled, m.labeled_st):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    return m


def dev_fm_job(did: int, fm: str) -> str:
    out = DEV_OUT / f"{did}_{fm}.parquet"
    if out.exists():
        return f"skip {out.name}"
    data = cm.dev_data(did)
    X, y = data.X, data.y
    sel = _check_against_stored(did, fm, _dev_selections(did))
    new = [fit_seed(settings.base_seed, did, int(r.mult), int(r.rep), r.strategy)
           for r in sel.itertuples()]
    check_streams(sel.seed.astype(int).tolist(), new, _all_selection_seeds(did))
    predict = study7.predictor(fm)
    rows = []
    for r, s_new in zip(sel.itertuples(), new):
        labeled = np.asarray(r.labeled, dtype=np.int64)
        _, rec = study7._fit(predict, X, y, labeled, s_new)
        row = dict(dataset=did, learner=fm, strategy=r.strategy, mult=int(r.mult),
                   rep=int(r.rep), sel_seed=int(r.seed), fit_seed=s_new,
                   budget=len(labeled), n_classes=rec["n_classes"],
                   coverage=rec["coverage"], seconds=rec["seconds"])
        for k in METRICS:
            row[k] = rec[k]
            row[f"orig_{k}"] = float(getattr(r, k))
        rows.append(row)
    _atomic(pd.DataFrame(rows), out)
    return f"done {out.name}"


def dev_classical_job(did: int) -> str:
    import e1_classical as e1

    out = DEV_OUT / f"{did}_classical.parquet"
    if out.exists():
        return f"skip {out.name}"
    data = cm.dev_data(did)
    X, y = data.X, data.y
    sd = X.std(axis=0)
    Xs = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    C = int(y.max()) + 1
    sel = _dev_selections(did)
    new = [fit_seed(settings.base_seed, did, int(r.mult), int(r.rep), r.strategy)
           for r in sel.itertuples()]
    check_streams(sel.seed.astype(int).tolist(), new, _all_selection_seeds(did))
    rows = []
    for r, s_new in zip(sel.itertuples(), new):
        lab = np.asarray(r.labeled, dtype=np.int64)
        for name in CLASSICAL:
            proba = predict_with_budget(e1.predictor(name), Xs, lab, y[lab], C, s_new)
            met = complement_metrics(y, proba, lab)
            rows.append(dict(dataset=did, learner=name, strategy=r.strategy,
                             mult=int(r.mult), rep=int(r.rep), sel_seed=int(r.seed),
                             fit_seed=s_new, budget=len(lab), n_classes=C,
                             coverage=len(np.unique(y[lab])) / C,
                             **{k: met[k] for k in METRICS}))
    frame = pd.DataFrame(rows)
    orig = pd.read_parquet(cm.OUT / "e1_runs.parquet",
                           columns=["dataset", "learner", "strategy", "mult", "rep", *METRICS])
    orig = orig[orig.dataset == did].rename(columns={k: f"orig_{k}" for k in METRICS})
    frame = frame.merge(orig, on=["dataset", "learner", "strategy", "mult", "rep"])
    assert len(frame) == len(rows), (did, len(frame), len(rows))
    _atomic(frame, out)
    return f"done {out.name}"


def repro_job(did: int, fm: str) -> list[dict]:
    """Refit one S and one R draw (m = 3, rep 0) with the ORIGINAL seed."""
    data = cm.dev_data(did)
    stored = pd.read_parquet(cm.S7 / "a" / f"{did}_{fm}.parquet")
    pick = stored[stored.strategy.isin(STRATS) & (stored.mult == 3) & (stored.rep == 0)]
    predict = study7.predictor(fm)
    rows = []
    for r in pick.itertuples():
        labeled = np.asarray(r.labeled, dtype=np.int64)
        proba, rec = study7._fit(predict, data.X, data.y, labeled, int(r.seed))
        old = np.asarray(r.proba, dtype=np.float32).reshape(proba.shape)
        rows.append(dict(dataset=did, learner=fm, strategy=r.strategy,
                         d_macro_f1=abs(rec["macro_f1"] - r.macro_f1),
                         d_proba=float(np.abs(proba.astype(np.float32) - old).max())))
    return rows


# ------------------------------------------------------------------ HELD-OUT
def _ho_context():
    import yaml

    from simaudit.runner import study13 as s13

    s13.check_heldout_go(PROJECT_ROOT)
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_i2.yaml").read_text())
    entries = pool["heldout"]
    sealed = PROJECT_ROOT.parent / "sealed_heldout"
    log_path = HO_OUT / "access_log.jsonl"
    s13.verify_manifest(sealed, [int(d["did"]) for d in entries], s13.AccessLog(log_path))
    loader = s13.sealed_loader(sealed, {int(d["did"]): str(d["sha256"]) for d in entries},
                               log_path, normalise_labels=True)
    return [int(d["did"]) for d in entries], loader


def ho_job(did: int, fm: str, loader) -> str:
    from simaudit.runner import study13 as s13

    s13.check_heldout_go(PROJECT_ROOT)
    key = s13.dataset_key(did)
    out = HO_OUT / f"{key}_{fm}.parquet"
    if out.exists():
        return f"skip {key}"
    s13_dir = settings.results_dir / "study13" / "heldout"
    data = loader(did, settings.data_cache_dir)
    perm = s13.row_permutation(len(data.y), did, settings.base_seed)
    X, y = data.X[perm], data.y[perm]
    allsel = pd.read_parquet(s13_dir / "selections" / f"{key}.parquet")
    sel = allsel[(allsel.arm == "T") & allsel.strategy.isin(STRATS)].reset_index(drop=True)
    stored = pd.read_parquet(s13_dir / f"{key}_{fm}.parquet")
    stored = stored[(stored.arm == "T") & stored.strategy.isin(STRATS)]
    m = sel.merge(stored, on=["strategy", "mult", "rep"], suffixes=("", "_st"))
    assert len(m) == len(sel) == len(stored) == 100, (key, len(m), len(sel), len(stored))
    assert (m.seed == m.seed_st).all()
    new = [fit_seed(settings.base_seed, did, int(r.mult), int(r.rep), r.strategy)
           for r in m.itertuples()]
    check_streams(m.seed.astype(int).tolist(), new, set(allsel.seed.astype(int)))
    predict = study4._predictor(fm)
    everything = np.arange(len(y), dtype=np.int64)
    rows = []
    for r, s_new in zip(m.itertuples(), new):
        labeled = np.asarray(r.labeled_pos, dtype=np.int64)
        assert np.array_equal(perm[labeled], np.asarray(r.labeled))
        query = np.setdiff1d(everything, labeled)
        rec = s13.fit_record(predict, X, y, labeled, query, s_new, perm)
        row = dict(dataset=key, learner=fm, strategy=r.strategy, mult=int(r.mult),
                   rep=int(r.rep), sel_seed=int(r.seed), fit_seed=s_new,
                   budget=rec["budget"], n_classes=rec["n_classes"],
                   coverage=rec["coverage"], seconds=rec["seconds"])
        for k in METRICS:
            row[k] = rec[k]
            row[f"orig_{k}"] = float(getattr(r, f"{k}"))
        rows.append(row)
    _atomic(pd.DataFrame(rows), out)
    return f"done {key}"


# ------------------------------------------------------------------ driver
def _provenance(mode: str, learners: list[str], workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "provenance.json"
    old = json.loads(path.read_text()) if path.exists() else []
    old.append(dict(mode=mode, learners=learners, workers=workers,
                    host=platform.node(), machine=platform.machine(),
                    started_utc=datetime.now(UTC).isoformat(timespec="seconds"),
                    base_seed=settings.base_seed, namespace=NAMESPACE,
                    nice=os.nice(0)))
    path.write_text(json.dumps(old, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("dev", "repro", "heldout"))
    ap.add_argument("learners")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    learners = args.learners.split(",")
    workers = min(args.workers, 3)
    _provenance(args.mode, learners, workers)
    ids = cm.dev_ids()
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init) as pool:
        if args.mode == "dev":
            futs = {}
            for lr in learners:
                for did in ids:
                    if lr == "classical":
                        futs[pool.submit(dev_classical_job, did)] = (did, lr)
                    else:
                        futs[pool.submit(dev_fm_job, did, lr)] = (did, lr)
        elif args.mode == "repro":
            futs = {pool.submit(repro_job, did, lr): (did, lr)
                    for lr in learners for did in ids[:10]}
        else:
            ho_ids, loader = _ho_context()
            futs = {pool.submit(ho_job, did, lr, loader): (did, lr)
                    for lr in learners for did in ho_ids}
        rows = []
        for i, f in enumerate(as_completed(futs), 1):
            job = futs[f]
            try:
                res = f.result()
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {job}: {exc!r}", flush=True)
                continue
            if args.mode == "repro":
                rows.extend(res)
                res = "repro"
            print(f"[{i}/{len(futs)} {time.time() - t0:.0f}s] {job} {res}", flush=True)
    if args.mode == "repro":
        path = OUT / "repro.csv"
        new = pd.DataFrame(rows)
        if path.exists():
            new = pd.concat([pd.read_csv(path), new]).drop_duplicates(
                ["dataset", "learner", "strategy"], keep="last")
        new.to_csv(path, index=False)
        print(new.groupby("learner")[["d_macro_f1", "d_proba"]].max())


if __name__ == "__main__":
    main()
