"""One (table, FM) unit of the balance-mechanism probe (plan/PLAN.txt, frozen plan; exploratory study, not pre-registered).
Design = evidence study 2026-11-06 (run_manip.py Part A + run_armV.py arm V), unchanged except: levels lambda in {0, 0.5, 1},
seed namespace NS = 62 (new; evidence used 61), tables from plan/frame.csv (content hash checked), per-fit failures recorded
(not retried: seeds are fixed, failures are deterministic) and the host recorded per unit.
Per (m, rep): fixed test split (namespace 36), nested class-wise row orders, one FM seed; fits at the class counts of
lambda 0 (prior-proportional), 0.5, 1 (balanced) and at a covered random composition (V). Coverage = every class present.
Usage: nice -n 15 <venv python> -I probe.py --key K --fm fm1   (cwd: BASE/tmp). Output raw/<key>_<fm>.parquet (atomic)."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import argparse, csv, socket, sys, time  # noqa: E402
from pathlib import Path  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
BASE = Path(__file__).resolve().parents[1]
SIMAUDIT = Path(os.environ.get("SIMAUDIT_ROOT", str(Path.home() / "ge-paper" / "simaudit")))
sys.path.insert(0, str(SIMAUDIT / "src"))
from simaudit.data.openml_loader import content_hash  # noqa: E402
FMS = ("fm1", "fm2", "fm3", "fm5"); MULTS = (3, 5, 10); LEVELS = (0.0, 0.5, 1.0); REPS = 5; NS = 62; BASE_SEED = 20261020


def counts_for(pool_counts, mult, lam):  # verbatim from evidence_2026-11-06/code/run_manip.py
    C = len(pool_counts); B = mult * C
    pi = pool_counts / pool_counts.sum(); w = (1 - lam) * pi + lam / C
    target = np.clip(B * w - 1.0, 0, None); rest = B - C
    target = target * (rest / target.sum()) if target.sum() > 0 else np.full(C, rest / C)
    base = np.floor(target).astype(int); rem = target - base; left = rest - base.sum()
    for k in np.argsort(-rem, kind="stable")[:left]:
        base[k] += 1
    n = base + 1; cap = pool_counts.astype(int)
    over = np.maximum(n - cap, 0).sum(); n = np.minimum(n, cap); order = np.argsort(-w, kind="stable")
    while over > 0:
        moved = False
        for k in order:
            if over and n[k] < cap[k]:
                n[k] += 1; over -= 1; moved = True
        if not moved:
            break
    if lam == 1.0 and np.all(cap >= mult):
        n = np.full(C, mult)
    return n


def frame_row(key):
    with (BASE / "plan/frame.csv").open() as f:
        for r in csv.DictReader(f):
            if r["key"] == key:
                return r
    raise KeyError(key)


def atomic(df, path):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp"); df.to_parquet(tmp, index=False); tmp.replace(path)


def run(key, fm):
    out = BASE / "raw" / f"{key}_{fm}.parquet"
    if out.exists():
        return out
    import torch
    torch.set_num_threads(1)
    from simaudit.analysis.study7 import metric_set
    from simaudit.runner import study13 as s13
    from simaudit.runner.study7 import predictor
    from simaudit.runner.study9 import seed_did
    from simaudit.selection import strategies as sel
    r = frame_row(key); did = int(r["did"])
    with np.load(BASE / "data/npz" / f"{key}.npz", allow_pickle=False) as z:
        X0 = np.asarray(z["X"], dtype=np.float64); y0 = np.asarray(z["y"]).astype(np.int64)
    if content_hash(X0, y0) != r["sha256"]:
        raise SystemExit(f"{key}: content hash != frozen frame")
    perm = s13.row_permutation(len(y0), did, BASE_SEED); X, y = X0[perm], y0[perm]
    test = s13.fixed_split(y, did, BASE_SEED); pool = np.setdiff1d(np.arange(len(y)), test)
    C = int(y.max()) + 1; pool_counts = np.bincount(y[pool], minlength=C); cols = np.arange(C, dtype=np.int64)
    predict = predictor(fm)
    partial = out.with_name(out.name.replace(".parquet", ".partial.parquet"))
    rows, done = [], set()
    if partial.exists():
        old = pd.read_parquet(partial); rows = old.to_dict("records"); done = set(zip(old["mult"].astype(int), old["rep"].astype(int)))
    host = socket.gethostname().split(".")[0]

    def fit(labeled, fm_seed):
        t1 = time.perf_counter()
        try:
            proba = sel.predict_with_budget(predict, X, labeled, y[labeled], C, fm_seed)
            m = metric_set(y[test], proba[test], cols, cols)
            return dict(macro_f1=m["macro_f1"], accuracy=m["accuracy"], balanced_accuracy=m["balanced_accuracy"],
                        seconds=time.perf_counter() - t1, error="")
        except Exception as exc:  # deterministic per seed; recorded, cell dropped by the analysis (plan sec. 5)
            return dict(macro_f1=np.nan, accuracy=np.nan, balanced_accuracy=np.nan, seconds=time.perf_counter() - t1, error=repr(exc)[:300])

    for rep in range(REPS):
        rng = np.random.default_rng([BASE_SEED, seed_did(did), NS, rep])
        class_perm = [rng.permutation(pool[y[pool] == c]) for c in range(C)]
        for mult in MULTS:
            if (mult, rep) in done:
                continue
            fm_seed = int(np.random.SeedSequence([BASE_SEED, seed_did(did), NS, mult, rep, 7]).generate_state(1)[0] % (2**31 - 1))
            fitted = {}
            comps = [(lam, counts_for(pool_counts, mult, lam)) for lam in LEVELS]
            rv = np.random.default_rng([BASE_SEED, seed_did(did), NS, rep, mult, 99])
            for attempt in range(100_000):
                nV = np.bincount(y[rv.choice(pool, size=mult * C, replace=False)], minlength=C)
                if (nV > 0).all():
                    break
            comps.append(("V", nV))
            for lev, n in comps:
                tkey = tuple(int(v) for v in n)
                if tkey not in fitted:
                    labeled = np.sort(np.concatenate([class_perm[c][: n[c]] for c in range(C)])).astype(np.int64)
                    fitted[tkey] = dict(fit(labeled, fm_seed), fitted_here=True)
                    rec = fitted[tkey]
                else:
                    rec = dict(fitted[tkey], fitted_here=False)
                rows.append(dict(dataset=key, learner=fm, mult=mult, rep=rep, level=str(lev), counts=";".join(map(str, tkey)),
                                 budget=int(sum(tkey)), C=C, pool_counts=";".join(map(str, pool_counts)), n_test=len(test),
                                 fm_seed=fm_seed, attempts_V=attempt + 1 if lev == "V" else 0, host=host, **rec))
            atomic(pd.DataFrame(rows), partial)
    atomic(pd.DataFrame(rows), out); partial.unlink(missing_ok=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--key", required=True); ap.add_argument("--fm", required=True, choices=FMS)
    a = ap.parse_args(); (BASE / "raw").mkdir(exist_ok=True)
    print("done", run(a.key, a.fm), flush=True)
