"""Stage A (TabEBM venv): splits, TabEBM and SMOTE generation, five classical predictors.

Usage: python -I code/run_stage_a.py --protocol S|R [--workers 6] [--reps 15]
One JSON per (table, N_real, protocol, rep) in out/stage_a/; resumable (existing files are
skipped). For tables with C <= 10, the arrays for the TabPFN stage go to out/arrays/.
"""

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cs_common as cc  # noqa: E402
import numpy as np  # noqa: E402

A_DIR = cc.OUT / "stage_a"
ARR = cc.OUT / "arrays"


def task_name(did, n, prot, r):
    return f"{did}_{n}_{prot}_{r:02d}"


def run_task(did: int, n_real: int, prot: str, r: int) -> str:
    import gen

    out = A_DIR / f"{task_name(did, n_real, prot, r)}.json"
    if out.exists():
        return "skip"
    # Amendment 2 machine deadline: no task starts after 2026-10-31 16:30 UTC (a task takes < 1.5 h)
    import datetime as _dt

    if _dt.datetime.now(_dt.timezone.utc) > _dt.datetime(2026, 10, 31, 16, 30, tzinfo=_dt.timezone.utc) or (cc.OUT / "STOP").exists():
        return "stopped"
    X, y = cc.load_table(did)
    C = int(y.max()) + 1
    pool, test = cc.test_split(X, y, did, r)
    Xp = cc.preprocess(X, pool)
    rec = dict(did=did, n_real=n_real, protocol=prot, rep=r, C=C, status="ok", rows=[])
    try:
        tr, va = cc.draw_subset(pool, y, did, n_real, r, prot)
    except ValueError as e:
        rec.update(status="S_infeasible", error=str(e))
        out.write_text(json.dumps(rec))
        return "infeasible"
    X_tr, y_tr = Xp[tr], y[tr]
    X_te, y_te = Xp[test], y[test]
    p = cc.PROTOCOLS.index(prot)
    (mseed,) = cc.ss_ints(did, r, n_real, p, 3)
    counts = cc.syn_counts(y_tr)
    sub = np.concatenate([tr, va])
    rec.update(
        n_train=int(len(tr)), n_val=int(len(va)), n_seen_train=int(len(np.unique(y_tr))),
        n_seen_subset=int(len(np.unique(y[sub]))), missed_train=bool(len(np.unique(y_tr)) < C),
        train_counts={int(k): int(v) for k, v in zip(*np.unique(y_tr, return_counts=True))},
    )
    t0 = time.perf_counter()
    Xg, yg = gen.tabebm_generate(X_tr, y_tr, counts, mseed)
    rec["t_tabebm"] = time.perf_counter() - t0
    rec["tabebm_finite"] = bool(np.isfinite(Xg).all())
    Xg = np.nan_to_num(Xg, nan=0.0, posinf=0.0, neginf=0.0)
    Xs, ys, nfb = gen.smote_generate(X_tr, y_tr, counts, mseed)
    rec["smote_fallback_classes"] = int(nfb)
    rec["n_syn_smote"] = int(len(ys))
    arms = {
        "baseline": (X_tr, y_tr),
        "smote": (np.vstack([X_tr, Xs]), np.concatenate([y_tr, ys])),
        "tabebm": (np.vstack([X_tr, Xg]), np.concatenate([y_tr, yg])),
    }
    for arm, (Xf, yf) in arms.items():
        for name in cc.CLASSICAL:
            pred = gen.fit_predict(name, Xf, yf, X_te, mseed)
            rec["rows"].append(dict(arm=arm, predictor=name, **cc.metrics(y_te, pred, C)))
    if C <= cc.TABPFN_MAX_C:
        ARR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ARR / f"{task_name(did, n_real, prot, r)}.npz",
            X_tr=X_tr, y_tr=y_tr, Xs=Xs, ys=ys, Xg=Xg, yg=yg, X_te=X_te, y_te=y_te,
            C=np.int64(C), seed=np.int64(mseed),
        )
    rec["t_total"] = time.perf_counter() - t0
    out.write_text(json.dumps(rec))
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=cc.PROTOCOLS, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reps", type=int, default=cc.N_REPS)
    a = ap.parse_args()
    A_DIR.mkdir(parents=True, exist_ok=True)
    cost = {40971: 0, 1472: 1, 40499: 2, 14: 3, 40966: 4, 841: 5}
    tasks = sorted(
        [(d, n, a.protocol, r) for d, n in cc.CELLS for r in range(a.reps)],
        key=lambda t: (cost[t[0]], -t[1], t[3]),
    )
    fails = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_task, *t): t for t in tasks}
        for f in as_completed(futs):
            t = futs[f]
            try:
                print(task_name(*t), f.result(), flush=True)
            except Exception:
                fails += 1
                print(task_name(*t), "FAILED", traceback.format_exc(), flush=True)
    (cc.OUT / f"stage_a_{a.protocol}.done").write_text(f"fails={fails}\n")


if __name__ == "__main__":
    main()
