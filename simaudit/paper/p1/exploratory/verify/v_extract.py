"""Independent verification (E15/E17/theory): per-fit feature extraction.

Written from scratch; does NOT import any analyst script under paper/p1/exploratory.
Reads results/study9/t1prime and results/study7/a (read only) plus the frozen
dataset cache through simaudit.data.openml_loader.load (read only).

For every stored fit it recomputes, from the stored float32 probabilities and the
true labels, the frozen five metrics (as a check) and a wider panel:
Brier, kappa, MCC, all-class AUC, CITL, binary LR+/LR-, per-class recalls,
majority/minority recall, labelled majority share, coverage, the T3 cap.

Output: verify/out/fits_t1.parquet and verify/out/fits_dev.parquet
Run: OMP_NUM_THREADS=1 nice -n 15 uv run python paper/p1/exploratory/verify/v_extract.py
"""

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "out"
T1 = ROOT / "results" / "study9" / "t1prime"
DEV = ROOT / "results" / "study7" / "a"
CACHE = ROOT / "data" / "cache"
BASE_SEED = 20261020
CLIP = 1e-6

sys.path.insert(0, str(ROOT / "src"))
from simaudit.data.openml_loader import load  # noqa: E402  (read-only loader)


# ------------------------------------------------------------------ helpers
def seed_did(did: int) -> int:
    return did if did >= 0 else 2**40 + did


def t1_perm(n: int, did: int) -> np.ndarray:
    """Re-derived (not imported) row permutation of the Study 9 runner (ns 13)."""
    return np.random.default_rng([BASE_SEED, seed_did(did), 13]).permutation(n)


def auc_binary(pos: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties (float64)."""
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_features(y: np.ndarray, P: np.ndarray, lab_y: np.ndarray, C: int,
                 maj: int, mino: int) -> dict:
    """All metrics for one fit. y: query labels; P: (len(y), C) float64."""
    seen = np.unique(lab_y)
    pred = P.argmax(axis=1)
    K = C
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    sup = cm.sum(axis=1).astype(float)
    npred = cm.sum(axis=0).astype(float)
    den = sup + npred
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    present = sup > 0
    rec = np.where(present, tp / np.where(present, sup, 1), np.nan)
    n = len(y)
    acc = tp.sum() / n
    # kappa / MCC
    po = acc
    pe = float((sup * npred).sum()) / n**2
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    cov_ytyp = tp.sum() * n - (sup * npred).sum()
    d1 = n**2 - (npred**2).sum()
    d2 = n**2 - (sup**2).sum()
    mcc = cov_ytyp / np.sqrt(d1 * d2) if d1 > 0 and d2 > 0 else 0.0
    # log loss (frozen convention: clip 1e-6, renormalise)
    Q = np.clip(P, CLIP, 1.0)
    Q = Q / Q.sum(axis=1, keepdims=True)
    ll = float(-np.mean(np.log(Q[np.arange(n), y])))
    # Brier (multiclass sum), raw probabilities
    Y1 = np.zeros_like(P)
    Y1[np.arange(n), y] = 1.0
    brier = float(np.mean(((P - Y1) ** 2).sum(axis=1)))
    # AUCs
    auc_cols = {}
    for c in range(C):
        pos = y == c
        if 0 < pos.sum() < n:
            auc_cols[c] = auc_binary(pos, P[:, c])
    seen_aucs = [auc_cols[c] for c in seen if c in auc_cols]
    auc_seen = float(np.mean(seen_aucs)) if seen_aucs and len(seen) > 1 else np.nan
    auc_all = float(np.mean(list(auc_cols.values()))) if auc_cols else np.nan
    # CITL: L1 between mean predicted distribution and query class shares
    share = sup / n
    citl = float(np.abs(P.mean(axis=0) - share).sum())
    # T3 per-draw macro-F1 cap
    u = sup
    unseen_mask = np.ones(C, bool)
    unseen_mask[seen] = False
    A = float(u[unseen_mask].sum())
    seen_present = [j for j in seen if u[j] > 0]
    umax = max([u[j] for j in seen_present], default=0.0)
    cap = (len(seen_present) - (A / (2 * umax + A) if A > 0 else 0.0)) / C
    out = dict(
        my_macro_f1=float(f1.mean()), my_accuracy=float(acc),
        my_balanced_accuracy=float(np.nanmean(rec)), my_log_loss=ll,
        my_auc_seen=auc_seen, auc_all=auc_all, brier=brier, kappa=float(kappa),
        mcc=float(mcc), citl=citl, n_seen=len(seen), mf1_cap=cap,
        rec_maj=float(rec[maj]), rec_min=float(rec[mino]),
        pred_maj_rate=float(npred[maj] / n),
        lab_maj_share=float(np.mean(lab_y == maj)),
        lab_min_share=float(np.mean(lab_y == mino)),
        n_query=n,
    )
    if C == 2:
        TP, FN = cm[mino, mino] + 0.5, cm[mino, maj] + 0.5
        FP, TN = cm[maj, mino] + 0.5, cm[maj, maj] + 0.5
        sens, spec = TP / (TP + FN), TN / (TN + FP)
        out["log_lrp"] = float(np.log(sens / (1 - spec)))
        out["neg_log_lrn"] = float(-np.log((1 - sens) / spec))
        # raw (uncorrected) versions for a sensitivity check
        tp_, fn_, fp_, tn_ = (cm[mino, mino], cm[mino, maj], cm[maj, mino], cm[maj, maj])
        out["raw_sens"] = tp_ / max(tp_ + fn_, 1)
        out["raw_spec"] = tn_ / max(tn_ + fp_, 1)
    return out


def maj_min(counts: np.ndarray) -> tuple[int, int]:
    """Majority = first argmax; minority = last argmin (distinct on ties)."""
    maj = int(np.argmax(counts))
    mino = len(counts) - 1 - int(np.argmin(counts[::-1]))
    return maj, mino


# ------------------------------------------------------------------ T1'
def t1_job(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    did = None
    key = d.dataset.iloc[0]
    did = DIDS[key]
    data = load(did, CACHE)
    y = data.y.astype(np.int64)
    C = int(y.max()) + 1
    counts = np.bincount(y, minlength=C)
    n = len(y)
    perm = t1_perm(n, did)
    s = min(500, n // 5)
    e = min(n, 2000)
    pool_idx = perm[s:]
    pool_counts = np.bincount(y[pool_idx], minlength=C)
    maj, mino = maj_min(pool_counts)
    pool_set = set(pool_idx.tolist())
    rows = []
    for r in d.itertuples(index=False):
        lab = np.asarray(r.labeled, dtype=np.int64)
        q = np.asarray(r.query, dtype=np.int64)
        P = np.asarray(r.proba, dtype=np.float64).reshape(len(q), C)
        # integrity checks
        exp_q = np.setdiff1d(perm[:e], lab)
        ok_query = np.array_equal(np.sort(q), np.sort(exp_q))
        ok_pool = all(int(i) in pool_set for i in lab)
        seen_ok = np.array_equal(np.asarray(r.seen), np.unique(y[lab]))
        rowsum = P.sum(axis=1)
        f = fit_features(y[q], P, y[lab], C, maj, mino)
        f.update(
            dataset=key, did=did, learner=r.learner, strategy=r.strategy, m=int(r.m),
            rep=int(r.rep), sel_seed=int(r.sel_seed), budget=int(r.budget), C=C,
            st_macro_f1=r.macro_f1, st_accuracy=r.accuracy,
            st_balanced_accuracy=r.balanced_accuracy, st_log_loss=r.log_loss,
            st_auc=r.auc, ok_query=ok_query, ok_pool=ok_pool, seen_ok=seen_ok,
            rowsum_maxdev=float(np.max(np.abs(rowsum - 1))),
            unseen_mass=float(P[:, np.setdiff1d(np.arange(C), np.unique(y[lab]))].sum())
            if len(np.unique(y[lab])) < C else 0.0,
            n=n, pool_n=len(pool_idx), stop_n=s, panel_n=e,
            ir_full=float(counts.max() / counts.min()),
            ir_pool=float(pool_counts.max() / pool_counts.min()),
            tv_full=float(0.5 * np.abs(counts / n - 1 / C).sum()),
            tv_pool=float(0.5 * np.abs(pool_counts / pool_counts.sum() - 1 / C).sum()),
            pool_counts=pool_counts.tolist(),
            labeled_first=lab.tolist() if int(r.m) == 1 else None,
        )
        rows.append(f)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ DEV
def dev_job(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    did = int(d.dataset.iloc[0])
    y = load(did, CACHE).y.astype(np.int64)
    C = int(y.max()) + 1
    counts = np.bincount(y, minlength=C)
    maj, mino = maj_min(counts)
    n = len(y)
    rows = []
    d = d[d.strategy.isin(["random", "stratified"])]
    for r in d.itertuples(index=False):
        lab = np.asarray(r.labeled, dtype=np.int64)
        P = np.asarray(r.proba, dtype=np.float64).reshape(n, C)
        q = np.setdiff1d(np.arange(n), lab)
        f = fit_features(y[q], P[q], y[lab], C, maj, mino)
        f.update(
            dataset=str(did), learner=r.learner, strategy=r.strategy, m=int(r.mult),
            rep=int(r.rep), C=C, st_macro_f1=r.macro_f1, st_accuracy=r.accuracy,
            st_balanced_accuracy=r.balanced_accuracy, st_log_loss=r.log_loss,
            st_auc=r.auc, n=n,
            tv_full=float(0.5 * np.abs(counts / n - 1 / C).sum()),
            ir_full=float(counts.max() / counts.min()),
        )
        rows.append(f)
    return pd.DataFrame(rows)


pool_yaml = yaml.safe_load((ROOT / "data" / "pool_t1prime.yaml").read_text())
DIDS = {(f"openml_{d['did']}" if d["did"] >= 0 else f"pmlb_{-d['did']}"): d["did"]
        for d in pool_yaml["datasets"]}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    workers = int(os.environ.get("VWORKERS", "4"))
    t1_files = sorted(T1.glob("*_fm*.parquet"))
    t1_files = [p for p in t1_files if "partial" not in p.name]
    dev_files = sorted(DEV.glob("*_fm*.parquet"))
    print(len(t1_files), "T1' files;", len(dev_files), "DEV files", flush=True)
    with Pool(workers) as pool:
        t1 = pd.concat(pool.map(t1_job, t1_files), ignore_index=True)
        t1.to_parquet(OUT / "fits_t1.parquet", index=False)
        print("T1' fits:", len(t1), flush=True)
        dev = pd.concat(pool.map(dev_job, dev_files), ignore_index=True)
        dev.to_parquet(OUT / "fits_dev.parquet", index=False)
        print("DEV fits:", len(dev), flush=True)
