"""E17 (EXPLORATORY, not pre-registered): full metric panel for the protocol gap.

Stratified minus random labelled sets, scored with ~35 classification metrics from the
stored per-query probabilities (no new fits):
- DEV: results/study7/a/{did}_{fm}.parquet (Study 7 Part (a), 20 tables x 4 FMs);
  proba = full n x C matrix, scored on the complement of the labelled rows.
- T1':  results/study9/t1prime/{key}_{fm}.parquet (Study 9, 52 tables x 4 FMs);
  proba = query x C, query rows in original order (`query`), positions in the permuted
  order (`query_pos`); y of the query = y_orig[query].
True labels come from the project loader (read-only). Before anything new is computed,
the stored macro_f1, accuracy, balanced_accuracy, log_loss and auc of every
stratified/random row are recomputed with the frozen `analysis.study7.metric_set`
AND with this script's own fast implementation; the max abs errors are written to
e17_reproduction.csv. Every panel metric is checked against scikit-learn 1.9.1 on a
seeded sample of draws (e17_sklearn_check.csv).

Missing-class rule (L0, paper Sec. 3 and `selection.strategies.expand_proba`): the FM
is fitted on the seen classes only; an unseen class has probability 0; a single-class
context predicts that class with probability 1. Metric conventions:
- decision metrics: argmax over all C columns; macro averages over ALL C classes with
  zero_division = 0 (an unseen class has F1 = precision = recall = Jaccard = 0);
  balanced accuracy = mean recall over classes present in U (frozen convention).
- top-2 accuracy (C >= 3): the top min(2, |S|) seen classes (an unseen class is never
  ranked, as under L0); ties at the threshold count as hits.
- AUC/AP "all": every class present in U with both outcomes; an unseen class has a
  constant score, so AUC 0.5 and AP = its prevalence; a single-class context -> 0.5.
- AUC/AP "seen": classes in the context only (OvR on all rows of U, as the stored
  pre-registered `auc`); NaN when only one class is seen (dropped from the table mean,
  as in the paper).
- log loss: clipped at 1e-6 and renormalised (frozen `_log_loss`); Brier = mean over
  rows of sum_k (p_k - 1[y=k])^2 (range [0, 2]); D2 = 1 - loss / loss(null), null =
  the class shares of U (sklearn's d2 null model), with the same clipped log loss.
- calibration: top-label ECE and classwise ECE (mean over all C classes), 15 equal-width
  bins (bin = ceil(15 p) - 1, p = 0 in bin 0); CITL = sum_k |mean p_k - share_k(U)|;
  KL(share(U) || mean p), mean p clipped at 1e-6 and renormalised.
- binary likelihood ratios: positive class = pool minority; Haldane +0.5 on each cell
  so that every draw is defined; reported as log LR+ and log LR-.
Orientation: gap = stratified - random for "higher is better" metrics and
random - stratified for losses/calibration errors, so a positive gap always means
"stratified looks better".

Run from simaudit/:
  OMP_NUM_THREADS=1 nice -n 15 .venv/bin/python paper/p1/exploratory/e17_metrics/e17_metrics.py
Writes paper/p1/exploratory/e17_metrics/*.csv, e17_draws.parquet (per-draw cache; delete
to rebuild), e17_metric_map.{pdf,png}, and paper/p1/exploratory/out/e17_numbers.tex.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common as cm  # noqa: E402
from simaudit.analysis.report import holm  # noqa: E402
from simaudit.analysis.study7 import complement_metrics, metric_set  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.runner import study9 as r9  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

OUTNUM = HERE.parent / "out"
S9 = settings.results_dir / "study9" / "t1prime"
WORKERS = int(os.environ.get("E17_WORKERS", "3"))
SEED = 20261027
CLIP = 1e-6
NBINS = 15
N_BOOT = 10_000
FMS = cm.FMS

# metric -> (family, higher_is_better, label)
METRICS: dict[str, tuple[str, bool, str]] = {
    "accuracy": ("F1 prevalence-weighted", True, "accuracy"),
    "top2_accuracy": ("F1 prevalence-weighted", True, "top-2 accuracy (C>=3)"),
    "weighted_f1": ("F1 prevalence-weighted", True, "weighted F1"),
    "kappa": ("F2 chance-corrected", True, "Cohen's kappa"),
    "mcc": ("F2 chance-corrected", True, "MCC"),
    "balanced_accuracy": ("F3 class-averaged", True, "balanced accuracy"),
    "macro_f1": ("F3 class-averaged", True, "macro-F1"),
    "macro_f05": ("F3 class-averaged", True, "macro F0.5"),
    "macro_f2": ("F3 class-averaged", True, "macro F2"),
    "macro_precision": ("F3 class-averaged", True, "macro precision"),
    "macro_recall": ("F3 class-averaged", True, "macro recall"),
    "macro_jaccard": ("F3 class-averaged", True, "macro Jaccard"),
    "auc_ovr_macro_all": ("F4 ranking (all classes)", True, "AUC OvR macro"),
    "auc_ovr_weighted_all": ("F4 ranking (all classes)", True, "AUC OvR weighted"),
    "auc_ovo_macro_all": ("F4 ranking (all classes)", True, "AUC OvO macro"),
    "ap_macro_all": ("F4 ranking (all classes)", True, "AP macro"),
    "auc_ovr_macro_seen": ("F5 ranking (seen classes)", True, "AUC OvR macro"),
    "auc_ovr_weighted_seen": ("F5 ranking (seen classes)", True, "AUC OvR weighted"),
    "auc_ovo_macro_seen": ("F5 ranking (seen classes)", True, "AUC OvO macro"),
    "ap_macro_seen": ("F5 ranking (seen classes)", True, "AP macro"),
    "log_loss": ("F6 proper scoring", False, "log loss"),
    "brier": ("F6 proper scoring", False, "Brier"),
    "d2_log_loss": ("F6 proper scoring", True, "D2 log loss"),
    "d2_brier": ("F6 proper scoring", True, "D2 Brier"),
    "ece": ("F7 calibration", False, "ECE (top-label)"),
    "ece_classwise": ("F7 calibration", False, "classwise ECE"),
    "citl_l1": ("F7 calibration", False, "CITL (L1)"),
    "kl_prior": ("F7 calibration", False, "KL(test prior || mean p)"),
    "log_lr_pos": ("F8 binary LR", True, "log LR+"),
    "log_lr_neg": ("F8 binary LR", False, "log LR-"),
}
FAMILIES = list(dict.fromkeys(f for f, _, _ in METRICS.values()))
EXTRA = ["citl_minority", "n_seen"]  # descriptive only


# ============================================================== metric primitives
def _auc(score: np.ndarray, pos: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties (= sklearn roc_auc_score)."""
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _ap(score: np.ndarray, pos: np.ndarray) -> float:
    """Average precision as sklearn: sum over distinct thresholds of dRecall * precision."""
    n1 = int(pos.sum())
    if n1 == 0:
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    s, p = score[order], pos[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]  # last index of each tie block
    tp = np.cumsum(p)[last]
    fp = (last + 1) - tp
    prec = tp / (tp + fp)
    rec = tp / n1
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def _ece(conf: np.ndarray, hit: np.ndarray) -> float:
    b = np.clip(np.ceil(conf * NBINS).astype(int) - 1, 0, NBINS - 1)
    n = len(conf)
    cnt = np.bincount(b, minlength=NBINS)
    sc = np.bincount(b, weights=conf, minlength=NBINS)
    sh = np.bincount(b, weights=hit.astype(float), minlength=NBINS)
    return float(np.abs(sc - sh).sum() / n)


def _ranking(y: np.ndarray, P: np.ndarray, classes: np.ndarray, rows_all: bool = True):
    """OvR macro/weighted AUC, OvO macro AUC (Hand-Till) and macro AP over ``classes``."""
    aucs, w, aps = [], [], []
    for k in classes:
        pos = y == k
        a = _auc(P[:, k], pos)
        if np.isfinite(a):
            aucs.append(a)
            w.append(pos.sum())
            aps.append(_ap(P[:, k], pos))
    ovo = []
    cl = list(classes)
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            a, b = cl[i], cl[j]
            sel = (y == a) | (y == b)
            if (y == a).sum() == 0 or (y == b).sum() == 0:
                continue
            ys, Ps = y[sel], P[sel]
            ovo.append(0.5 * (_auc(Ps[:, a], ys == a) + _auc(Ps[:, b], ys == b)))
    if not aucs:
        return np.nan, np.nan, np.nan, np.nan
    w = np.asarray(w, float)
    return (float(np.mean(aucs)), float(np.dot(aucs, w) / w.sum()),
            float(np.mean(ovo)) if ovo else np.nan, float(np.mean(aps)))


def panel(y: np.ndarray, P: np.ndarray, seen: np.ndarray, C: int, pos_class: int) -> dict[str, float]:
    """Every E17 metric of one draw (y: true labels of U; P: |U| x C probabilities)."""
    n = len(y)
    P = np.asarray(P, np.float64)
    pred = P.argmax(1)
    cmx = np.bincount(y * C + pred, minlength=C * C).reshape(C, C).astype(float)
    tp = np.diag(cmx)
    t = cmx.sum(1)
    p = cmx.sum(0)
    fn, fp = t - tp, p - tp
    present = t > 0

    def safe(a, b):
        return np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)

    prec, rec = safe(tp, p), safe(tp, t)

    def fbeta(beta):
        b2 = beta * beta
        return safe((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp)

    f1 = fbeta(1.0)
    acc = tp.sum() / n
    out: dict[str, float] = dict(
        accuracy=acc,
        weighted_f1=float((f1 * t).sum() / t.sum()),
        balanced_accuracy=float(rec[present].mean()),
        macro_f1=float(f1.mean()),
        macro_f05=float(fbeta(0.5).mean()),
        macro_f2=float(fbeta(2.0).mean()),
        macro_precision=float(prec.mean()),
        macro_recall=float(rec.mean()),
        macro_jaccard=float(safe(tp, tp + fp + fn).mean()),
    )
    pe = float((t * p).sum()) / n**2
    out["kappa"] = float((acc - pe) / (1 - pe)) if pe < 1 else 0.0
    cov_ytyp = tp.sum() * n - (t * p).sum()
    den = np.sqrt((n**2 - (p**2).sum()) * (n**2 - (t**2).sum()))
    out["mcc"] = float(cov_ytyp / den) if den > 0 else 0.0
    # top-2 among seen classes
    if C >= 3:
        S = np.zeros(C, bool)
        S[seen] = True
        k = min(2, int(S.sum()))
        py = P[np.arange(n), y]
        higher = ((P > py[:, None]) & S[None, :]).sum(1)
        out["top2_accuracy"] = float(np.mean(S[y] & (higher < k)))
    else:
        out["top2_accuracy"] = np.nan
    # ranking
    allc = np.nonzero(present)[0]
    (out["auc_ovr_macro_all"], out["auc_ovr_weighted_all"], out["auc_ovo_macro_all"],
     out["ap_macro_all"]) = _ranking(y, P, allc)
    sc = np.array([c for c in seen if present[c]], dtype=int)
    if len(seen) > 1:
        (out["auc_ovr_macro_seen"], out["auc_ovr_weighted_seen"], out["auc_ovo_macro_seen"],
         out["ap_macro_seen"]) = _ranking(y, P, sc)
    else:
        out.update(auc_ovr_macro_seen=np.nan, auc_ovr_weighted_seen=np.nan,
                   auc_ovo_macro_seen=np.nan, ap_macro_seen=np.nan)
    # proper scoring rules
    Q = np.clip(P, CLIP, 1.0)
    Q = Q / Q.sum(1, keepdims=True)
    ll = float(-np.mean(np.log(Q[np.arange(n), y])))
    share = t / n
    ll0 = float(-np.sum(share[present] * np.log(share[present])))
    Y = np.zeros_like(P)
    Y[np.arange(n), y] = 1.0
    br = float(((P - Y) ** 2).sum(1).mean())
    br0 = float(1.0 - (share**2).sum())
    out.update(log_loss=ll, brier=br, d2_log_loss=1 - ll / ll0, d2_brier=1 - br / br0)
    # calibration
    out["ece"] = _ece(P.max(1), pred == y)
    out["ece_classwise"] = float(np.mean([_ece(P[:, k], y == k) for k in range(C)]))
    mp = P.mean(0)
    out["citl_l1"] = float(np.abs(mp - share).sum())
    mq = np.clip(mp, CLIP, None)
    mq = mq / mq.sum()
    out["kl_prior"] = float(np.sum(share[present] * np.log(share[present] / mq[present])))
    out["citl_minority"] = float(mp[pos_class] - share[pos_class])
    out["n_seen"] = float(len(seen))
    # binary likelihood ratios (Haldane)
    if C == 2:
        TP, FN = cmx[pos_class, pos_class], cmx[pos_class].sum() - cmx[pos_class, pos_class]
        o = 1 - pos_class
        TN, FP = cmx[o, o], cmx[o].sum() - cmx[o, o]
        sens = (TP + 0.5) / (TP + FN + 1)
        spec = (TN + 0.5) / (TN + FP + 1)
        out["log_lr_pos"] = float(np.log(sens / (1 - spec)))
        out["log_lr_neg"] = float(np.log((1 - sens) / spec))
    else:
        out["log_lr_pos"] = out["log_lr_neg"] = np.nan
    return out


# ============================================================== data access
def t1_pool() -> dict[str, int]:
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())
    return {r9.dataset_key(int(d["did"])): int(d["did"]) for d in pool["datasets"]}


def table_traits(y_pool: np.ndarray) -> dict[str, float]:
    tr = cm.pool_traits(y_pool)
    return dict(C=tr["C"], n=tr["n"], IR=tr["IR"], pi_min=tr["pi_min"], tv=tr["tv"],
                minority=int(np.argmin(np.bincount(y_pool))))


STORED = ("macro_f1", "accuracy", "balanced_accuracy", "log_loss", "auc")


def job(args: tuple[str, str, int, str]) -> tuple[list[dict], dict]:
    """Panel of every stratified/random draw of one (set, table, FM) + reproduction errors."""
    set_name, key, did, fm = args
    if set_name == "dev":
        y = cm.dev_data(did).y.astype(np.int64)
        df = pd.read_parquet(cm.S7 / "a" / f"{did}_{fm}.parquet")
        df = df.rename(columns={"mult": "m"})
        tr = table_traits(y)
    else:
        data = load(did, settings.data_cache_dir)
        y = data.y.astype(np.int64)
        df = pd.read_parquet(S9 / f"{key}_{fm}.parquet")
        perm = r9.row_permutation(len(y), did, settings.base_seed)
        s, _ = r9.layout(len(y))
        tr = table_traits(y[perm][s:])
    df = df[df.strategy.isin(["stratified", "random"])].reset_index(drop=True)
    C = int(df.n_classes.iloc[0])
    rows, err_frozen, err_own = [], {k: 0.0 for k in STORED}, {k: 0.0 for k in STORED}
    for r in df.itertuples():
        lab = np.asarray(r.labeled, np.int64)
        if set_name == "dev":
            Pfull = np.asarray(r.proba, np.float64).reshape(len(y), C)
            rest = np.setdiff1d(np.arange(len(y)), lab)
            yq, P = y[rest], Pfull[rest]
            fro = complement_metrics(y, Pfull, lab)
        else:
            q = np.asarray(r.query, np.int64)
            yq = y[q]
            P = np.asarray(r.proba, np.float64).reshape(len(q), C)
            fro = metric_set(yq, P, np.arange(C, dtype=np.int64), np.asarray(r.seen, np.int64))
        seen = np.unique(y[lab])
        assert np.array_equal(seen, np.asarray(r.seen)), "seen classes do not match"
        pm = panel(yq, P, seen, C, tr["minority"])
        own = dict(macro_f1=pm["macro_f1"], accuracy=pm["accuracy"],
                   balanced_accuracy=pm["balanced_accuracy"], log_loss=pm["log_loss"],
                   auc=pm["auc_ovr_macro_seen"])
        for k in STORED:
            sv = float(getattr(r, k))
            for src, err in ((fro[k], err_frozen), (own[k], err_own)):
                if np.isnan(sv) and np.isnan(src):
                    continue
                e = abs(src - sv) if not (np.isnan(sv) or np.isnan(src)) else np.inf
                err[k] = max(err[k], e)
        rows.append(dict(set=set_name, table=key, learner=fm, strategy=r.strategy, m=int(r.m),
                         rep=int(r.rep), coverage=float(r.coverage), **pm))
    rep = dict(set=set_name, table=key, learner=fm, n_rows=len(df),
               **{f"frozen_{k}": v for k, v in err_frozen.items()},
               **{f"own_{k}": v for k, v in err_own.items()})
    return rows, {**rep, **{f"trait_{k}": v for k, v in tr.items()}}


# ============================================================== sklearn cross-check
def sklearn_check(n_per_set: int = 150) -> pd.DataFrame:
    """Compare the fast panel with sklearn on a seeded sample of draws."""
    from sklearn import metrics as sk

    rng = np.random.default_rng(SEED)
    res = []
    for set_name, jobs in (("dev", dev_jobs()), ("t1prime", t1_jobs())):
        for i in rng.choice(len(jobs), size=min(n_per_set, len(jobs)), replace=False):
            _, key, did, fm = jobs[i]
            if set_name == "dev":
                y = cm.dev_data(did).y.astype(np.int64)
                df = pd.read_parquet(cm.S7 / "a" / f"{did}_{fm}.parquet")
            else:
                y = load(did, settings.data_cache_dir).y.astype(np.int64)
                df = pd.read_parquet(S9 / f"{key}_{fm}.parquet")
            df = df[df.strategy.isin(["stratified", "random"])]
            r = df.iloc[int(rng.integers(len(df)))]
            C = int(r.n_classes)
            lab = np.asarray(r.labeled, np.int64)
            if set_name == "dev":
                rest = np.setdiff1d(np.arange(len(y)), lab)
                yq = y[rest]
                P = np.asarray(r.proba, np.float64).reshape(len(y), C)[rest]
            else:
                q = np.asarray(r.query, np.int64)
                yq = y[q]
                P = np.asarray(r.proba, np.float64).reshape(len(q), C)
            seen = np.unique(y[lab])
            pos = int(np.argmin(np.bincount(y)))
            mine = panel(yq, P, seen, C, pos)
            pred = P.argmax(1)
            L = list(range(C))
            Q = np.clip(P, CLIP, 1)
            Q = Q / Q.sum(1, keepdims=True)
            ref = dict(
                accuracy=sk.accuracy_score(yq, pred),
                micro_f1_vs_acc=sk.f1_score(yq, pred, labels=L, average="micro", zero_division=0),
                hamming_vs_acc=1 - sk.hamming_loss(yq, pred),
                weighted_f1=sk.f1_score(yq, pred, labels=L, average="weighted", zero_division=0),
                kappa=sk.cohen_kappa_score(yq, pred),
                mcc=sk.matthews_corrcoef(yq, pred),
                balanced_accuracy=sk.balanced_accuracy_score(yq, pred),
                macro_f1=sk.f1_score(yq, pred, labels=L, average="macro", zero_division=0),
                macro_f05=sk.fbeta_score(yq, pred, beta=0.5, labels=L, average="macro", zero_division=0),
                macro_f2=sk.fbeta_score(yq, pred, beta=2, labels=L, average="macro", zero_division=0),
                macro_precision=sk.precision_score(yq, pred, labels=L, average="macro", zero_division=0),
                macro_recall=sk.recall_score(yq, pred, labels=L, average="macro", zero_division=0),
                macro_jaccard=sk.jaccard_score(yq, pred, labels=L, average="macro", zero_division=0),
                log_loss=sk.log_loss(yq, Q, labels=L),
                brier=sk.brier_score_loss(yq, P, labels=L) * (2 if C == 2 else 1),
                d2_brier=sk.d2_brier_score(yq, P, labels=L),
            )
            present = np.unique(yq)
            if len(present) == C and len(seen) == C:
                if C == 2:
                    ref["auc_ovr_macro_all"] = sk.roc_auc_score(yq, P[:, 1])
                    ref["auc_ovr_macro_seen"] = ref["auc_ovr_macro_all"]
                else:
                    ref["auc_ovr_macro_all"] = sk.roc_auc_score(yq, P, multi_class="ovr", average="macro", labels=L)
                    ref["auc_ovr_weighted_all"] = sk.roc_auc_score(yq, P, multi_class="ovr", average="weighted", labels=L)
                    ref["auc_ovo_macro_all"] = sk.roc_auc_score(yq, P / P.sum(1, keepdims=True), multi_class="ovo", average="macro", labels=L)
                    ref["top2_accuracy"] = sk.top_k_accuracy_score(yq, P, k=2, labels=L)
                ref["ap_macro_all"] = sk.average_precision_score(np.eye(C)[yq], P, average="macro") if C > 2 else \
                    np.mean([sk.average_precision_score(yq == k, P[:, k]) for k in range(2)])
                ref["d2_log_loss"] = sk.d2_log_loss_score(yq, Q, labels=L)
            if C == 2 and len(seen) == 2:
                ref["auc_ovo_macro_all"] = sk.roc_auc_score(yq, P[:, 1])
            for k, v in ref.items():
                mk = "accuracy" if k.endswith("_vs_acc") else k
                if k in ("brier",) and C == 2:
                    pass
                res.append(dict(set=set_name, table=key, learner=fm, metric=k, C=C,
                                n_seen=len(seen), abs_err=abs(float(mine[mk]) - float(v))))
    return pd.DataFrame(res)


def dev_jobs() -> list[tuple[str, str, int, str]]:
    return [("dev", str(d), d, fm) for d in cm.dev_ids() for fm in FMS]


def t1_jobs() -> list[tuple[str, str, int, str]]:
    return [("t1prime", k, d, fm) for k, d in t1_pool().items() for fm in FMS]


# ============================================================== statistics
def rank_biserial(x: np.ndarray) -> float:
    x = x[x != 0]
    if len(x) == 0:
        return 0.0
    r = stats.rankdata(np.abs(x))
    return float((r[x > 0].sum() - r[x < 0].sum()) / r.sum())


def summarize(x: np.ndarray, seed: int) -> dict[str, float]:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    bm = x[idx].mean(1)
    rb = np.array([rank_biserial(x[i]) for i in idx[:2000]])
    p = float(stats.wilcoxon(x).pvalue) if np.any(x != 0) and len(x) > 1 else 1.0
    return dict(n=len(x), mean=float(x.mean()), ci_lo=float(np.quantile(bm, .025)),
                ci_hi=float(np.quantile(bm, .975)), median=float(np.median(x)),
                win=float(np.mean(x > 0)), rb=rank_biserial(x),
                rb_lo=float(np.quantile(rb, .025)), rb_hi=float(np.quantile(rb, .975)), p=p)


def cell_gaps(draws: pd.DataFrame) -> pd.DataFrame:
    """Oriented gap per (set, table, FM, m, metric): mean S - mean R (sign-flipped for losses)."""
    mets = list(METRICS)
    g = draws.groupby(["set", "table", "learner", "m", "strategy"])[mets].mean()
    S = g.xs("stratified", level="strategy")
    R = g.xs("random", level="strategy")
    S, R = S.align(R, join="inner")
    sign = np.array([1.0 if METRICS[k][1] else -1.0 for k in mets])
    gap = (S - R) * sign
    gap = gap.reset_index().melt(id_vars=["set", "table", "learner", "m"], var_name="metric", value_name="gap")
    return gap.dropna(subset=["gap"])


def tests(gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (st, met, lr, m), g in gaps.groupby(["set", "metric", "learner", "m"]):
        if len(g) < 5:
            continue
        rows.append(dict(set=st, family=METRICS[met][0], metric=met, learner=lr, m=m,
                         **summarize(g.gap.to_numpy(), SEED + int(m))))
    pooled = gaps.groupby(["set", "metric", "m", "table"]).gap.mean().reset_index()
    for (st, met, m), g in pooled.groupby(["set", "metric", "m"]):
        if len(g) < 5:
            continue
        rows.append(dict(set=st, family=METRICS[met][0], metric=met, learner="pooled", m=m,
                         **summarize(g.gap.to_numpy(), SEED + int(m))))
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, idx in out.groupby(["set", "family", out.learner == "pooled"]).groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].tolist())
    return out


def moderators(gaps: pd.DataFrame, traits: pd.DataFrame) -> pd.DataFrame:
    pooled = gaps.groupby(["set", "metric", "m", "table"]).gap.mean().reset_index()
    pooled = pooled.merge(traits, on=["set", "table"])
    rows = []
    for (st, met, m), g in pooled.groupby(["set", "metric", "m"]):
        if len(g) < 8:
            continue
        row = dict(set=st, family=METRICS[met][0], metric=met, m=m, n=len(g))
        for var in ("tv", "C"):
            if g[var].nunique() > 1:
                rho, lo, hi, p = cm.spearman_ci(g[var].to_numpy(), g.gap.to_numpy(), n_boot=2000, seed=SEED)
                row.update({f"rho_{var}": rho, f"rho_{var}_lo": lo, f"rho_{var}_hi": hi, f"p_{var}": p})
        b = g[g.C == 2].gap
        mc = g[g.C > 2].gap
        row.update(mean_binary=b.mean() if len(b) else np.nan, n_binary=len(b),
                   mean_multiclass=mc.mean() if len(mc) else np.nan, n_multiclass=len(mc))
        med = g.tv.median()
        row.update(mean_low_imb=g[g.tv <= med].gap.mean(), mean_high_imb=g[g.tv > med].gap.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def imbalance_halves(gaps: pd.DataFrame, traits: pd.DataFrame) -> pd.DataFrame:
    """Pooled-FM gap tests within the low/high half of pool imbalance (TV to uniform, median split per set)."""
    pooled = gaps.groupby(["set", "metric", "m", "table"]).gap.mean().reset_index().merge(traits, on=["set", "table"])
    med = traits.groupby("set").tv.median()
    pooled["half"] = np.where(pooled.tv > pooled.set.map(med), "high_imbalance", "low_imbalance")
    rows = []
    for (st, half, met, m), g in pooled.groupby(["set", "half", "metric", "m"]):
        if len(g) < 5:
            continue
        rows.append(dict(set=st, half=half, family=METRICS[met][0], metric=met, m=m,
                         **summarize(g.gap.to_numpy(), SEED + int(m))))
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, idx in out.groupby(["set", "half", "family"]).groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].tolist())
    return out


def flips(draws: pd.DataFrame) -> pd.DataFrame:
    """Share of FM pairs whose order differs between the stratified and random protocol.

    Noise baseline ("null"): the same statistic between two disjoint halves of the random
    draws (reps 0-4 vs 5-9, i.e. 5 reps each, like stratified), so that a flip rate can be
    read against the flips that pure resampling noise produces."""
    d = draws.copy()
    rnd = d[d.strategy == "random"]
    half_a = rnd[rnd.rep < 5].assign(strategy="stratified")
    half_b = rnd[(rnd.rep >= 5) & (rnd.rep < 10)].assign(strategy="random")
    real = _flips(d).assign(kind="protocol")
    null = _flips(pd.concat([half_a, half_b])).assign(kind="null_split_half")
    return pd.concat([real, null], ignore_index=True)


def flip_excess(fl: pd.DataFrame) -> pd.DataFrame:
    """Per table: pair-flip rate (mean over budgets) of the protocol minus the split-half null;
    Wilcoxon across tables, Holm over metrics within each set."""
    t = fl.groupby(["set", "metric", "table", "kind"]).pair_flip.mean().unstack("kind").dropna()
    rows = []
    for (st, met), g in t.groupby(level=["set", "metric"]):
        x = (g["protocol"] - g["null_split_half"]).to_numpy()
        rows.append(dict(set=st, family=METRICS[met][0], metric=met, n=len(x),
                         protocol=g["protocol"].mean(), null=g["null_split_half"].mean(), excess=x.mean(),
                         p=float(stats.wilcoxon(x).pvalue) if np.any(x != 0) else 1.0))
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, idx in out.groupby("set").groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].tolist())
    return out


def _flips(draws: pd.DataFrame) -> pd.DataFrame:
    mets = list(METRICS)
    g = draws.groupby(["set", "table", "m", "strategy", "learner"])[mets].mean()
    rows = []
    for (st, tab, m), cell in g.groupby(level=["set", "table", "m"]):
        cell = cell.droplevel(["set", "table", "m"])
        if "stratified" not in cell.index.get_level_values(0) or "random" not in cell.index.get_level_values(0):
            continue
        S, R = cell.xs("stratified"), cell.xs("random")
        S, R = S.align(R, join="inner")
        if len(S) < 4:
            continue
        for met in mets:
            s, r = S[met].to_numpy(), R[met].to_numpy()
            if np.isnan(s).any() or np.isnan(r).any():
                continue
            sign = 1 if METRICS[met][1] else -1
            disc, tot = 0, 0
            for i in range(4):
                for j in range(i + 1, 4):
                    a, b = np.sign(s[i] - s[j]), np.sign(r[i] - r[j])
                    if a == 0 or b == 0:
                        continue
                    tot += 1
                    disc += a != b
            if tot == 0:
                continue
            rows.append(dict(set=st, table=tab, m=m, metric=met, family=METRICS[met][0],
                             pair_flip=disc / tot,
                             top_flip=float(np.argmax(sign * s) != np.argmax(sign * r))))
    return pd.DataFrame(rows)


def correlation(gaps: pd.DataFrame, draws: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import adjusted_rand_score

    mets = [k for k in METRICS if not k.startswith("log_lr") and k != "top2_accuracy"]
    wide = gaps[gaps.metric.isin(mets)].pivot_table(index=["set", "table", "learner", "m"], columns="metric", values="gap")
    wide = wide[mets]
    corr_gap = wide.corr(method="spearman", min_periods=50)
    # draw level: random draws, centred within (table, FM, m) cells
    rd = draws[draws.strategy == "random"].copy()
    keys = ["set", "table", "learner", "m"]
    cen = rd[mets] - rd.groupby(keys)[mets].transform("mean")
    corr_draw = cen.corr(method="spearman", min_periods=50)
    fam = np.array([METRICS[k][0] for k in mets])
    out = []
    for name, C_ in (("gap", corr_gap), ("draw", corr_draw)):
        D = 1 - C_.fillna(0).to_numpy()
        np.fill_diagonal(D, 0)
        Z = linkage(squareform(np.clip((D + D.T) / 2, 0, 2), checks=False), "average")
        for k in (3, 4, len(set(fam))):
            lab = fcluster(Z, k, "maxclust")
            out.append(dict(level=name, k=k, ari_vs_families=adjusted_rand_score(fam, lab),
                            clusters="; ".join(
                                "{" + ", ".join(np.array(mets)[lab == c]) + "}" for c in np.unique(lab))))
    return corr_gap, corr_draw, pd.DataFrame(out)


# ============================================================== side analysis: selector clusters
def selector_clusters(key: str, did: int) -> list[dict]:
    """k-medoids (Study 9 greedy facility location) partitions vs classes; LUCoS if cached."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    from simaudit.graph.knn import zscore
    from simaudit.selection import lucos

    data = load(did, settings.data_cache_dir)
    perm = r9.row_permutation(len(data.y), did, settings.base_seed)
    X, y = data.X[perm], data.y[perm].astype(np.int64)
    s, _ = r9.layout(len(y))
    pool = np.arange(s, len(y), dtype=np.int64)
    C = int(y.max()) + 1
    sel = pd.read_parquet(S9 / "selections" / f"{key}.parquet")
    rows = []
    embeds = {"kmedoids_nested": zscore(X)}
    path = lucos._cache_dir() / f"{lucos.cache_key(X)}.npy"
    if path.exists():
        embeds["lucos_nested"] = np.load(path).astype(np.float32)
    for name, Z in embeds.items():
        st = sel[(sel.strategy == name) & (sel.rep == 0)]
        for m in (1, 3):
            r = st[st.m == m]
            if r.empty:
                continue
            L = np.asarray(r.labeled_pos.iloc[0], np.int64)
            # partition of the pool: nearest selected medoid in the selector's own space
            Zp = np.asarray(Z, np.float32)
            d2 = ((Zp[pool][:, None, :] - Zp[L][None, :, :]) ** 2).sum(-1) if len(pool) * len(L) < 4e6 else None
            if d2 is None:
                continue
            assign = d2.argmin(1)
            yp = y[pool]
            # majority-label purity of the medoid partition (label each cluster by its medoid's class)
            med_lab = y[L][assign]
            rows.append(dict(table=key, selector=name, m=m, C=C, B=len(L),
                             ami=adjusted_mutual_info_score(yp, assign),
                             ari=adjusted_rand_score(yp, assign),
                             medoid_label_acc=float(np.mean(med_lab == yp)),
                             coverage=len(np.unique(y[L])) / C))
    return rows


def selector_side(draws_t1: pd.DataFrame) -> pd.DataFrame:
    jobs = list(t1_pool().items())
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(selector_clusters, [k for k, _ in jobs], [d for _, d in jobs]))
    cl = pd.DataFrame([r for rs in res for r in rs])
    # selector gap vs random (stored macro-F1 and accuracy, mean over FMs)
    rows = []
    for key in t1_pool():
        df = pd.concat([pd.read_parquet(S9 / f"{key}_{fm}.parquet",
                                        columns=["strategy", "m", "rep", "macro_f1", "accuracy", "learner"])
                        for fm in FMS])
        g = df.groupby(["strategy", "m"])[["macro_f1", "accuracy"]].mean()
        for sel_name in ("kmedoids_nested", "lucos_nested"):
            for m in (1, 3):
                if (sel_name, m) in g.index and ("random", m) in g.index:
                    d = g.loc[(sel_name, m)] - g.loc[("random", m)]
                    rows.append(dict(table=key, selector=sel_name, m=m,
                                     gap_f1=d.macro_f1, gap_acc=d.accuracy))
    return cl.merge(pd.DataFrame(rows), on=["table", "selector", "m"], how="inner")


# ============================================================== figure
def figure(tst: pd.DataFrame, halves: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#c4352b", "#f0efec", "#2a6fb0"])
    mets = [k for k in METRICS]
    hi = halves[halves.half == "high_imbalance"].assign(learner="pooled")
    sets = (("dev", "DEV (Study 7)", [1, 3, 5, 10], tst),
            ("t1prime", "T1′ (Study 9)", [1, 2, 3, 5, 10, 20], tst),
            ("t1prime", "T1′, more imbalanced half", [1, 3, 10, 20], hi))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 9.2), gridspec_kw=dict(width_ratios=[4, 6, 4]), sharey=True)
    for ax, (st, title, ms, src) in zip(axes, sets):
        sub = src[(src.set == st) & (src.learner == "pooled")]
        M = np.full((len(mets), len(ms)), np.nan)
        A = np.full_like(M, np.nan)
        sig = np.zeros_like(M, bool)
        for i, met in enumerate(mets):
            for j, m in enumerate(ms):
                r = sub[(sub.metric == met) & (sub.m == m)]
                if len(r):
                    M[i, j], A[i, j], sig[i, j] = r.rb.iloc[0], r["median"].iloc[0], r.p_holm.iloc[0] < 0.05
        ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i in range(len(mets)):
            for j in range(len(ms)):
                if np.isfinite(A[i, j]):
                    v = A[i, j]
                    txt = f"{v:+.2f}" if abs(v) >= 0.005 else f"{v:+.3f}"
                    ax.text(j, i, txt + ("*" if sig[i, j] else ""), ha="center", va="center", fontsize=6.5,
                            color="white" if abs(M[i, j]) > 0.75 else "#1f1f1f")
        ax.set_xticks(range(len(ms)), [f"{m}C" for m in ms], fontsize=8)
        ax.set_title(f"{title}\nn = {int(sub.n.max())} tables", fontsize=9)
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        prev = None
        for i, met in enumerate(mets):
            f = METRICS[met][0]
            if prev is not None and f != prev:
                ax.axhline(i - 0.5, color="#fcfcfb", lw=2.5)
            prev = f
    labels = [f"{METRICS[k][2]}  [{METRICS[k][0].split()[0]}]" for k in mets]
    axes[0].set_yticks(range(len(mets)), labels, fontsize=7.5)
    fig.suptitle("E17 metric map (exploratory): stratified − random, oriented so that > 0 = stratified looks better\n"
                 "colour = rank-biserial effect across tables (FMs pooled); text = median gap across tables in metric units; * Holm p < 0.05 within family",
                 fontsize=8.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("rank-biserial (−1: random better, +1: stratified better)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(HERE / "e17_metric_map.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})  # no local timezone in the PDF
    fig.savefig(HERE / "e17_metric_map.png", dpi=170, bbox_inches="tight")
    plt.close(fig)



# ============================================================== LaTeX macros
_W = {1: "One", 2: "Two", 3: "Three", 5: "Five", 10: "Ten", 20: "Twenty", 30: "Thirty"}
_SET = {"dev": "Dev", "t1prime": "Tp"}
_TOK = {"f05": "FHalf", "f2": "FTwo", "f1": "FOne", "d2": "DTwo", "top2": "TopTwo", "l1": "LOne",
        "ovr": "Ovr", "ovo": "Ovo", "auc": "Auc", "ap": "Ap", "ece": "Ece", "citl": "Citl", "kl": "Kl",
        "lr": "Lr", "mcc": "Mcc"}


def _camel(name: str) -> str:
    return "".join(_TOK.get(t, t.capitalize()) for t in name.split("_"))


def _fmt(v: float, d: int = 3) -> str:
    if not np.isfinite(v):
        return "n/a"
    s = f"{v:+.{d}f}"
    return ("$+$" if s[0] == "+" else "$-$") + s[1:]


def numbers(tst, halves, mod, fl, clus, repro, srho, fx) -> None:
    L = ["% Generated by paper/p1/exploratory/e17_metrics/e17_metrics.py. Do not edit.",
         "% E17 (EXPLORATORY): full metric panel for the stratified - random protocol gap.",
         "% Gap oriented so that > 0 = stratified looks better; FMs pooled per table unless stated.",
         "% Source CSV (in paper/p1/exploratory/e17_metrics/) in a comment on each line."]

    def add(name, val, src):
        L.append(f"\\newcommand{{\\ExESeventeen{name}}}{{{val}}} % {src}")

    rep_cols = [c for c in repro if c.startswith(("frozen_", "own_"))]
    add("ReproMaxErr", f"{float(repro[rep_cols].replace(np.inf, np.nan).max().max()):.1e}", "e17_reproduction.csv (max over all S/R rows, 5 stored metrics)")
    add("ReproRows", f"{int(repro.n_rows.sum())}", "e17_reproduction.csv (n_rows summed)")
    for st in ("dev", "t1prime"):
        add(f"{_SET[st]}Tables", f"{repro[repro.set == st].table.nunique()}", "e17_reproduction.csv")
    pooled = tst[tst.learner == "pooled"]
    for r in pooled[pooled.m.isin([1, 3, 10])].itertuples():
        base = f"{_SET[r.set]}{_camel(r.metric)}M{_W[int(r.m)]}"
        src = f"e17_gap_tests.csv set={r.set} metric={r.metric} learner=pooled m={r.m}"
        d = 2 if r.metric in ("log_loss", "d2_log_loss", "kl_prior") else 3
        add(base, _fmt(r.mean, d), src + " (mean)")
        add(base + "Lo", _fmt(r.ci_lo, d), src + " (bootstrap 2.5%)")
        add(base + "Hi", _fmt(r.ci_hi, d), src + " (bootstrap 97.5%)")
        add(base + "Med", _fmt(r.median, d), src + " (median)")
        add(base + "Rb", _fmt(r.rb, 2), src + " (rank-biserial)")
        add(base + "Ph", f"{r.p_holm:.2g}", src + " (Holm within family)")
    hi = halves[(halves.half == "high_imbalance") & halves.m.isin([1, 3, 10, 20])]
    for r in hi.itertuples():
        base = f"HiImb{_SET[r.set]}{_camel(r.metric)}M{_W[int(r.m)]}"
        src = f"e17_imbalance_halves.csv set={r.set} half=high metric={r.metric} m={r.m}"
        add(base, _fmt(r.median, 3), src + " (median)")
        add(base + "Rb", _fmt(r.rb, 2), src + " (rank-biserial)")
        add(base + "Ph", f"{r.p_holm:.2g}", src + " (Holm within family and half)")
        add(base + "N", f"{int(r.n)}", src + " (tables)")
    for r in mod[mod.m.isin([1, 3, 10])].itertuples():
        add(f"Rho{_SET[r.set]}{_camel(r.metric)}M{_W[int(r.m)]}", _fmt(r.rho_tv, 2),
            f"e17_moderators.csv set={r.set} metric={r.metric} m={r.m} (Spearman gap vs TV imbalance)")
    fr = fl.groupby(["kind", "set", "metric"])[["pair_flip", "top_flip"]].mean().reset_index()
    for r in fr.itertuples():
        k = "Null" if r.kind != "protocol" else ""
        add(f"Flip{k}{_SET[r.set]}{_camel(r.metric)}", f"{r.pair_flip:.2f}",
            f"e17_flip_rates.csv kind={r.kind} set={r.set} metric={r.metric} (pair flip rate)")
        add(f"TopFlip{k}{_SET[r.set]}{_camel(r.metric)}", f"{r.top_flip:.2f}",
            f"e17_flip_rates.csv kind={r.kind} set={r.set} metric={r.metric} (top-1 change rate)")
    for r in clus.itertuples():
        if r.k in (3, len(FAMILIES) - 1):
            add(f"ClusterAri{'Gap' if r.level == 'gap' else 'Draw'}K{_W.get(int(r.k), 'Seven')}", f"{r.ari_vs_families:.2f}",
                f"e17_clusters.csv level={r.level} k={r.k}")
    for r in srho[(srho["var"] == "ami")].itertuples():
        sel = "Kmed" if r.selector.startswith("kmed") else "Lucos"
        tgt = "FOne" if r.target == "gap_f1" else "Acc"
        add(f"Sel{sel}AmiRho{tgt}M{_W[int(r.m)]}", _fmt(r.rho, 2), f"e17_selector_cluster_rho.csv selector={r.selector} m={r.m} var=ami target={r.target}")
        add(f"Sel{sel}AmiRho{tgt}M{_W[int(r.m)]}Lo", _fmt(r.lo, 2), "same row (bootstrap 2.5%)")
        add(f"Sel{sel}AmiRho{tgt}M{_W[int(r.m)]}Hi", _fmt(r.hi, 2), "same row (bootstrap 97.5%)")
        if r.target == "gap_f1":
            add(f"Sel{sel}AmiMedM{_W[int(r.m)]}", f"{r.median_var:.3f}", "same row (median AMI)")
    for r in fx.itertuples():
        add(f"FlipExcess{_SET[r.set]}{_camel(r.metric)}", _fmt(r.excess, 2),
            f"e17_flip_excess.csv set={r.set} metric={r.metric} (protocol - split-half null, mean over tables)")
        add(f"FlipExcess{_SET[r.set]}{_camel(r.metric)}Ph", f"{r.p_holm:.2g}", "same row (Holm over metrics within set)")
    OUTNUM.mkdir(exist_ok=True)
    (OUTNUM / "e17_numbers.tex").write_text("\n".join(L) + "\n")


# ============================================================== main
def main() -> None:
    warnings.filterwarnings("ignore")
    cache = HERE / "e17_draws.parquet"
    if cache.exists():
        draws = pd.read_parquet(cache)
        repro = pd.read_csv(HERE / "e17_reproduction.csv")
    else:
        jobs = dev_jobs() + t1_jobs()
        with ProcessPoolExecutor(WORKERS) as ex:
            res = list(ex.map(job, jobs, chunksize=1))
        draws = pd.DataFrame([r for rs, _ in res for r in rs])
        repro = pd.DataFrame([rep for _, rep in res])
        repro.to_csv(HERE / "e17_reproduction.csv", index=False)
        draws.to_parquet(cache, index=False)
        chk = sklearn_check()
        chk.to_csv(HERE / "e17_sklearn_check_draws.csv", index=False)
        chk.groupby(["metric"]).abs_err.agg(["count", "max"]).reset_index().to_csv(
            HERE / "e17_sklearn_check.csv", index=False)
    traits = (repro[["set", "table"] + [c for c in repro if c.startswith("trait_")]]
              .drop_duplicates(["set", "table"]).rename(columns=lambda c: c.replace("trait_", "")))
    traits.to_csv(HERE / "e17_table_traits.csv", index=False)
    gaps = cell_gaps(draws)
    gaps.to_csv(HERE / "e17_cell_gaps.csv", index=False)
    tst = tests(gaps)
    tst.to_csv(HERE / "e17_gap_tests.csv", index=False)
    mod = moderators(gaps, traits)
    mod.to_csv(HERE / "e17_moderators.csv", index=False)
    fl = flips(draws)
    fl.to_csv(HERE / "e17_flips_cells.csv", index=False)
    flsum = fl.groupby(["kind", "set", "family", "metric"])[["pair_flip", "top_flip"]].mean().reset_index()
    flm = fl.groupby(["kind", "set", "metric", "m"])[["pair_flip", "top_flip"]].mean().reset_index()
    flsum.to_csv(HERE / "e17_flip_rates.csv", index=False)
    flm.to_csv(HERE / "e17_flip_rates_by_m.csv", index=False)
    fx = flip_excess(fl)
    fx.to_csv(HERE / "e17_flip_excess.csv", index=False)
    cg, cd, clus = correlation(gaps, draws)
    cg.to_csv(HERE / "e17_corr_gaps.csv")
    cd.to_csv(HERE / "e17_corr_draws.csv")
    clus.to_csv(HERE / "e17_clusters.csv", index=False)
    # calibration-in-the-large of the minority class (descriptive)
    citl = (draws.groupby(["set", "table", "learner", "m", "strategy"]).citl_minority.mean()
            .unstack("strategy").reset_index())
    citl.to_csv(HERE / "e17_citl_minority.csv", index=False)
    side_path = HERE / "e17_selector_clusters.csv"
    if not side_path.exists():
        selector_side(draws).to_csv(side_path, index=False)
    halves = imbalance_halves(gaps, traits)
    halves.to_csv(HERE / "e17_imbalance_halves.csv", index=False)
    sel = pd.read_csv(side_path)
    srows = []
    for (sn, m), g in sel.groupby(["selector", "m"]):
        for var in ("ami", "ari", "medoid_label_acc", "coverage"):
            for tgt in ("gap_f1", "gap_acc"):
                rho, lo, hi_, p = cm.spearman_ci(g[var].to_numpy(), g[tgt].to_numpy(), n_boot=2000, seed=SEED)
                srows.append(dict(selector=sn, m=m, var=var, target=tgt, n=len(g), median_var=g[var].median(),
                                  rho=rho, lo=lo, hi=hi_, p=p))
    pd.DataFrame(srows).to_csv(HERE / "e17_selector_cluster_rho.csv", index=False)
    figure(tst, halves)
    numbers(tst, halves, mod, fl, clus, repro, pd.DataFrame(srows), fx)
    print("done")


if __name__ == "__main__":
    main()
