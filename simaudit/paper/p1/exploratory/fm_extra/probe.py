"""Timing and unseen-class probe for one extra FM (exploratory, DEV, read-only data).

Usage (from simaudit/, in the model's own venv, one thread, under nice):
    OMP_NUM_THREADS=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
      nice -n 10 .venv_fm_extra/bin/python paper/p1/exploratory/fm_extra/probe.py fm6

(a) Timing: one fit+predict per context size B in {2, 10, 50, 100, 300} (random rows
    with every class, labels 0..K-1), predicting the full table (Study 7 predicts all n
    rows and scores the complement). Tables: 46562 (n=1000, d=33, C=2, largest DEV
    table) and 40496 (n=500, C=10, the most classes on DEV). The first call (weight
    load, warm-up) is timed separately.
(b) Unseen classes (mirror of E8): the context has labels {0, 2} of a 3-class table
    (47155 waveform), no row of class 1. Records the native output shape and, where
    the model's head has more output units than context classes, the probability the
    raw head gives to the unused unit 1 (read through the model's own logits path).
Writes out/probe_{fm}.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import adapters as ad  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.settings import settings  # noqa: E402

SIZES = (2, 10, 50, 100, 300)
TABLES = (46562, 40496)


def covered_sample(y: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """``size`` rows (at least one per class when size >= C)."""
    C = int(y.max()) + 1
    first = [rng.choice(np.flatnonzero(y == c)) for c in range(min(C, size))]
    rest = rng.choice(np.setdiff1d(np.arange(len(y)), first), size - len(first), replace=False)
    return np.sort(np.r_[first, rest]).astype(np.int64)


def timing(fm: str) -> list[dict]:
    _, predict = ad.PREDICTORS[fm]
    rows = []
    for did in TABLES:
        d = load(did, settings.data_cache_dir)
        rng = np.random.default_rng([did, 6])
        for b in SIZES:
            lab = covered_sample(d.y, b, rng)
            seen, y_fit = np.unique(d.y[lab], return_inverse=True)
            if len(seen) < 2:
                continue
            t = time.perf_counter()
            p = predict(d.X[lab], y_fit.astype(np.int64), d.X, 1)
            sec = time.perf_counter() - t
            first = not rows
            rows.append(dict(dataset=did, n=len(d.y), d=d.X.shape[1], C=int(d.y.max()) + 1, B=b,
                             K=len(seen), seconds=round(sec, 3), first_call=first,
                             rowsum_ok=bool(np.allclose(p.sum(1), 1, atol=1e-4))))
            print(rows[-1], flush=True)
            if first:  # repeat the first cell warm
                t = time.perf_counter()
                predict(d.X[lab], y_fit.astype(np.int64), d.X, 1)
                rows.append(dict(rows[-1], seconds=round(time.perf_counter() - t, 3), first_call=False))
                print(rows[-1], flush=True)
    return rows


def unseen(fm: str) -> dict:
    d = load(47155, settings.data_cache_dir)
    rng = np.random.default_rng(8)
    lab = np.r_[rng.choice(np.flatnonzero(d.y == 0), 10, replace=False),
                rng.choice(np.flatnonzero(d.y == 2), 10, replace=False)]
    out: dict = dict(table=47155, context_labels=[0, 2], missing_class=1)
    # 1. Raw labels {0, 2} passed straight to the package API (no re-encoding).
    raw_y = d.y[lab].astype(np.int64)
    try:
        p = ad.PREDICTORS[fm][1](d.X[lab], raw_y, d.X, 1)
        out["raw_labels"] = dict(ok=True, shape=list(p.shape))
    except Exception as exc:  # noqa: BLE001 - documenting the failure mode
        out["raw_labels"] = dict(ok=False, error=repr(exc)[:300])
    # 2. Head inspection: output units vs context classes, mass on the unused unit.
    out.update(head_probe(fm, d.X[lab], raw_y, d.X, d.y))
    return out


def head_probe(fm: str, X_fit, y_fit, X, y) -> dict:
    """Probability of the unused unit 1 when the head is read beyond the seen classes."""
    if fm == "fm6":
        m = ad._tabdpt()
        m.fit(X_fit, y_fit)  # labels used as unit indices: units 0 and 2 carry the context
        m.num_classes = 3  # read three units instead of len(unique(y)) = 2
        p = m.ensemble_predict_proba(X, n_ensembles=8, seed=1, permute_classes=False)
        return dict(head_units=int(m.max_num_classes), forced_C=3, **_mass(p, y))
    if fm == "fm7":
        from autogluon.tabular.models.mitra import sklearn_interface as si

        clf = ad._mitra_classifier(si)(fine_tune=False, device="cpu", n_estimators=1, seed=1,
                                 hf_model=ad._mitra_dir(), verbose=False)
        clf.fit(X_fit, y_fit, X_val=X_fit[:1], y_val=y_fit[:1])
        logits = clf.trainers[0].predict(clf.X, clf.y, X)[..., :3]
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return dict(head_units=10, forced_C=3, **_mass(p, y))
    return dict(head_units=None, note="LabelEncoder over context labels; no raw head access in the public API")


def _mass(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.asarray(p, float)
    return dict(mean_p_unused=float(p[:, 1].mean()),
                mean_p_unused_on_true_class1=float(p[y == 1, 1].mean()),
                share_argmax_unused=float((p.argmax(1) == 1).mean()))


def main() -> None:
    fm = sys.argv[1]
    res = dict(fm=fm, name=ad.PREDICTORS[fm][0], timing=timing(fm), unseen=unseen(fm))
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / f"probe_{fm}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["unseen"], indent=2))


if __name__ == "__main__":
    main()
