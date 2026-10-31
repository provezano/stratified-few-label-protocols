"""Stage B (P1 venv, tabpfn 9.2.0): TabPFN v2 downstream on the stage-A arrays.

Usage: python -I code/run_stage_b.py --protocol S|R [--workers 6]
Reads only this study's own out/arrays/*.npz (allow_pickle=False). Resumable.
"""

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
ARR = OUT / "arrays"
B_DIR = OUT / "stage_b"


def metrics(y_true, y_pred, C):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return dict(
        ba=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=list(range(C)), average="macro", zero_division=0)),
        acc=float(accuracy_score(y_true, y_pred)),
    )


def run(path: str) -> str:
    import numpy as np
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    torch.set_num_threads(1)
    p = Path(path)
    out = B_DIR / (p.stem + ".json")
    if out.exists():
        return "skip"
    d = np.load(p, allow_pickle=False)
    C, seed = int(d["C"]), int(d["seed"])
    arms = {
        "baseline": (d["X_tr"], d["y_tr"]),
        "smote": (np.vstack([d["X_tr"], d["Xs"]]), np.concatenate([d["y_tr"], d["ys"]])),
        "tabebm": (np.vstack([d["X_tr"], d["Xg"]]), np.concatenate([d["y_tr"], d["yg"]])),
    }
    rows = []
    for arm, (Xf, yf) in arms.items():
        seen = np.unique(yf)
        if len(seen) == 1:
            pred = np.full(len(d["y_te"]), seen[0])
        else:
            m = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=seed % (2**31))
            m.fit(Xf, np.searchsorted(seen, yf))
            pred = seen[np.asarray(m.predict(d["X_te"]), int)]
        rows.append(dict(arm=arm, predictor="tabpfn", **metrics(d["y_te"], pred, C)))
    out.write_text(json.dumps(dict(task=p.stem, rows=rows)))
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    B_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(str(f) for f in ARR.glob(f"*_{a.protocol}_*.npz"))
    fails = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run, f): f for f in files}
        for f in as_completed(futs):
            try:
                print(Path(futs[f]).stem, f.result(), flush=True)
            except Exception:
                fails += 1
                print(Path(futs[f]).stem, "FAILED", traceback.format_exc(), flush=True)
    (OUT / f"stage_b_{a.protocol}.done").write_text(f"fails={fails}\n")


if __name__ == "__main__":
    main()
