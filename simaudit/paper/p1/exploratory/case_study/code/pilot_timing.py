"""Timing pilot (PREREG Sec. 7): TabEBM per-class generation time on synthetic data only."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import cs_common as cc  # noqa: E402
import gen  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402

cc.OUT.mkdir(exist_ok=True)
rows = ["D,n_k,num_samples,seconds_per_class,seconds_per_step"]
for D, n_k, s in [(9, 8, 250), (19, 3, 20), (40, 4, 45), (77, 2, 60), (76, 4, 100), (77, 5, 62)]:
    X, y = make_classification(n_samples=n_k * 2, n_features=D, n_informative=min(D, 5), random_state=0)
    X = ((X - X.mean(0)) / (X.std(0) + 1e-9)).astype(np.float32)
    y = np.zeros(len(X), int)
    t = time.perf_counter()
    g, _ = gen.tabebm_generate(X[:n_k], y[:n_k], {0: s}, seed=1)
    dt = time.perf_counter() - t
    assert g.shape == (s, D) and np.isfinite(g).all()
    rows.append(f"{D},{n_k},{s},{dt:.2f},{dt/200:.4f}")
    print(rows[-1], flush=True)
(cc.OUT / "pilot_timing.csv").write_text("\n".join(rows) + "\n")
