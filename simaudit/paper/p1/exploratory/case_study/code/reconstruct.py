"""Reconstruction check (PREREG Sec. 2): (N, D, C) after the paper's class filter vs Table 3."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import cs_common as cc  # noqa: E402

cc.OUT.mkdir(exist_ok=True)
rows = ["did,name,N,D,C,paper_N,paper_D,paper_C,n_missing_cells,min_class,max_class,match"]
for did, (pn, pd_, pc) in cc.PAPER_SHAPE.items():
    X, y = cc.load_table(did)
    n, d = X.shape
    c = len(np.unique(y))
    cnt = np.bincount(y)
    ok = (n, d, c) == (pn, pd_, pc)
    rows.append(f"{did},{cc.NAMES[did]},{n},{d},{c},{pn},{pd_},{pc},{int(np.isnan(X).sum())},{cnt.min()},{cnt.max()},{ok}")
(cc.OUT / "reconstruction.csv").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
