"""T1' subset and cost estimate for the dose arms F, S and N (EXPLORATORY).

Preregistration: paper/p1/exploratory/b_dose/PREREG_T1P_SUBSET.md.

Subset rule (metadata only; no Study 12 result is read):
- the 52 tables of data/pool_t1prime.yaml, split by C (binary: C = 2;
  multiclass: C > 2, C read from the cached label vector);
- within each group, tables sorted by key, then shuffled with
  default_rng([base_seed, 41, 1201]);
- a subset of size K takes the first round(K * 35/52) binary and the first
  K - round(K * 35/52) multiclass tables of these orders.
K is the largest value <= 16 whose estimated CPU time is <= the cap.

Cost model: per FM, OLS of log(mean DEV seconds per fit) on log(eval rows) and
log(d) over the 20 DEV tables of b_dose/out/raw; fits per table = distinct
feasible contexts of design.rep_cells on the table's labels (no fit).

Usage (from simaudit/): .venv/bin/python paper/p1/exploratory/b_dose/t1p_subset.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import design  # noqa: E402

from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

FMS = ("fm2", "fm1", "fm5", "fm3")
MULTS = (3, 5, 10)
REPS = 5
K_MAX = 16
CAP_CPU_H = 60.0
SUBSET_TAG = 1201
PANEL_MAX = 1000  # evaluation panel: first min(n, 1000) rows of the Study 12 permutation


def seed_did(did: int) -> int:
    """Non-negative seed key, as runner.study12.seed_did."""
    return did if did >= 0 else 2**40 + did


def labels(did: int) -> np.ndarray:
    z = np.load(settings.data_cache_dir / "npz" / f"{did}.npz", allow_pickle=False)
    return z["y"].astype(np.int64), int(z["X"].shape[1])


def registry() -> pd.DataFrame:
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())
    rows = []
    for d in pool["datasets"]:
        y, n_feat = labels(int(d["did"]))
        rows.append(dict(key=d["key"], did=int(d["did"]), n=int(d["n"]), d=n_feat, C=int(y.max()) + 1))
    return pd.DataFrame(rows)


def ordered(reg: pd.DataFrame) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng([settings.base_seed, design.SEED_NAMESPACE, SUBSET_TAG])
    out = []
    for binary in (True, False):
        keys = sorted(reg[(reg.C == 2) == binary].key)
        out.append([keys[i] for i in rng.permutation(len(keys))])
    return out[0], out[1]


def subset(reg: pd.DataFrame, k: int) -> list[str]:
    binary, multi = ordered(reg)
    kb = int(np.floor(k * len(binary) / len(reg) + 0.5))
    return binary[:kb] + multi[: k - kb]


def fits_per_table(did: int) -> int:
    y, _ = labels(did)
    total = 0
    for mult, rep in product(MULTS, range(REPS)):
        cells = design.rep_cells(y, seed_did(did), mult, rep, settings.base_seed)
        total += len({c.labeled for c in cells if c.feasible})
    return total


def cost_model() -> dict[str, np.ndarray]:
    sys.path.insert(0, str(HERE.parent))
    from common import dev_data

    raw = pd.concat([pd.read_parquet(p) for p in (HERE / "out" / "raw").glob("fm*_*.parquet")])
    fit = raw[raw.feasible & raw.fitted_here.fillna(False).astype(bool)]
    g = fit.groupby(["learner", "dataset"]).seconds.mean().reset_index()
    meta = {d: dev_data(d) for d in g.dataset.unique()}
    g["m"] = g.dataset.map(lambda d: len(meta[d].y))
    g["d"] = g.dataset.map(lambda d: meta[d].X.shape[1])
    coef = {}
    for fm, h in g.groupby("learner"):
        A = np.column_stack([np.ones(len(h)), np.log(h.m), np.log(h.d)])
        coef[fm] = np.linalg.lstsq(A, np.log(h.seconds), rcond=None)[0]
    return coef


def estimate(reg: pd.DataFrame, keys: list[str], coef) -> pd.DataFrame:
    rows = []
    for key in keys:
        r = reg[reg.key == key].iloc[0]
        fits = fits_per_table(int(r.did))
        m = min(int(r.n), PANEL_MAX)
        for fm in FMS:
            sec = float(np.exp(coef[fm] @ np.array([1.0, np.log(m), np.log(r.d)])))
            rows.append(dict(key=key, n=r.n, d=r.d, C=r.C, fm=fm, fits=fits, sec_per_fit=sec,
                             cpu_h=fits * sec / 3600))
    return pd.DataFrame(rows)


def main() -> None:
    reg = registry()
    coef = cost_model()
    for k in range(K_MAX, 7, -1):
        est = estimate(reg, subset(reg, k), coef)
        total = est.cpu_h.sum()
        print(f"K={k}: {total:.1f} CPU-h")
        if total <= CAP_CPU_H:
            break
    keys = subset(reg, k)
    print(f"\nchosen K={k}")
    per = est.pivot_table(index=["key", "n", "d", "C", "fits"], columns="fm", values="cpu_h").reset_index()
    print(per.to_markdown(index=False, floatfmt=".2f"))
    print(est.groupby("fm").cpu_h.sum().round(1).to_dict(), f"total {est.cpu_h.sum():.1f}")
    pd.DataFrame(dict(key=keys)).merge(reg).to_csv(HERE / "t1p_subset.csv", index=False)


if __name__ == "__main__":
    main()
