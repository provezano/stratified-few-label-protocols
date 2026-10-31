"""E16 step 4 (exploratory): analysis of the class-merging experiment (e16_merge_run.py).

Same rows, features, seeds and random labelled rows; only the label granularity changes.
The original-C version is rebuilt from the stored Study 9 probabilities restricted to the
first 500 panel positions (as the merged fits), so all versions are scored on the same rows.
The exact decomposition of e16_analysis.decompose is applied to every (table, FM, version, m).

Run from simaudit/: OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python paper/p1/exploratory/e16_multiclass/e16_merge_analysis.py
Writes out/e16_merge_cells.csv, out/e16_merge.csv and out/E16_MERGE_SUMMARY.md.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import e16_common as ec  # noqa: E402
from e16_analysis import decompose  # noqa: E402
from e16_merge_run import MS, PANEL  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.runner.study9 import layout, row_permutation  # noqa: E402
from simaudit.settings import settings  # noqa: E402

OUT = ec.OUT
MERGE = OUT / "merge"
S9 = settings.results_dir / "study9" / "t1prime"


def original(key: str, did: int, fm: str) -> tuple[list[dict], np.ndarray]:
    data = load(did, settings.data_cache_dir)
    y = data.y
    C = int(y.max()) + 1
    perm = row_permutation(len(y), did, settings.base_seed)
    s, _ = layout(len(y))
    cP = np.bincount(y[perm][s:], minlength=C)
    df = pd.read_parquet(S9 / f"{key}_{fm}.parquet", columns=["strategy", "m", "rep", "labeled", "query", "query_pos", "proba"])
    df = df[df.strategy.isin(["random", "stratified"]) & df.m.isin(MS)]
    rows = []
    for t in df.itertuples(index=False):
        q, qp = np.asarray(t.query, int), np.asarray(t.query_pos, int)
        P = np.asarray(t.proba, float).reshape(len(q), C)
        keep = qp < PANEL
        q, P = q[keep], P[keep]
        pred = P.argmax(1)
        cm = np.bincount(y[q] * C + pred, minlength=C * C).reshape(C, C)
        tp = np.diag(cm).astype(float)
        den = cm.sum(0) + cm.sum(1)
        mf = float(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0).mean())
        rows.append(dict(set="merge", dataset=f"{key}|orig", fm=fm, strategy=t.strategy, m=int(t.m), rep=int(t.rep),
                         C=C, N=np.bincount(y[np.asarray(t.labeled, int)], minlength=C).tolist(), cm=cm.ravel().tolist(),
                         soft=P.mean(0).tolist(), macro_f1=mf))
    return rows, cP


def main():
    files = sorted(MERGE.glob("*_fm*.parquet"))
    M = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    D, T = [], []
    for (key, did, fm), g in M.groupby(["dataset", "did", "fm"]):
        rows, cP = original(key, int(did), fm)
        D += rows
        if f"{key}|orig" not in {t["dataset"] for t in T}:
            T.append(dict(dataset=f"{key}|orig", C=len(cP), counts_pool=cP.tolist()))
        for v, h in g.groupby("version"):
            mp = np.asarray(h.merge_map.iloc[0])
            Cv = int(h.C.iloc[0])
            ds = f"{key}|{v}"
            if ds not in {t["dataset"] for t in T}:
                T.append(dict(dataset=ds, C=Cv, counts_pool=np.bincount(mp, weights=cP, minlength=Cv).astype(int).tolist()))
            for t in h.itertuples(index=False):
                D.append(dict(set="merge", dataset=ds, fm=fm, strategy=t.strategy, m=int(t.m), rep=int(t.rep), C=Cv,
                              N=list(t.N), cm=list(t.cm), soft=list(t.soft), macro_f1=float(t.macro_f1)))
    D, T = pd.DataFrame(D), pd.DataFrame(T)
    T["kind"] = np.where(T.C > 2, "multi", "binary")
    L = ec.per_class(D, T)
    X = decompose(L, T)
    assert X.identity_err.abs().max() < 1e-12
    X["table"] = X.dataset.str.split("|").str[0]
    X["version"] = X.dataset.str.split("|").str[1]
    pis = {r.dataset: np.asarray(r.counts_pool, float) / np.sum(r.counts_pool) for r in T.itertuples()}
    X["C_simpson"] = [1 / np.sum(pis[d] ** 2) for d in X.dataset]
    X["pi_min"] = [pis[d].min() for d in X.dataset]
    X["rel_over"] = X.overpred / X.pred
    X["spill_rel"] = X.spill / X.pred
    X.to_csv(OUT / "e16_merge_cells.csv", index=False)
    cols = ["obs", "pred", "cov_own", "sel", "bal", "red", "absb", "spill", "overpred", "rel_over", "spill_rel", "MFS",
            "C_simpson", "pi_min"]
    rows = []
    for (fm, m, v), g in X.groupby(["fm", "m", "version"]):
        rec = dict(fm=fm, m=m, version=v)
        rec.update(ec.boot_frame(g, cols, by="table"))
        rows.append(rec)
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "e16_merge.csv", index=False)
    # paired contrasts vs the original labels (same table, same FM, same m)
    P = []
    base = X[X.version == "orig"].set_index(["table", "fm", "m"])
    for v in ("ter_bal", "ter_imb", "bin_bal", "bin_imb"):
        o = X[X.version == v].set_index(["table", "fm", "m"])
        j = o[["obs", "pred", "spill", "overpred", "rel_over", "spill_rel"]].join(
            base[["obs", "pred", "spill", "overpred", "rel_over", "spill_rel"]], rsuffix="_orig", how="inner").reset_index()
        for c in ("obs", "pred", "spill", "overpred", "rel_over", "spill_rel"):
            j["d_" + c] = j[c] - j[c + "_orig"]
        for (fm, m), g in j.groupby(["fm", "m"]):
            rec = dict(version=v, fm=fm, m=m)
            rec.update(ec.boot_frame(g, ["d_obs", "d_pred", "d_spill", "d_overpred", "d_rel_over", "d_spill_rel"], by="table"))
            P.append(rec)
    P = pd.DataFrame(P)
    P.to_csv(OUT / "e16_merge_paired.csv", index=False)
    show = ["fm", "m", "version", "tables", "obs", "pred", "cov_own", "sel", "red", "absb", "spill", "overpred", "rel_over",
            "MFS", "C_simpson"]
    txt = ["# E16 class merging (exploratory; NEW fits, TabICL and TabPFN v2; 10 T1' tables with C >= 4)", "",
           f"{len(M)} new fits, total {M.seconds.sum() / 3600:.2f} fit-hours; panel = first {PANEL} positions.", "",
           "## Decomposition by version (means over tables)", "", A[show].round(3).to_markdown(index=False), "",
           "## Paired change vs original labels (version - orig)", "", P.round(3).to_markdown(index=False), ""]
    (OUT / "E16_MERGE_SUMMARY.md").write_text("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
