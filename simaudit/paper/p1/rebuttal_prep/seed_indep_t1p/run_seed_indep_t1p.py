"""Independent learner seed on PMLB/UCI (T1', Study 12) for the four FMs. EXPLORATORY.
Run once (fit, repro, analyze) after all fits finished; outputs in out/. Same design and reading as
paper/p1/exploratory/seed_indep/PLAN.md (DEV), transposed to Study 12:

- Stored labelled sets: results/study12/selections/{key}.parquet, strategies stratified and random,
  attempt 0 (labeled_pos = positions in the fixed row permutation, runner.study12.row_permutation).
  Checked against the stored fits results/study12/{key}_{fm}.parquet: same (strategy, mult, rep),
  same selection seed, and perm[labeled_pos] == stored `labeled` in every cell.
- New learner seed: run_seed_indep.fit_seed (SeedSequence child, namespace 0xF00D); asserts that it
  differs from every selection seed and redraw seed of the table and that the streams differ
  (run_seed_indep.check_streams).
- Fit/metrics: runner.study12.fit_record (panel query = E minus L, panel_metrics) unchanged; only
  the seed argument changes. Probabilities are not stored (metrics only).
- Reproduction control (mode repro): one S and one R cell at m = 3, rep 0, on 10 tables refitted with
  the ORIGINAL seed; max |d macro-F1| and max |d proba| vs stored, per FM, on the run host.
- Outputs: results/seed_indep/t1p/{key}_{fm}.parquet (raw, git-ignored),
  paper/p1/rebuttal_prep/seed_indep_t1p/out/ (summary from mode analyze).

Reading (fixed before the run, copied from the DEV plan): per FM, materially unchanged if the 95%
table-bootstrap CI of (gap_indep - gap_orig), macro-F1, mean over m, lies within +-0.01; changed if
the CI excludes 0 and |mean| > 0.01; otherwise inconclusive.

Usage (from simaudit/):
    python paper/p1/rebuttal_prep/seed_indep_t1p/run_seed_indep_t1p.py fit fm2,fm1,fm3 --workers 4
    python paper/p1/rebuttal_prep/seed_indep_t1p/run_seed_indep_t1p.py fit fm5 --workers 4   # on the host that holds the TabPFN v3.5 licence token
    python paper/p1/rebuttal_prep/seed_indep_t1p/run_seed_indep_t1p.py repro fm2,fm1,fm3,fm5
    python paper/p1/rebuttal_prep/seed_indep_t1p/run_seed_indep_t1p.py analyze fm1,fm2,fm3,fm5
    add --limit 1 for a smoke test (first table only; writes to results/seed_indep/t1p_smoke/).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "exploratory" / "seed_indep"))
sys.path.insert(0, str(HERE.parents[1] / "exploratory"))
import run_seed_indep as rsi  # noqa: E402  (fit_seed, check_streams, _atomic, _init)
from simaudit.analysis import study12 as a12  # noqa: E402
from simaudit.analysis.report import bootstrap_ci  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.runner import study4  # noqa: E402
from simaudit.runner import study12 as r12  # noqa: E402
from simaudit.runner.study9 import dataset_key  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

STRATS = ("stratified", "random")
METRICS = ("macro_f1", "accuracy", "balanced_accuracy")
S12 = settings.results_dir / "study12"
OUT = HERE / "out"


def raw_dir(smoke: bool) -> Path:
    return settings.results_dir / "seed_indep" / ("t1p_smoke" if smoke else "t1p")


def t1p_ids() -> list[int]:
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())
    return [int(d["did"]) for d in pool["datasets"]]


def _data(did: int):
    data = load(did, settings.data_cache_dir)
    perm = r12.row_permutation(len(data.y), did, settings.base_seed)
    return data.X[perm], data.y[perm], perm


def _cells(key: str, fm: str, perm: np.ndarray) -> tuple[pd.DataFrame, set[int]]:
    sel = pd.read_parquet(S12 / "selections" / f"{key}.parquet")
    all_seeds = set(sel.seed.astype(int)) - {-1}
    sel = sel[sel.strategy.isin(STRATS) & (sel.attempt == 0)]
    st = pd.read_parquet(S12 / f"{key}_{fm}.parquet", columns=["strategy", "mult", "rep", "seed", "labeled", *METRICS])
    st = st[st.strategy.isin(STRATS)]
    m = sel.merge(st, on=["strategy", "mult", "rep"], suffixes=("", "_st"))
    assert len(m) == len(sel) == len(st), (key, fm, len(m), len(sel), len(st))
    assert (m.seed.astype(int) == m.seed_st.astype(int)).all(), key
    for a, b in zip(m.labeled_pos, m.labeled):
        assert np.array_equal(perm[np.asarray(a, dtype=np.int64)], np.asarray(b, dtype=np.int64)), key
    return m, all_seeds


def fit_job(did: int, fm: str, smoke: bool) -> str:
    key = dataset_key(did)
    out = raw_dir(smoke) / f"{key}_{fm}.parquet"
    if out.exists():
        return f"skip {out.name}"
    X, y, perm = _data(did)
    cells, all_seeds = _cells(key, fm, perm)
    new = [rsi.fit_seed(settings.base_seed, did, int(r.mult), int(r.rep), r.strategy) for r in cells.itertuples()]
    rsi.check_streams(cells.seed.astype(int).tolist(), new, all_seeds)
    fn = study4._predictor(fm)
    rows = []
    for r, s_new in zip(cells.itertuples(), new):
        lab = np.asarray(r.labeled_pos, dtype=np.int64)
        rec = r12.fit_record(fn, X, y, lab, s_new, perm)
        row = dict(dataset=key, learner=fm, strategy=r.strategy, mult=int(r.mult), rep=int(r.rep),
                   sel_seed=int(r.seed), fit_seed=s_new, budget=len(lab), coverage=rec["coverage"],
                   seconds=rec["seconds"])
        for k in METRICS:
            row[k] = rec[k]
            row[f"orig_{k}"] = float(getattr(r, k))
        rows.append(row)
    rsi._atomic(pd.DataFrame(rows), out)
    return f"done {out.name}"


def repro_job(did: int, fm: str) -> list[dict]:
    key = dataset_key(did)
    X, y, perm = _data(did)
    st = pd.read_parquet(S12 / f"{key}_{fm}.parquet")
    sel = pd.read_parquet(S12 / "selections" / f"{key}.parquet")
    fn = study4._predictor(fm)
    res = []
    for strat in STRATS:
        s = sel[(sel.strategy == strat) & (sel.mult == 3) & (sel.rep == 0) & (sel.attempt == 0)].iloc[0]
        o = st[(st.strategy == strat) & (st.mult == 3) & (st.rep == 0)].iloc[0]
        rec = r12.fit_record(fn, X, y, np.asarray(s.labeled_pos, dtype=np.int64), int(s.seed), perm)
        res.append(dict(dataset=key, learner=fm, strategy=strat, d_macro_f1=abs(rec["macro_f1"] - o.macro_f1),
                        d_proba=float(np.max(np.abs(np.asarray(rec["proba"], float) - np.asarray(o.proba, float))))))
    return res


def analyze(fms: list[str], smoke: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, lines = [], ["# Independent learner seed on PMLB/UCI (T1'): EXPLORATORY", ""]
    for fm in fms:
        parts = [pd.read_parquet(p) for p in sorted(raw_dir(smoke).glob(f"*_{fm}.parquet"))]
        if not parts:
            continue
        d = pd.concat(parts)
        for met in METRICS:
            gaps = {}
            for tag, col in (("orig", f"orig_{met}"), ("indep", met)):
                x = d.rename(columns={"mult": "mult"}).assign(**{met: d[col]})
                means = a12.cell_means(x, met)
                gaps[tag] = a12.delta(means, (fm, "stratified"), (fm, "random"), a12.MULTS, met)
            diff = (gaps["indep"] - gaps["orig"]).dropna()
            lo, hi = bootstrap_ci(diff.to_numpy()) if len(diff) > 1 else (np.nan, np.nan)
            verdict = ("materially unchanged" if lo >= -0.01 and hi <= 0.01 else
                       "changed" if (lo > 0 or hi < 0) and abs(diff.mean()) > 0.01 else "inconclusive")
            cell = (d[met] - d[f"orig_{met}"]).abs()
            rows.append(dict(learner=fm, metric=met, n_tables=len(diff), gap_orig=gaps["orig"].mean(),
                             gap_indep=gaps["indep"].mean(), diff=diff.mean(), lo=lo, hi=hi,
                             share_cells_gt_001=float((cell > 0.01).mean()), mean_abs_cell=float(cell.mean()),
                             verdict=verdict if met == "macro_f1" else "(secondary) " + verdict))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "summary_t1p.csv", index=False)
    (OUT / "SUMMARY_T1P.md").write_text("\n".join(lines) + t.to_markdown(index=False) + "\n")
    print(t.round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("fit", "repro", "analyze"))
    ap.add_argument("learners")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    fms = a.learners.split(",")
    smoke = a.limit > 0
    if a.mode == "analyze":
        return analyze(fms, smoke)
    ids = t1p_ids()[: a.limit] if a.limit else t1p_ids()
    raw_dir(smoke).mkdir(parents=True, exist_ok=True)
    prov = raw_dir(smoke) / "provenance.json"
    old = json.loads(prov.read_text()) if prov.exists() else []
    old.append(dict(mode=a.mode, learners=fms, workers=a.workers, host=platform.node(), machine=platform.machine(),
                    started_utc=datetime.now(UTC).isoformat(timespec="seconds"), base_seed=settings.base_seed,
                    namespace=rsi.NAMESPACE))
    prov.write_text(json.dumps(old, indent=2))
    t0, res_rows = time.time(), []
    with ProcessPoolExecutor(min(a.workers, 6), initializer=rsi._init) as pool:
        if a.mode == "fit":
            futs = {pool.submit(fit_job, did, fm, smoke): (did, fm) for fm in fms for did in ids}
        else:
            futs = {pool.submit(repro_job, did, fm): (did, fm) for fm in fms for did in ids[:10]}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {futs[f]}: {exc!r}", flush=True)
                continue
            if a.mode == "repro":
                res_rows.extend(r)
                r = "repro"
            print(f"[{i}/{len(futs)} {time.time() - t0:.0f}s] {futs[f]} {r}", flush=True)
    if a.mode == "repro":
        OUT.mkdir(parents=True, exist_ok=True)
        rp = pd.DataFrame(res_rows)
        rp.to_csv(OUT / "repro_t1p.csv", index=False)
        print(rp.groupby("learner")[["d_macro_f1", "d_proba"]].max())


if __name__ == "__main__":
    main()
