"""B2 (EXPLORATORY): E17 metric panel with seven FMs.

Original code: e17_metrics/e17_metrics.py (panel, cell_gaps, tests, moderators,
imbalance_halves, flips, flip_excess, correlation, numbers), imported unchanged, with two
re-bindings: (a) ``_flips`` is replaced by ``flips_n``, identical except that it ranks all FMs
present instead of the first four (the original loops over ``range(4)``); for four FMs it is
the same function (checked in run "4"); (b) ``job12`` scores Study 12 files (column ``mult``
instead of Study 9's ``m``; ``query`` and ``labeled`` are original row indices as in Study 9).

Draw sets (stated, because they differ from the pre-registered panel):
- DEV: Study 7 Part (a) draws for all seven FMs (the four from the existing per-draw cache
  e17_draws.parquet, the three extras newly scored from their stored ``proba``). Like for like.
- T1': the existing E17 panel uses the Study 9 draws of the four pre-registered FMs. The
  extras have Study 12 draws only (m in {1, 3, 5, 10}). For a like-for-like comparison every
  FM is scored on the Study 12 draws: (A) seven FMs on Study 12; (B) the four pre-registered
  FMs on Study 12 (comparator for A). T1' quantities at m = 2, 7, 15, 20, 30 cannot be extended.
- Table traits (TV, C) are the existing E17 ones (Study 9 pool layout), so the imbalance
  halves are the same tables.
Run "4": reproduces the existing E17 CSVs and e17_numbers.tex from the existing cache (and
re-scores a sample of stored fits to check the cache). Run "7": writes
sevenfm/out/b2/seven/{A,B}/ and sevenfm/out/sevenfm_b2.tex.
Usage: nice -n 15 .venv/bin/python paper/p1/exploratory/sevenfm/b2_e17.py {4|7}
"""

from __future__ import annotations

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

S, FMS, OUT = H.setup("b2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as cm  # noqa: E402

cm.FMS = FMS
import e17_metrics as e  # noqa: E402
from simaudit.analysis.study7 import metric_set  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.runner import study9 as r9  # noqa: E402
from simaudit.settings import settings  # noqa: E402

E17 = H.EXPL / "e17_metrics"
S12 = settings.results_dir / "study12"
WORKERS = 2


def flips_n(draws: pd.DataFrame) -> pd.DataFrame:
    """e17_metrics._flips with the FM count taken from the data (all FMs of the cell must be present)."""
    mets = list(e.METRICS)
    g = draws.groupby(["set", "table", "m", "strategy", "learner"])[mets].mean()
    nfm = draws.learner.nunique()
    rows = []
    for (st, tab, m), cell in g.groupby(level=["set", "table", "m"]):
        cell = cell.droplevel(["set", "table", "m"])
        if "stratified" not in cell.index.get_level_values(0) or "random" not in cell.index.get_level_values(0):
            continue
        S_, R_ = cell.xs("stratified"), cell.xs("random")
        S_, R_ = S_.align(R_, join="inner")
        if len(S_) < nfm:
            continue
        for met in mets:
            s, r = S_[met].to_numpy(), R_[met].to_numpy()
            if np.isnan(s).any() or np.isnan(r).any():
                continue
            sign = 1 if e.METRICS[met][1] else -1
            disc, tot = 0, 0
            for i in range(nfm):
                for j in range(i + 1, nfm):
                    a, b = np.sign(s[i] - s[j]), np.sign(r[i] - r[j])
                    if a == 0 or b == 0:
                        continue
                    tot += 1
                    disc += a != b
            if tot == 0:
                continue
            rows.append(dict(set=st, table=tab, m=m, metric=met, family=e.METRICS[met][0],
                             pair_flip=disc / tot,
                             top_flip=float(np.argmax(sign * s) != np.argmax(sign * r))))
    return pd.DataFrame(rows)


def job12(args):
    """e17_metrics.job, T1' branch, on a Study 12 file (results_dir/study12/{key}_{fm}.parquet)."""
    _, key, did, fm = args
    data = load(did, settings.data_cache_dir)
    y = data.y.astype(np.int64)
    df = pd.read_parquet(S12 / f"{key}_{fm}.parquet").rename(columns={"mult": "m"})
    perm = r9.row_permutation(len(y), did, settings.base_seed)
    s, _ = r9.layout(len(y))
    tr = e.table_traits(y[perm][s:])
    df = df[df.strategy.isin(["stratified", "random"])].reset_index(drop=True)
    C = int(df.n_classes.iloc[0])
    rows, err_frozen, err_own = [], {k: 0.0 for k in e.STORED}, {k: 0.0 for k in e.STORED}
    for r in df.itertuples():
        lab = np.asarray(r.labeled, np.int64)
        q = np.asarray(r.query, np.int64)
        yq = y[q]
        P = np.asarray(r.proba, np.float64).reshape(len(q), C)
        fro = metric_set(yq, P, np.arange(C, dtype=np.int64), np.asarray(r.seen, np.int64))
        seen = np.unique(y[lab])
        assert np.array_equal(seen, np.asarray(r.seen)), "seen classes do not match"
        pm = e.panel(yq, P, seen, C, tr["minority"])
        own = dict(macro_f1=pm["macro_f1"], accuracy=pm["accuracy"], balanced_accuracy=pm["balanced_accuracy"],
                   log_loss=pm["log_loss"], auc=pm["auc_ovr_macro_seen"])
        for k in e.STORED:
            sv = float(getattr(r, k))
            for src, err in ((fro[k], err_frozen), (own[k], err_own)):
                if np.isnan(sv) and np.isnan(src):
                    continue
                err[k] = max(err[k], abs(src - sv) if not (np.isnan(sv) or np.isnan(src)) else np.inf)
        rows.append(dict(set="t1prime", table=key, learner=fm, strategy=r.strategy, m=int(r.m),
                         rep=int(r.rep), coverage=float(r.coverage), **pm))
    rep = dict(set="t1prime", table=key, learner=fm, n_rows=len(df),
               **{f"frozen_{k}": v for k, v in err_frozen.items()}, **{f"own_{k}": v for k, v in err_own.items()})
    return rows, {**rep, **{f"trait_{k}": v for k, v in tr.items()}}


def pipeline(draws: pd.DataFrame, repro: pd.DataFrame, traits: pd.DataFrame, out: Path) -> None:
    """e17_metrics.main from the cache on, outputs in ``out`` (no figure, no selector side analysis)."""
    out.mkdir(parents=True, exist_ok=True)
    e.OUTNUM = out
    e._flips = flips_n
    traits.to_csv(out / "e17_table_traits.csv", index=False)
    gaps = e.cell_gaps(draws)
    gaps.to_csv(out / "e17_cell_gaps.csv", index=False)
    tst = e.tests(gaps)
    tst.to_csv(out / "e17_gap_tests.csv", index=False)
    mod = e.moderators(gaps, traits)
    mod.to_csv(out / "e17_moderators.csv", index=False)
    fl = e.flips(draws)
    fl.to_csv(out / "e17_flips_cells.csv", index=False)
    fl.groupby(["kind", "set", "family", "metric"])[["pair_flip", "top_flip"]].mean().reset_index().to_csv(
        out / "e17_flip_rates.csv", index=False)
    fl.groupby(["kind", "set", "metric", "m"])[["pair_flip", "top_flip"]].mean().reset_index().to_csv(
        out / "e17_flip_rates_by_m.csv", index=False)
    fx = e.flip_excess(fl)
    fx.to_csv(out / "e17_flip_excess.csv", index=False)
    cg, cd, clus = e.correlation(gaps, draws)
    clus.to_csv(out / "e17_clusters.csv", index=False)
    halves = e.imbalance_halves(gaps, traits)
    halves.to_csv(out / "e17_imbalance_halves.csv", index=False)
    srho = pd.read_csv(E17 / "e17_selector_cluster_rho.csv")  # Study 9 selectors; not FM-count dependent
    e.numbers(tst, halves, mod, fl, clus, repro, srho, fx)


def traits_from(repro):
    return (repro[["set", "table"] + [c for c in repro if c.startswith("trait_")]]
            .drop_duplicates(["set", "table"]).rename(columns=lambda c: c.replace("trait_", "")))


def main() -> None:
    warnings.filterwarnings("ignore")
    cache = pd.read_parquet(E17 / "e17_draws.parquet")
    repro = pd.read_csv(E17 / "e17_reproduction.csv")
    traits = traits_from(repro)
    if S == "4":
        # cache check: re-score 8 stored fits (2 DEV tables, 1 T1' table x 4 FMs) and compare with the cache
        mets = list(e.METRICS)
        worst = 0.0
        t1k = sorted(e.t1_pool().items())[0]
        for args in [("dev", "337", 337, fm) for fm in FMS] + [("dev", "1063", 1063, "fm3")] + \
                    [("t1prime", t1k[0], t1k[1], fm) for fm in ("fm1", "fm5")]:
            n = pd.DataFrame(e.job(args)[0])
            o = cache[(cache.set == args[0]) & (cache.table == args[1]) & (cache.learner == args[3])]
            k = ["strategy", "m", "rep"]
            mg = n.merge(o, on=k, suffixes=("", "_c"))
            assert len(mg) == len(n) == len(o), (len(mg), len(n), len(o))
            d = np.nanmax(np.abs(mg[mets].to_numpy(float) - mg[[c + "_c" for c in mets]].to_numpy(float)))
            worst = max(worst, float(d))
        print(f"[verify cache] max |re-scored - cached| over panel metrics: {worst:.2e}")
        pipeline(cache, repro, traits, OUT)
        bad = [] if worst < 1e-9 else ["cache"]
        for f in ("e17_gap_tests.csv", "e17_moderators.csv", "e17_flips_cells.csv", "e17_flip_rates.csv",
                  "e17_flip_rates_by_m.csv", "e17_flip_excess.csv", "e17_clusters.csv", "e17_imbalance_halves.csv",
                  "e17_table_traits.csv", "e17_cell_gaps.csv"):
            bad += [] if H.verify_csv(OUT / f, E17 / f) else [f]
        new = [ln for ln in (OUT / "e17_numbers.tex").read_text().splitlines() if ln.startswith("\\newcommand")]
        bad += H.verify_lines(new, H.EXPL / "out" / "e17_numbers.tex", "e17_numbers")
        print("VERIFY B2:", "PASS" if not bad else f"FAIL {bad}")
        return

    # ---- run 7: score the extras (DEV, Study 7 Part (a)) and all seven FMs on T1' Study 12 draws
    sc = OUT / "draws_scored.parquet"
    if sc.exists():
        new = pd.read_parquet(sc)
        rep_new = pd.read_csv(OUT / "repro_scored.csv")
    else:
        jobs_dev = [("dev", str(d), d, fm) for d in cm.dev_ids() for fm in H.EXTRA]
        jobs_t1 = [("t1prime", k, d, fm) for k, d in e.t1_pool().items() for fm in H.FMS7]
        with ProcessPoolExecutor(WORKERS) as ex:
            res = list(ex.map(e.job, jobs_dev, chunksize=1)) + list(ex.map(job12, jobs_t1, chunksize=1))
        new = pd.DataFrame([r for rs, _ in res for r in rs])
        rep_new = pd.DataFrame([rp for _, rp in res])
        new.to_parquet(sc, index=False)
        rep_new.to_csv(OUT / "repro_scored.csv", index=False)
    dev4 = cache[cache.set == "dev"]
    t1_12 = new[new.set == "t1prime"]
    dev7 = pd.concat([dev4, new[new.set == "dev"]], ignore_index=True)
    rep_dev7 = pd.concat([repro[repro.set == "dev"], rep_new[rep_new.set == "dev"]], ignore_index=True)
    rep_t1 = rep_new[rep_new.set == "t1prime"]
    # traits: existing E17 traits (Study 9 pool layout) for both sets
    A = pd.concat([dev7, t1_12], ignore_index=True)
    B = pd.concat([dev4, t1_12[t1_12.learner.isin(H.FMS4)]], ignore_index=True)
    pipeline(A, pd.concat([rep_dev7, rep_t1], ignore_index=True), traits, OUT / "A")
    pipeline(B, pd.concat([repro[repro.set == "dev"], rep_t1[rep_t1.learner.isin(H.FMS4)]], ignore_index=True),
             traits, OUT / "B")
    lines = []
    for tag, d, note in (("A", "ExSevFmESeventeen", "B2 (A): seven FMs; DEV Study 7 Part (a) draws, T1' Study 12 draws "
                          "(the pre-registered E17 T1' panel uses Study 9 draws)"),
                         ("B", "ExSevFmFourSTwelveESeventeen", "B2 (B): four pre-registered FMs re-scored on the T1' "
                          "Study 12 draws (comparator for A); DEV rows = the existing four-FM DEV panel")):
        for ln in (OUT / tag / "e17_numbers.tex").read_text().splitlines():
            if not ln.startswith("\\newcommand") or "{\\ExESeventeenSel" in ln:
                continue
            if tag == "B" and "Tp" not in H.macro_name(ln) and "Repro" not in H.macro_name(ln):
                continue  # B differs from the existing panel only on T1'
            ln = ln.replace("\\newcommand{\\ExESeventeen", "\\newcommand{\\" + d, 1)
            lines.append(ln + f" [SEVEN-FM EXTENSION, exploratory: {note}]")
    H.write_tex(H.HERE / "out" / "sevenfm_b2.tex", lines, "B2 E17 metric panel, seven FMs (A) and four FMs on Study 12 draws (B)")
    print(len(lines), "macros")


if __name__ == "__main__":
    main()
