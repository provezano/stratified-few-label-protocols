"""Post-hoc prior corrections on T1′ Study 12 probabilities (EXPLORATORY; plan: PLAN.md).

DistPFN, DistPFN-T (../distpfn.py), EM and the uniform p/pi target (reports/p1_corrections.py),
oracle reference. No FM is called. Writes out/{runs.parquet, gap.csv, effect.csv,
TABLES.md, distpfn_t1p_numbers.tex, DONE}.
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent), str(ROOT / "reports")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from distpfn import distpfn, distpfn_t  # noqa: E402
from p1_corrections import em_prior, reweight  # noqa: E402
from run_distpfn import DIG, fmt, scores, summ  # noqa: E402
from simaudit.data.openml_loader import load  # noqa: E402
from simaudit.runner.study9 import dataset_key  # noqa: E402
from simaudit.settings import PROJECT_ROOT, settings  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
S12 = settings.results_dir / "study12"
FMS = ("fm1", "fm5", "fm2", "fm3")
NAMES = dict(fm1="TabPFN v2", fm5="TabPFN v3.5", fm2="TabICLv2", fm3="Kumo-Tabular")
MULTS = (1, 3, 5, 10)
CORR = ("none", "distpfn", "distpfn_t", "em", "uniform", "oracle")
METRICS = ("macro_f1", "accuracy")


def run_scores(proba: np.ndarray, yq: np.ndarray, ylab: np.ndarray, n_classes: int) -> dict:
    ctx = np.bincount(ylab, minlength=n_classes)
    seen = np.flatnonzero(ctx > 0)
    prior = ctx[seen] / ctx.sum()
    p = proba[:, seen].astype(np.float64)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
    probs = {c: p for c in CORR}
    if len(seen) > 1:
        oracle = np.bincount(yq, minlength=n_classes)[seen].astype(float)
        oracle = np.maximum(oracle, 0.5) / np.maximum(oracle, 0.5).sum()
        probs.update(
            distpfn=distpfn(p, prior),
            distpfn_t=distpfn_t(p, prior),
            em=reweight(p, prior, em_prior(p, prior)),
            uniform=reweight(p, prior, np.full(len(seen), 1 / len(seen))),
            oracle=reweight(p, prior, oracle),
        )
    out = {}
    for name, q in probs.items():
        for k, v in scores(yq, seen[q.argmax(axis=1)], n_classes).items():
            out[f"{name}_{k}"] = v
    return out


def collect() -> pd.DataFrame:
    cache = OUT / "runs.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    pool = yaml.safe_load((PROJECT_ROOT / "data" / "pool_t1prime.yaml").read_text())
    rows = []
    for did in (int(d["did"]) for d in pool["datasets"]):
        y = load(did, settings.data_cache_dir).y
        key = dataset_key(did)
        for fm in FMS:
            fr = pd.read_parquet(S12 / f"{key}_{fm}.parquet")
            fr = fr[fr.strategy.isin(["random", "stratified"])]
            for r in fr.itertuples():
                q = np.asarray(r.query, dtype=np.int64)
                lab = np.asarray(r.labeled, dtype=np.int64)
                nc = int(r.n_classes)
                proba = np.asarray(r.proba, dtype=np.float64).reshape(len(q), nc)
                rows.append(dict(dataset=key, fm=fm, strategy=r.strategy, mult=int(r.mult),
                                 rep=int(r.rep), stored_macro_f1=r.macro_f1,
                                 stored_accuracy=r.accuracy, stored_pc_macro_f1=r.pc_macro_f1,
                                 **run_scores(proba, y[q], y[lab], nc)))
        print(f"t1p: {key}", file=sys.stderr, flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(cache, index=False)
    return out


def tables(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = runs.groupby(["fm", "mult", "strategy", "dataset"]).mean(numeric_only=True)
    gap, eff = [], []
    for fm in FMS:
        for m in MULTS:
            s, r = means.loc[(fm, m, "stratified")], means.loc[(fm, m, "random")]
            idx = s.index.intersection(r.index)
            s, r = s.loc[idx], r.loc[idx]
            for met in METRICS:
                g0 = s[f"none_{met}"] - r[f"none_{met}"]
                for c in CORR:
                    g = s[f"{c}_{met}"] - r[f"{c}_{met}"]
                    gap.append(dict(fm=fm, mult=m, metric=met, corr=c, kind="gap", **summ(g)))
                    if c != "none":
                        gap.append(dict(fm=fm, mult=m, metric=met, corr=c, kind="gap_change",
                                        **summ(g0 - g)))
                        for st, fr in (("random", r), ("stratified", s)):
                            eff.append(dict(fm=fm, mult=m, metric=met, corr=c, strategy=st,
                                            **summ(fr[f"{c}_{met}"] - fr[f"none_{met}"])))
    return pd.DataFrame(gap), pd.DataFrame(eff)


def verdict(gap: pd.DataFrame, fm: str, m: int, c: str) -> str:
    sel = gap[(gap.fm == fm) & (gap.mult == m) & (gap.metric == "macro_f1") & (gap["corr"] == c)]
    g, ch = sel[sel.kind == "gap"].iloc[0], sel[sel.kind == "gap_change"].iloc[0]
    if abs(g["mean"]) <= 0.01 and g.ci_lo <= 0 <= g.ci_hi:
        return "removes"
    return "reduces" if ch.ci_lo > 0 else "no"


def write_tables(runs: pd.DataFrame, gap: pd.DataFrame, eff: pd.DataFrame) -> None:
    chk = {m: float((runs[f"none_{m}"] - runs[f"stored_{m}"]).abs().max()) for m in METRICS}
    pc = float((runs.uniform_macro_f1 - runs.stored_pc_macro_f1).abs().max())
    n = runs.dataset.nunique()
    L = ["# Post-hoc prior corrections on T1′ (Study 12 probabilities), EXPLORATORY", "",
         f"Plan: PLAN.md (commit R022). {n} tables; random 20 / stratified 5 reps. "
         "Means over tables, paired bootstrap 95% CI (10k).", "",
         f"Checks: max |none − stored| macro-F1 {chk['macro_f1']:.1e}, accuracy "
         f"{chk['accuracy']:.1e}; max |uniform − stored pc_macro_f1| {pc:.1e}.", ""]
    for met in METRICS:
        L += [f"## Stratified − random gap, {met}", "",
              "| FM | m | " + " | ".join(CORR) + " | verdict D / DT / EM / uni |",
              "|---|---|" + "---|" * (len(CORR) + 1)]
        for fm in FMS:
            for m in MULTS:
                sel = gap[(gap.fm == fm) & (gap.mult == m) & (gap.metric == met) & (gap.kind == "gap")]
                cells = [fmt(sel[sel["corr"] == c].iloc[0]) for c in CORR]
                v = (" / ".join(verdict(gap, fm, m, c) for c in ("distpfn", "distpfn_t", "em", "uniform"))
                     if met == "macro_f1" else "")
                L.append(f"| {NAMES[fm]} | {m} | " + " | ".join(cells) + f" | {v} |")
        L.append("")
    for met in METRICS:
        L += [f"## Per-arm change (corrected − none), {met}", "",
              "| FM | m | arm | " + " | ".join(CORR[1:]) + " |", "|---|---|---|" + "---|" * (len(CORR) - 1)]
        for fm in FMS:
            for m in MULTS:
                for st in ("random", "stratified"):
                    sel = eff[(eff.fm == fm) & (eff.mult == m) & (eff.metric == met) & (eff.strategy == st)]
                    L.append(f"| {NAMES[fm]} | {m} | {st} | "
                             + " | ".join(fmt(sel[sel["corr"] == c].iloc[0]) for c in CORR[1:]) + " |")
        L.append("")
    (OUT / "TABLES.md").write_text("\n".join(L))


def macros(gap: pd.DataFrame, eff: pd.DataFrame) -> None:
    def f(x: float) -> str:
        return f"{x:+.3f}".replace("-", "$-$").replace("+", "$+$")

    def name(*parts: str) -> str:
        return "".join(DIG.get(ch, ch) for ch in "ExDistTp" + "".join(parts))

    tag = dict(none="None", distpfn="D", distpfn_t="DT", em="EM", uniform="Uni", oracle="Or")
    mt = dict(macro_f1="F", accuracy="A")
    ft = dict(fm1="PFNtwo", fm5="PFNthreefive", fm2="ICL", fm3="Kumo")
    st = dict(random="Rand", stratified="Strat")
    lines = ["% AUTO-GENERATED by paper/p1/exploratory/distpfn/t1p/run_t1p.py. EXPLORATORY "
             "(post-hoc prior corrections on stored Study 12 T1′ probabilities; not pre-registered). Do not edit."]

    def emit(base: str, r, src: str) -> None:
        for suf, val in (("", r.mean), ("Lo", r.ci_lo), ("Hi", r.ci_hi)):
            lines.append(f"\\newcommand{{\\{base}{suf}}}{{{f(val)}}} % EXPLORATORY; source: {src}")

    for r in gap.itertuples():
        kind = "Gap" if r.kind == "gap" else "GapChange"
        emit(name(ft[r.fm], mt[r.metric], kind, tag[r.corr], "M", str(r.mult)), r,
             f"t1p/out/gap.csv ({r.kind}, {r.corr}, {r.fm}, {r.metric}, m={r.mult})")
    for r in eff.itertuples():
        emit(name(ft[r.fm], mt[r.metric], st[r.strategy], tag[r.corr], "M", str(r.mult)), r,
             f"t1p/out/effect.csv ({r.strategy}, {r.corr}, {r.fm}, {r.metric}, m={r.mult})")
    (OUT / "distpfn_t1p_numbers.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    runs = collect()
    gap, eff = tables(runs)
    gap.to_csv(OUT / "gap.csv", index=False)
    eff.to_csv(OUT / "effect.csv", index=False)
    write_tables(runs, gap, eff)
    macros(gap, eff)
    (OUT / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
