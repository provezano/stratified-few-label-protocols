"""DistPFN on the stored Study 7 Part (a) DEV runs (EXPLORATORY; plan: PLAN.md).

Post hoc on stored probabilities: no FM is called. Writes out/runs.parquet,
out/gap.csv, out/effect.csv, out/SUMMARY.md, out/distpfn_numbers.tex.
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as cm  # noqa: E402
from distpfn import distpfn, distpfn_inst, distpfn_t  # noqa: E402
from simaudit.analysis.report import bootstrap_ci  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
CORR = ("none", "distpfn", "distpfn_t", "distpfn_inst", "uniform", "oracle")
METRICS = ("macro_f1", "accuracy")


def scores(y_true: np.ndarray, pred: np.ndarray, n_classes: int) -> dict[str, float]:
    """Macro-F1 over all classes (zero_division = 0) and accuracy (as p1_corrections)."""
    c = np.zeros((n_classes, n_classes))
    np.add.at(c, (y_true, pred), 1)
    tp = np.diag(c)
    den = c.sum(0) + c.sum(1)
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    return dict(macro_f1=float(f1.mean()), accuracy=float(tp.sum() / len(y_true)))


def run_scores(proba: np.ndarray, y: np.ndarray, labeled: np.ndarray) -> dict[str, float]:
    n_classes = proba.shape[1]
    rest = np.setdiff1d(np.arange(len(y)), labeled)
    ctx = np.bincount(y[labeled], minlength=n_classes)
    seen = np.flatnonzero(ctx > 0)
    prior = ctx[seen] / ctx.sum()
    p = proba[rest][:, seen].astype(np.float64)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
    probs = {c: p for c in CORR}
    if len(seen) > 1:
        oracle = np.bincount(y[rest], minlength=n_classes)[seen].astype(float)
        oracle = np.maximum(oracle, 0.5) / np.maximum(oracle, 0.5).sum()
        uni = np.full(len(seen), 1 / len(seen))
        probs.update(
            distpfn=distpfn(p, prior),
            distpfn_t=distpfn_t(p, prior),
            distpfn_inst=distpfn_inst(p, prior),
            uniform=p * (uni / prior),
            oracle=p * (oracle / prior),
        )
    out = {}
    for name, q in probs.items():
        for k, v in scores(y[rest], seen[q.argmax(axis=1)], n_classes).items():
            out[f"{name}_{k}"] = v
    return out


def collect() -> pd.DataFrame:
    cache = OUT / "runs.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    for did in cm.dev_ids():
        y = cm.dev_data(did).y
        n_classes = int(y.max()) + 1
        for fm in cm.FMS:
            fr = pd.read_parquet(cm.S7 / "a" / f"{did}_{fm}.parquet")
            fr = fr[fr.strategy.isin(["random", "stratified"])]
            for r in fr.itertuples():
                lab = np.asarray(r.labeled, dtype=np.int64)
                proba = np.asarray(r.proba, dtype=np.float64).reshape(len(y), n_classes)
                rows.append(
                    dict(dataset=did, fm=fm, strategy=r.strategy, mult=int(r.mult), rep=r.rep,
                         stored_macro_f1=r.macro_f1, stored_accuracy=r.accuracy,
                         **run_scores(proba, y, lab))
                )
        print(f"distpfn: {did}", file=sys.stderr, flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(cache, index=False)
    return out


def summ(x: pd.Series) -> dict[str, float]:
    v = x.dropna().to_numpy()
    lo, hi = bootstrap_ci(v)
    return dict(n=len(v), mean=float(v.mean()), ci_lo=lo, ci_hi=hi, win=float(np.mean(v > 0)))


def tables(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = runs.groupby(["fm", "mult", "strategy", "dataset"]).mean(numeric_only=True)
    gap, eff = [], []
    for fm in cm.FMS:
        for m in cm.MULTS:
            s, r = means.loc[(fm, m, "stratified")], means.loc[(fm, m, "random")]
            for met in METRICS:
                g0 = s[f"none_{met}"] - r[f"none_{met}"]
                for c in CORR:
                    g = s[f"{c}_{met}"] - r[f"{c}_{met}"]
                    gap.append(dict(fm=fm, mult=m, metric=met, corr=c, kind="gap", **summ(g)))
                    if c != "none":
                        gap.append(dict(fm=fm, mult=m, metric=met, corr=c, kind="gap_change",
                                        **summ(g0 - g)))
                        for strat, frame in (("random", r), ("stratified", s)):
                            eff.append(dict(fm=fm, mult=m, metric=met, corr=c, strategy=strat,
                                            **summ(frame[f"{c}_{met}"] - frame[f"none_{met}"])))
    return pd.DataFrame(gap), pd.DataFrame(eff)


def verdict(gap: pd.DataFrame, fm: str, m: int, c: str = "distpfn") -> str:
    sel = gap[(gap.fm == fm) & (gap.mult == m) & (gap.metric == "macro_f1") & (gap["corr"] == c)]
    g = sel[sel.kind == "gap"].iloc[0]
    ch = sel[sel.kind == "gap_change"].iloc[0]
    if abs(g["mean"]) <= 0.01 and g.ci_lo <= 0 <= g.ci_hi:
        return "removes"
    return "reduces" if ch.ci_lo > 0 else "does not reduce"


def fmt(r: pd.Series) -> str:
    return f"{r['mean']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"


def summary(runs: pd.DataFrame, gap: pd.DataFrame, eff: pd.DataFrame) -> None:
    chk = {m: float((runs[f"none_{m}"] - runs[f"stored_{m}"]).abs().max()) for m in METRICS}
    L = ["# DistPFN on stored Study 7 DEV contexts (EXPLORATORY)", "",
         "Plan: PLAN.md (commit R023, before the run). Post hoc on the stored Study 7 "
         "Part (a) probabilities; no FM called; 20 DEV tables; random 20 reps, stratified 5 "
         "reps per (table, FM, m). Means over tables, paired bootstrap 95% CI (10k).", "",
         f"Check: recomputed 'none' vs stored, max |diff| macro-F1 = {chk['macro_f1']:.1e}, "
         f"accuracy = {chk['accuracy']:.1e}.", ""]
    for met in METRICS:
        L += [f"## Stratified - random gap, {met}", "",
              "| FM | m | none | DistPFN | DistPFN-T | DistPFN-inst | uniform | oracle | gap change (DistPFN) | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for fm in cm.FMS:
            for m in cm.MULTS:
                sel = gap[(gap.fm == fm) & (gap.mult == m) & (gap.metric == met)]
                cells = [fmt(sel[(sel["corr"] == c) & (sel.kind == "gap")].iloc[0]) for c in CORR]
                ch = fmt(sel[(sel["corr"] == "distpfn") & (sel.kind == "gap_change")].iloc[0])
                v = verdict(gap, fm, m) if met == "macro_f1" else ""
                L.append(f"| {cm.FM_NAMES[fm]} | {m} | " + " | ".join(cells) + f" | {ch} | {v} |")
        L.append("")
    for met in METRICS:
        L += [f"## Effect of the correction under each protocol (corrected - none), {met}", "",
              "| FM | m | protocol | DistPFN | DistPFN-T | uniform | oracle |", "|---|---|---|---|---|---|---|"]
        for fm in cm.FMS:
            for m in cm.MULTS:
                for st in ("random", "stratified"):
                    sel = eff[(eff.fm == fm) & (eff.mult == m) & (eff.metric == met) & (eff.strategy == st)]
                    cells = [fmt(sel[sel["corr"] == c].iloc[0]) for c in ("distpfn", "distpfn_t", "uniform", "oracle")]
                    L.append(f"| {cm.FM_NAMES[fm]} | {m} | {st} | " + " | ".join(cells) + " |")
        L.append("")
    (OUT / "TABLES.md").write_text("\n".join(L))


DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


def macros(gap: pd.DataFrame, eff: pd.DataFrame) -> None:
    def f(x: float) -> str:
        return f"{x:+.3f}".replace("-", "$-$").replace("+", "$+$")

    def name(*parts: str) -> str:
        return "".join(DIG.get(ch, ch) for ch in "ExDist" + "".join(parts))

    tag = {"none": "None", "distpfn": "D", "distpfn_t": "DT", "uniform": "Uni", "oracle": "Or"}
    met_tag = {"macro_f1": "F", "accuracy": "A"}
    fmt_tag = {"fm1": "PFNtwo", "fm5": "PFNthreefive", "fm2": "ICL", "fm3": "Kumo"}
    lines = ["% AUTO-GENERATED by paper/p1/exploratory/distpfn/run_distpfn.py. EXPLORATORY "
             "(DistPFN post hoc on stored Study 7 Part (a) DEV probabilities; not pre-registered). Do not edit."]
    for r in gap.itertuples():
        if r.corr not in tag:
            continue
        kind = "Gap" if r.kind == "gap" else "GapChange"
        base = name(fmt_tag[r.fm], met_tag[r.metric], kind, tag[r.corr], "M", str(r.mult))
        if r.kind == "gap_change" and r.corr != "distpfn":
            continue
        src = f"out/gap.csv ({r.kind}, {r.corr}, {r.fm}, {r.metric}, m={r.mult})"
        for suf, val in (("", r.mean), ("Lo", r.ci_lo), ("Hi", r.ci_hi)):
            lines.append(f"\\newcommand{{\\{base}{suf}}}{{{f(val)}}} % EXPLORATORY; source: {src}")
    for r in eff[(eff["corr"].isin(["distpfn", "distpfn_t"])) & (eff.strategy == "random")].itertuples():
        base = name(fmt_tag[r.fm], met_tag[r.metric], "Rand", tag[r.corr], "M", str(r.mult))
        src = f"out/effect.csv (random, {r.corr}, {r.fm}, {r.metric}, m={r.mult})"
        for suf, val in (("", r.mean), ("Lo", r.ci_lo), ("Hi", r.ci_hi)):
            lines.append(f"\\newcommand{{\\{base}{suf}}}{{{f(val)}}} % EXPLORATORY; source: {src}")
    (OUT / "distpfn_numbers.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    runs = collect()
    gap, eff = tables(runs)
    gap.to_csv(OUT / "gap.csv", index=False)
    eff.to_csv(OUT / "effect.csv", index=False)
    summary(runs, gap, eff)
    macros(gap, eff)
    (OUT / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
