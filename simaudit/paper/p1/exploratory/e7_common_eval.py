"""E7 (exploratory, R3-W4/R2-W4/R2-Q4): re-score both protocols on a common evaluation set.

Under the frozen Study 7 metric each draw is scored on its own complement U = pool ∖ L,
so U depends on the protocol (stratified draws remove more minority rows). From the
stored per-row probabilities (Study 7 Part (a), 5 stratified + 20 random draws per
(table, m); selections are identical across the four FMs) we re-score on rows that no
compared draw labelled:

- **paired (primary)**: random draw r is paired with stratified draw r mod 5 (the
  5 stratified reps 0-4 meet random reps 0-4 exactly; reps 5-19 reuse them), and both
  are scored on pool ∖ (L_s ∪ L_r), the intersection of their two complements. Class
  support is reported (minimum count per class over the pairs); a pair in which some
  class has 0 rows is excluded and counted. Cell gap = mean over the kept pairs.
- **strict (secondary)**: U* = rows never labelled by ANY of the 25 stratified or
  random draws of that (table, m); every draw is scored on U*. U* is small or empty at
  large m on small tables, and the 5 stratified draws can use up a rare class; cells
  with |U*| < 30 or with a class absent from U* are excluded and counted.

Metric conventions as the frozen ``metric_set``: argmax over all C columns (unseen
classes have probability 0), accuracy, balanced accuracy over classes present in the
evaluation rows; macro-F1 over the C classes for the current score, and over the
classes for the paired set (every class is present after the exclusion) and over the
classes present in U* for the strict set (identical for usable cells). Alignment check: the fast scorer reproduces every stored macro-F1,
accuracy and balanced accuracy on the original complements to 1e-9 (else stop).

Gap = stratified − random per table (mean over draws), per FM and m and averaged
over m; summary over the 20 tables: mean, bootstrap 95% CI (10k), Wilcoxon, Holm over
the 4 FMs within metric × m × variant; and the change common − current with the
same statistics.

Writes out/e7_cells.csv, out/e7_gaps.csv, out/E7_SUMMARY.md.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as cm  # noqa: E402

TOL = 1e-9
MIN_ROWS = 30
METRICS = ("macro_f1", "accuracy", "balanced_accuracy")


def run(job: tuple[int, str]) -> tuple[list[dict], list[str]]:
    did, fm = job
    y = cm.dev_data(did).y
    n = len(y)
    df = cm.load_a(did, fm, proba=True)
    C = int(df.n_classes.iloc[0])
    rows, errors = [], []
    preds, labs = {}, {}
    for r in df.itertuples():
        P = np.asarray(r.proba, dtype=np.float64).reshape(n, C)
        pred = P.argmax(1)
        lab = np.asarray(r.labeled, dtype=np.int64)
        mask = np.ones(n, bool)
        mask[lab] = False
        got = cm.fast_scores(y[mask], pred[mask], C)
        for k, v in zip(METRICS, got):
            if abs(v - getattr(r, k)) > TOL:
                errors.append(f"{did} {fm} {r.strategy} m={r.mult} rep={r.rep}: {k} {v} vs {getattr(r, k)}")
        preds[(r.strategy, r.mult, r.rep)] = pred
        labs[(r.strategy, r.mult, r.rep)] = mask
    for m in cm.MULTS:
        keys_s = [k for k in preds if k[0] == "stratified" and k[1] == m]
        keys_r = [k for k in preds if k[0] == "random" and k[1] == m]
        strict = np.logical_and.reduce([labs[k] for k in keys_s + keys_r])
        present = np.unique(y[strict]) if strict.any() else np.array([], int)
        ok = strict.sum() >= MIN_ROWS and len(present) == C
        rec = dict(dataset=did, fm=fm, mult=m, C=C, n=n, strict_rows=int(strict.sum()),
                   strict_classes=len(present), strict_ok=bool(ok))
        for k in METRICS:
            cur = df[df.mult == m].groupby("strategy")[k].mean()
            rec[f"cur_{k}"] = cur["stratified"] - cur["random"]
        if ok:
            sc = {k: np.array([cm.fast_scores(y[strict], preds[k][strict], C, labels=present) for k in ks]).mean(0)
                  for k, ks in (("s", keys_s), ("r", keys_r))}
            for i, k in enumerate(METRICS):
                rec[f"strict_{k}"] = sc["s"][i] - sc["r"][i]
        pair_rows, pair_sizes, pair_min_class, pair_missing = [], [], [], 0
        for kr in keys_r:
            ks = ("stratified", m, kr[2] % len(keys_s))
            u = labs[ks] & labs[kr]
            support = np.bincount(y[u], minlength=C)
            if support.min() == 0:
                pair_missing += 1
                continue
            a = cm.fast_scores(y[u], preds[ks][u], C)
            b = cm.fast_scores(y[u], preds[kr][u], C)
            pair_rows.append(np.subtract(a, b))
            pair_sizes.append(int(u.sum()))
            pair_min_class.append(int(support.min()))
        if pair_rows:
            pr = np.mean(pair_rows, axis=0)
            for i, k in enumerate(METRICS):
                rec[f"pair_{k}"] = pr[i]
        rec.update(pair_kept=len(pair_rows), pair_missing=pair_missing,
                   pair_rows_min=min(pair_sizes) if pair_sizes else 0,
                   pair_rows_mean=float(np.mean(pair_sizes)) if pair_sizes else 0.0,
                   pair_min_class=min(pair_min_class) if pair_min_class else 0,
                   strict_min_class=int(np.bincount(y[strict], minlength=C).min()))
        rows.append(rec)
    return rows, errors


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    out = []
    for variant in ("cur", "strict", "pair", "strict_minus_cur", "pair_minus_cur"):
        for k in METRICS:
            c = cells.copy()
            if variant.endswith("_minus_cur"):
                base = variant.split("_")[0]
                c["v"] = c[f"{base}_{k}"] - c[f"cur_{k}"]
                sel = c.strict_ok if base == "strict" else c.pair_kept > 0
            else:
                c["v"] = c[f"{variant}_{k}"]
                sel = c.strict_ok if variant == "strict" else (c.pair_kept > 0 if variant == "pair" else np.ones(len(c), bool))
            c = c[sel]
            for fm, g in c.groupby("fm"):
                for m, gm in g.groupby("mult"):
                    out.append(dict(variant=variant, metric=k, fm=fm, mult=str(m), **cm.summarize(gm.v.to_numpy())))
                if variant.startswith("strict"):
                    continue  # the strict set is not defined for every (table, m): no average over m
                avg = g.groupby("dataset").v.mean()
                out.append(dict(variant=variant, metric=k, fm=fm, mult="all", **cm.summarize(avg.to_numpy())))
    res = pd.DataFrame(out)
    return pd.concat([cm.add_holm(g) for _, g in res.groupby(["variant", "metric", "mult"], sort=False)])


def main() -> None:
    jobs = [(d, fm) for d in cm.dev_ids() for fm in cm.FMS]
    with ProcessPoolExecutor(3) as pool:
        res = list(pool.map(run, jobs))
    errors = [e for _, es in res for e in es]
    if errors:
        (cm.OUT / "E7_SUMMARY.md").write_text("# E7 stopped: alignment failures\n\n" + "\n".join(errors[:50]) + "\n")
        raise SystemExit(f"{len(errors)} alignment failures")
    cells = pd.DataFrame([r for rs, _ in res for r in rs])
    cells.to_csv(cm.OUT / "e7_cells.csv", index=False)
    gaps = summarise(cells)
    gaps.to_csv(cm.OUT / "e7_gaps.csv", index=False)
    one = cells[cells.fm == "fm1"]
    L = ["# E7: protocol gap on a common evaluation set (exploratory, DEV, stored Study 7 Part (a))", "",
         f"Alignment: the fast scorer reproduces all stored macro-F1, accuracy and BA (stratified + random, 4 FMs) to {TOL:g}.", "",
         "## Size of the common sets (per table × m; identical across FMs)", "",
         "| m | paired: mean rows | paired: min rows | paired: min class count | paired: pairs excluded (class with 0 rows) / 400 | strict: median rows | strict: median min class count | strict: cells usable (≥30 rows, all classes) |",
         "|---|---|---|---|---|---|---|---|"]
    for m, g in one.groupby("mult"):
        L.append(f"| {m} | {g.pair_rows_mean.mean():.0f} | {g.pair_rows_min.min()} | {g.pair_min_class.min()} | "
                 f"{int(g.pair_missing.sum())} | {g.strict_rows.median():.0f} | {g.strict_min_class.median():.0f} | "
                 f"{int(g.strict_ok.sum())}/20 |")
    L += ["", "Gap = stratified − random; mean over tables [bootstrap 95% CI]; Holm over the 4 FMs within metric × m × variant "
          "(`*` p_holm < 0.05). `cur` = frozen per-draw complement; `pair` = rep-matched paired common rows (primary); `strict` = intersection "
          "(only usable cells, n in brackets).", ""]
    for k in METRICS:
        L += [f"## {k}", "", "| FM | variant | m=1 | m=3 | m=5 | m=10 | mean over m |", "|---|---|---|---|---|---|---|"]
        for fm in cm.FMS:
            for v in ("cur", "pair", "pair_minus_cur", "strict", "strict_minus_cur"):
                cs = []
                for m in ("1", "3", "5", "10", "all"):
                    r = gaps[(gaps.variant == v) & (gaps.metric == k) & (gaps.fm == fm) & (gaps.mult == m)]
                    if r.empty:
                        cs.append("—")
                        continue
                    r = r.iloc[0]
                    s = f"{r['mean']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]" + (" *" if r.p_holm < 0.05 else "")
                    if v.startswith("strict"):
                        s += f" (n={int(r.n)})"
                    cs.append(s)
                L.append(f"| {cm.FM_NAMES[fm]} | {v} | " + " | ".join(cs) + " |")
        L.append("")
    (cm.OUT / "E7_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
