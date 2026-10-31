"""E8 (exploratory, R3-W1): schema-aware sensitivity of the coverage term.

The FMs (and our wrapper) output only the classes present in the context (L0). A
learner that knows the label schema but has no example of a class could at best
guess it. From the stored Study 7 Part (a) probabilities we emulate such a learner:

- **argmax-α**: every class absent from the context gets the score α
  (α ∈ {0.05, 0.1, 0.2}); seen classes keep their stored probability; the row is
  renormalised and the argmax is taken over all C classes. An absent class is
  predicted exactly where max_seen p < α (ties to the seen class are impossible
  except at p = α). With a single-class context the seen class has p = 1, so
  nothing changes there (in particular for every binary draw that misses a class).
- **guess-α** (extra, stochastic): the renormalised absent mass α|A|/(1 + α|A|) is
  used as the probability of predicting a uniformly chosen absent class for each
  row (fixed seed), otherwise the seen argmax. This is the most a label-free guess
  can buy without information about the absent class.

Stratified draws always cover every class and are unchanged. Re-scored on the frozen
complement with macro-F1 over all C classes. Reported per FM and m: the gap
(stratified − random) and its coverage part, observed as in reports/theory_checks.py:
mean over random draws of (MF^S − MF^R_draw)·1[draw misses a class].

Native behaviour of each FM for classes absent from the context (code read only,
nothing executed) is documented in out/E8_SUMMARY.md.

Writes out/e8_cells.csv, out/e8_gaps.csv, out/E8_SUMMARY.md.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as cm  # noqa: E402

ALPHAS = (0.05, 0.1, 0.2)
VARIANTS = ["l0"] + [f"argmax{a}" for a in ALPHAS] + [f"guess{a}" for a in ALPHAS]


def run(job: tuple[int, str]) -> list[dict]:
    did, fm = job
    y = cm.dev_data(did).y
    n = len(y)
    df = cm.load_a(did, fm, proba=True)
    C = int(df.n_classes.iloc[0])
    out = []
    for r in df.itertuples():
        P = np.asarray(r.proba, dtype=np.float64).reshape(n, C)
        lab = np.asarray(r.labeled, dtype=np.int64)
        mask = np.ones(n, bool)
        mask[lab] = False
        seen = np.zeros(C, bool)
        seen[np.unique(y[lab])] = True
        absent = np.flatnonzero(~seen)
        yt, Pu = y[mask], P[mask]
        rec = dict(dataset=did, fm=fm, C=C, strategy=r.strategy, mult=int(r.mult), rep=int(r.rep),
                   miss=bool(len(absent) > 0), n_absent=len(absent))
        base = Pu.argmax(1)
        rec["l0"] = cm.fast_scores(yt, base, C)[0]
        rng = np.random.default_rng([int(r.seed) % (2**32), 8])
        for a in ALPHAS:
            if len(absent) == 0:
                rec[f"argmax{a}"] = rec[f"guess{a}"] = rec["l0"]
                rec[f"share_absent_pred{a}"] = 0.0
                continue
            Q = Pu.copy()
            Q[:, absent] = a
            pred = Q.argmax(1)
            rec[f"argmax{a}"] = cm.fast_scores(yt, pred, C)[0]
            rec[f"share_absent_pred{a}"] = float(np.isin(pred, absent).mean())
            mass = a * len(absent) / (1 + a * len(absent))
            g = base.copy()
            hit = rng.random(len(yt)) < mass
            g[hit] = rng.choice(absent, size=int(hit.sum()))
            rec[f"guess{a}"] = cm.fast_scores(yt, g, C)[0]
        out.append(rec)
    return out


def cells(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (did, fm, m), g in draws.groupby(["dataset", "fm", "mult"]):
        s, r = g[g.strategy == "stratified"], g[g.strategy == "random"]
        rec = dict(dataset=did, fm=fm, mult=m, C=int(g.C.iloc[0]), p_miss=float(r.miss.mean()))
        for v in VARIANTS:
            ms = s[v].mean()
            rec[f"gap_{v}"] = ms - r[v].mean()
            rec[f"cov_{v}"] = float(((ms - r[v]) * r.miss).mean())
        for a in ALPHAS:
            rec[f"share_absent_pred{a}"] = float(r.loc[r.miss, f"share_absent_pred{a}"].mean()) if r.miss.any() else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def summary(c: pd.DataFrame) -> pd.DataFrame:
    out = []
    for subset, sel in (("all", np.ones(len(c), bool)), ("binary", c.C == 2), ("multiclass", c.C > 2)):
        for (fm, m), g in c[sel].groupby(["fm", "mult"]):
            for v in VARIANTS:
                for part in ("gap", "cov"):
                    x = g[f"{part}_{v}"].to_numpy()
                    lo, hi = cm.bootstrap_ci(x)
                    d = x - g[f"{part}_l0"].to_numpy()
                    dlo, dhi = cm.bootstrap_ci(d)
                    out.append(dict(subset=subset, fm=fm, mult=m, variant=v, part=part, n=len(x), mean=x.mean(),
                                    ci_lo=lo, ci_hi=hi, change=d.mean(), change_lo=dlo, change_hi=dhi,
                                    change_p=cm.wilcoxon_p(d)))
    return pd.DataFrame(out)


NATIVE = """## Native behaviour of the FMs for classes absent from the context (code read, nothing executed)

- **Project wrapper** (`src/simaudit/selection/strategies.py::predict_with_budget`): the labels are re-encoded to the
  seen classes (`np.unique(y_lab, return_inverse=True)`); with one seen class the FM is not called and every row
  gets probability 1 for that class; otherwise the FM's columns are scattered into a C-column matrix and the absent
  columns are 0 (`expand_proba`). This only maps indices; it imposes nothing the models would not do themselves.
- **TabPFN v2 and v3.5** (`.venv/.../tabpfn/classifier.py` `fit`, `tabpfn/preprocessing/label_encoder.py`, tabpfn 9.2.0):
  `TabPFNLabelEncoder` wraps sklearn's `LabelEncoder`; `classes_ = encoder.classes_` and `n_classes_ = len(classes_)`,
  i.e. the classes present in the fitted `y`. `predict_proba` returns `n_classes_` columns (outputs are sliced to
  `[:n_classes_]`). There is no argument to declare extra classes; an absent class cannot be emitted.
- **TabICL** (`tabicl/_sklearn/classifier.py` `fit`, tabicl 2.2.0): `self.y_encoder_ = LabelEncoder()`,
  `classes_ = y_encoder_.classes_`, `n_classes_ = len(classes_)`: the same, no schema argument.
- **Kumo-Tabular** (`sdm/models/kumo/relational/model.py`, `sdm/models/kumo/tabular/batching.py`): the class set is
  `y_context.categorical.categories[0]` and `num_classes = len(categories)`; the output has one column per category.
  Our wrapper builds the categorical from the context labels (query rows carry a placeholder equal to a context
  label), so only seen classes exist. The API derives classes from the categorical dtype, so declaring an unused
  category might be possible; whether the model would ever predict a class with no context row is untested and
  would need a new fit (owner B, not done here).
- Hence L0 (predict only context classes) is the native behaviour of all four evaluated learners, not a restriction
  added by the study.
"""


def main() -> None:
    jobs = [(d, fm) for d in cm.dev_ids() for fm in cm.FMS]
    with ProcessPoolExecutor(3) as pool:
        draws = pd.DataFrame([r for rs in pool.map(run, jobs) for r in rs])
    c = cells(draws)
    c.to_csv(cm.OUT / "e8_cells.csv", index=False)
    s = summary(c)
    s.to_csv(cm.OUT / "e8_gaps.csv", index=False)
    rnd = draws[draws.strategy == "random"]
    L = ["# E8: schema-aware sensitivity of the coverage term (exploratory, DEV, stored Study 7 Part (a))", "",
         f"Random draws that miss a class: {int(rnd.miss.sum())} of {len(rnd)} "
         f"(binary: {int(rnd[rnd.C == 2].miss.sum())}, all of them single-class contexts, unchanged by argmax-α).", "",
         "Share of evaluation rows predicted as an absent class (missed random draws, mean over cells):", ""]
    for a in ALPHAS:
        L.append(f"- α = {a}: argmax {c[f'share_absent_pred{a}'].mean():.4f} (all m), "
                 f"m=1 multiclass {c[(c.mult == 1) & (c.C > 2)][f'share_absent_pred{a}'].mean():.4f}")
    L += ["", "Mean over tables [bootstrap 95% CI]; change = variant − L0 (Wilcoxon p unadjusted).", ""]
    for subset in ("all", "multiclass"):
        for m in (1, 3):
            L += [f"## {subset}, m = {m}", "", "| FM | variant | gap | coverage part | change in coverage part (p) |", "|---|---|---|---|---|"]
            for fm in cm.FMS:
                for v in VARIANTS:
                    g = s[(s.subset == subset) & (s.fm == fm) & (s.mult == m) & (s.variant == v)]
                    gp, cv = g[g.part == "gap"].iloc[0], g[g.part == "cov"].iloc[0]
                    L.append(f"| {cm.FM_NAMES[fm]} | {v} | {gp['mean']:+.3f} [{gp.ci_lo:+.3f}, {gp.ci_hi:+.3f}] | "
                             f"{cv['mean']:+.3f} [{cv.ci_lo:+.3f}, {cv.ci_hi:+.3f}] | {cv.change:+.4f} [{cv.change_lo:+.4f}, {cv.change_hi:+.4f}] (p={cv.change_p:.2g}) |")
            L.append("")
    L.append(NATIVE)
    (cm.OUT / "E8_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
