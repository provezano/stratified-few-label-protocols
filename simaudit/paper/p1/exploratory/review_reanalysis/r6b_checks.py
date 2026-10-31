"""Round-6 items left over from the first pass (EXPLORATORY; read-only inputs).

(1) m6-6, Study 12 S-3: a noise baseline matched to the protocol contrast. The protocol flip rate
    compares 5 stratified with 20 random repetitions; the reported baseline compares random reps
    0-9 with 10-19. Here the baseline compares random reps 0-4 with 5-19 (5 against 15), and, as a
    check, the mean over the four splits of one block of five reps against the other fifteen. The
    flip rule, the learners and the cluster bootstrap are those of simaudit.analysis.study12.
(2) m6-8, human check of the systematic review (32-paper sample): the category counts of the
    blind human coding next to the final codes, overall and in the frame of the headline
    (few-shot papers that describe their draw, i.e. category A, A', B or C), each by its own coding;
    and a recount of the adjudication (exploratory/human_check/data/adjudication.csv) restricted to
    the category disagreements: verdicts, and for "final right" the primary source of the evidence
    (appendix if the decisive source names an appendix, else the official code if it names one, else
    the main text). The recount must match \\ExHcNDisCat and \\ExHcVerdict* of hc_numbers.tex.
(3) m6-9, Study 9: the H3b statistic (Kendall tau_b), parsed from reports/STUDY9_T1PRIME_REPORT.md;
    the Holm p-values of the other eight tests already exist as \\SnineHolm* in numbers_p1.tex.
Writes out/r6b_flip.csv, out/r6b_human.csv, out/rv_r6b.tex.
Run from simaudit/: nice -n 15 .venv/bin/python paper/p1/exploratory/review_reanalysis/r6b_checks.py
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from rv_common import OUT, Macros, fmt
from simaudit.analysis import study12 as a12
from simaudit.settings import PROJECT_ROOT, settings

LEARNERS = a12.FMS + ("fm1_bal", "fm5_bal")
HC = OUT.parents[1] / "human_check" / "data"  # copies of research_notes/.../coding (identical)
STATED = {"A", "A'", "B", "C"}


def flip_split(runs: pd.DataFrame, block: range, metric: str) -> dict:
    """Protocol flips (S vs R, as in a12.flip_rate) and noise flips (random block vs the rest)."""
    sub = runs[runs.learner.isin(LEARNERS)]
    rnd = sub[sub.strategy == "random"].copy()
    rnd["strategy"] = np.where(rnd.rep.isin(block), "half_a", "half_b")
    both = pd.concat([sub[sub.strategy.isin(["stratified", "random"])], rnd])
    cells = both.pivot_table(index=["dataset", "mult", "learner"], columns="strategy", values=metric)
    j = a12._flips(cells, "stratified", "random").join(
        a12._flips(cells, "half_a", "half_b"), rsuffix="_noise")
    f, d = j.flips.to_numpy(float), j.decisive.to_numpy(float)
    fn, dn = j.flips_noise.to_numpy(float), j.decisive_noise.to_numpy(float)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(j), size=(2000, len(j)))
    diff = f[idx].sum(1) / np.maximum(d[idx].sum(1), 1) - fn[idx].sum(1) / np.maximum(dn[idx].sum(1), 1)
    rate, base = f.sum() / d.sum(), fn.sum() / dn.sum()
    return dict(metric=metric, block=f"{block.start}-{block.stop - 1}", rate=rate,
                noise=base, noise_lo=a12._cluster_ci(fn, dn)[0], noise_hi=a12._cluster_ci(fn, dn)[1],
                noise_flips=int(fn.sum()), noise_decisive=int(dn.sum()), diff=rate - base,
                diff_lo=float(np.quantile(diff, 0.025)), diff_hi=float(np.quantile(diff, 0.975)))


def flips(m: Macros) -> None:
    runs = a12.load_runs(settings.results_dir / "study12")
    ref = a12.flip_rate(runs, LEARNERS)  # reproduces \StwelveFlipF / \StwelveFlipFNoise
    assert (ref["flips"], ref["decisive"]) == (410, 1184), (ref["flips"], ref["decisive"])
    rows = [flip_split(runs, range(b, b + 5), metric)
            for metric in ("macro_f1", "accuracy") for b in (0, 5, 10, 15)]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r6b_flip.csv", index=False)
    src = "r6b_checks.py (1); results/study12"
    for metric, w in (("macro_f1", "F"), ("accuracy", "Acc")):
        x = df[df.metric == metric]
        r0 = x[x.block == "0-4"].iloc[0]
        m.put(f"RsixbFlip{w}Matched", f"{100 * r0.noise:.1f}\\%", src + " noise rate, reps 0-4 vs 5-19")
        m.put(f"RsixbFlip{w}MatchedLo", f"{100 * r0.noise_lo:.1f}\\%", src)
        m.put(f"RsixbFlip{w}MatchedHi", f"{100 * r0.noise_hi:.1f}\\%", src)
        m.put(f"RsixbFlip{w}MatchedDiff", fmt(r0["diff"], sign=True), src + " protocol minus matched noise")
        m.put(f"RsixbFlip{w}MatchedDiffLo", fmt(r0.diff_lo, sign=True), src)
        m.put(f"RsixbFlip{w}MatchedDiffHi", fmt(r0.diff_hi, sign=True), src)
        m.put(f"RsixbFlip{w}BlocksMin", f"{100 * x.noise.min():.1f}\\%", src + " four blocks of 5 vs 15")
        m.put(f"RsixbFlip{w}BlocksMax", f"{100 * x.noise.max():.1f}\\%", src)
    print(df.round(3).to_string())


def human(m: Macros) -> None:
    h = pd.read_csv(HC / "human_check_form.csv", dtype=str)
    f = pd.read_csv(HC / "final_coding.csv", dtype=str)
    j = h.merge(f, on="paper_id", suffixes=("_h", "_f"))
    assert len(j) == len(h) == 32
    cats = ["A", "A'", "B", "C", "D", "E"]
    rows = []
    for who in ("h", "f"):
        cat, fs = j[f"category_{who}"].str.strip(), j[f"few_shot_{who}"].str.strip()
        frame = (fs == "yes") & cat.isin(STATED)
        row = dict(coding="human" if who == "h" else "final")
        row.update({c: int((cat == c).sum()) for c in cats})
        row.update(fs_stated=int(frame.sum()), fs_A=int((frame & (cat == "A")).sum()),
                   fs_Ap=int((frame & (cat == "A'")).sum()))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r6b_human.csv", index=False)
    src = "r6b_checks.py (2); exploratory/human_check/data (human_check_form.csv, final_coding.csv)"
    names = {"A": "A", "A'": "Ap", "B": "B", "C": "C", "D": "D", "E": "E",
             "fs_stated": "FsStated", "fs_A": "FsA", "fs_Ap": "FsAp"}
    for _, r in df.iterrows():
        w = "Hum" if r.coding == "human" else "Fin"
        for k, n in names.items():
            m.put(f"RsixbHc{w}{n}", str(int(r[k])), src)
    print(df.to_string())
    adj = pd.read_csv(HC / "adjudication.csv", dtype=str).fillna("")
    k = j.merge(adj, on="paper_id", how="left").fillna("")
    cat = k[k.category_h.str.strip() != k.category_f.str.strip()]
    right = cat[cat.verdict == "final right"]

    def source(s: str) -> str:
        return "app" if "App." in s else "code" if "official code" in s else "main"

    src_counts = right.decisive_source.map(source).value_counts()
    hc = (OUT.parents[1] / "human_check" / "out" / "hc_numbers.tex").read_text(encoding="utf-8")
    hcv = dict(re.findall(r"\\newcommand\{\\ExHc(\w+)\}\{([^}]*)\}", hc))
    n_both = int((cat.verdict == "both defensible").sum())
    n_amb = int((cat.verdict == "codebook ambiguous").sum())
    assert len(cat) == int(hcv["NDisCat"]), (len(cat), hcv["NDisCat"])
    assert len(right) == len(cat) - int(hcv["VerdictBothDef"]) - int(hcv["VerdictAmbig"])
    assert (n_both, n_amb) == (int(hcv["VerdictBothDef"]), int(hcv["VerdictAmbig"]))
    src = "r6b_checks.py (2); exploratory/human_check/data/adjudication.csv, category disagreements"
    m.put("RsixbHcCatDis", str(len(cat)), src)
    m.put("RsixbHcCatFinalRight", str(len(right)), src + ", verdict final right")
    for key, name in (("app", "App"), ("code", "Code"), ("main", "Main")):
        m.put(f"RsixbHcCatSrc{name}", str(int(src_counts.get(key, 0))), src + f", primary source {key}")
    print("category disagreements", len(cat), "final right", len(right), dict(src_counts),
          "both defensible", n_both, "ambiguous", n_amb)


def study9_h3b(m: Macros) -> None:
    text = (PROJECT_ROOT / "reports" / "STUDY9_T1PRIME_REPORT.md").read_text(encoding="utf-8")
    hit = re.search(r"\| H3b:[^|]*\| τ_b = ([−\-]?\d+\.\d+) \|", text)
    assert hit, "H3b row not found"
    tau = float(hit.group(1).replace("−", "-"))
    m.put("RsixbSnineHThreebTau", fmt(tau, nd=2), "reports/STUDY9_T1PRIME_REPORT.md Holm table, H3b")
    print("H3b tau_b", tau)


def main() -> None:
    m = Macros("r6b")
    flips(m)
    human(m)
    study9_h3b(m)
    m.write()


if __name__ == "__main__":
    main()
