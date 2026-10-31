"""Human check of the P1 systematic review: agreement and effect on the headline counts.

EXPLORATORY (not pre-registered). Inputs in data/ (copies, sha256 in ADJUDICATION.md):
  human_check_form.csv  owner's blind coding of the 32-paper random sample
  final_coding.csv      final (LLM, adjudicated) coding of all 161 included papers
  adjudication.csv      proposed adjudication of the 11 papers with a disagreement on
                        category or coverage_forced (confirmed by the owner, 2026-10-29)
Outputs: agreement.csv (this folder); in out/: correction.csv, results.json, hc_numbers.tex.

Run from anywhere:  ../../../../.venv/bin/python hc_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "out"
B = 10_000
SEED = 20261029
VARS = ["category", "coverage_forced", "few_shot", "missed_class_reported", "main_metric"]
METRIC_ALIAS = {"auroc": "AUC", "auc": "AUC", "roc-auc": "AUC"}
GROUP = {"A": "A", "A'": "Ap", "B": "S", "C": "S", "D": "U", "E": "U"}  # S = stated, no labels; U = unstated
GROUPS = ["A", "Ap", "S", "U"]


def norm_metric(x: str) -> str:
    return METRIC_ALIAS.get(str(x).strip().lower(), str(x).strip())


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    cats = np.union1d(a, b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return float("nan") if np.isclose(pe, 1.0) else (po - pe) / (1 - pe)


def boot_ci(a, b, stat, rng):
    n = len(a)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        vals.append(stat(a[i], b[i]))
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load():
    h = pd.read_csv(DATA / "human_check_form.csv", dtype=str)
    f = pd.read_csv(DATA / "final_coding.csv", dtype=str)
    adj = pd.read_csv(DATA / "adjudication.csv", dtype=str).fillna("")
    m = h[["paper_id"] + VARS].merge(f[["paper_id"] + VARS], on="paper_id", suffixes=("_h", "_f"))
    assert len(m) == 32, len(m)
    for s in ("_h", "_f"):
        m["main_metric" + s] = m["main_metric" + s].map(norm_metric)
        for v in VARS:
            m[v + s] = m[v + s].str.strip()
    # Adjudicated codes: proposal where a disagreement existed, else the agreed code.
    m = m.merge(adj, on="paper_id", how="left").fillna("")
    dis = (m.category_h != m.category_f) | (m.coverage_forced_h != m.coverage_forced_f)
    assert set(m.loc[dis, "paper_id"]) == set(adj.paper_id), "adjudication.csv must list every disagreement"
    m["category_a"] = np.where(m.category_adj != "", m.category_adj, m.category_f)
    m["coverage_forced_a"] = np.where(m.coverage_adj != "", m.coverage_adj, m.coverage_forced_f)
    # Alternative adjudication: 'both defensible' / 'codebook ambiguous' cases resolved toward the human.
    m["category_a2"] = np.where(m.alt_category != "", m.alt_category, m.category_a)
    m["coverage_forced_a2"] = np.where(m.alt_coverage != "", m.alt_coverage, m.coverage_forced_a)
    return m, f


def agreement(m, rng):
    rows = []
    pairs = [("human", "final", "_h", "_f", VARS),
             ("human", "adjudicated", "_h", "_a", ["category", "coverage_forced"]),
             ("final", "adjudicated", "_f", "_a", ["category", "coverage_forced"]),
             ("human", "adjudicated_alt", "_h", "_a2", ["category", "coverage_forced"]),
             ("final", "adjudicated_alt", "_f", "_a2", ["category", "coverage_forced"])]
    for x, y, sx, sy, vs in pairs:
        for v in vs:
            a, b = m[v + sx].to_numpy(), m[v + sy].to_numpy()
            k = kappa(a, b)
            raw = float(np.mean(a == b))
            klo, khi = boot_ci(a, b, kappa, rng)
            rlo, rhi = boot_ci(a, b, lambda p, q: float(np.mean(p == q)), rng)
            rows.append(dict(pair=f"{x}_vs_{y}", variable=v, n=len(a), agree=int((a == b).sum()),
                             raw=raw, raw_lo=rlo, raw_hi=rhi, kappa=k, kappa_lo=klo, kappa_hi=khi))
    return pd.DataFrame(rows)


def correct_counts(sample_f, sample_t, frame_counts):
    """Direct confusion-matrix correction: true_g = sum_f N_f * P(true=g | final=f).
    A final group absent from the (bootstrap) sample is left as is (no misclassification assumed)."""
    out = dict.fromkeys(GROUPS, 0.0)
    for fg, nf in frame_counts.items():
        sel = sample_f == fg
        if sel.sum() == 0:
            out[fg] += nf
            continue
        for tg in GROUPS:
            out[tg] += nf * np.mean(sample_t[sel] == tg)
    k = out["A"] + out["Ap"]
    n_stated = k + out["S"]
    return dict(A=out["A"], Ap=out["Ap"], S=out["S"], U=out["U"], K=k, Nstated=n_stated,
                share_stated=k / n_stated if n_stated else float("nan"))


def correction(m, f, rng):
    fs = f[f.few_shot.str.strip() == "yes"]
    frames = {"fewshot": fs, "all": f}
    rows = []
    for ref, col in [("adjudicated", "category_a"), ("adjudicated_alt", "category_a2"), ("human", "category_h")]:
        sf = m.category_f.map(GROUP).to_numpy()
        st = m[col].map(GROUP).to_numpy()
        for fname, fr in frames.items():
            counts = fr.category.str.strip().map(GROUP).value_counts().reindex(GROUPS, fill_value=0).to_dict()
            point = correct_counts(sf, st, counts)
            boots = []
            for _ in range(B):
                i = rng.integers(0, len(sf), len(sf))
                boots.append(correct_counts(sf[i], st[i], counts))
            bt = pd.DataFrame(boots)
            for key in ["A", "Ap", "K", "Nstated", "share_stated"]:
                rows.append(dict(reference=ref, frame=fname, quantity=key, observed=(
                    counts["A"] if key == "A" else counts["Ap"] if key == "Ap" else
                    counts["A"] + counts["Ap"] if key == "K" else
                    counts["A"] + counts["Ap"] + counts["S"] if key == "Nstated" else
                    (counts["A"] + counts["Ap"]) / (counts["A"] + counts["Ap"] + counts["S"])),
                    corrected=point[key], lo=float(np.percentile(bt[key], 2.5)),
                    hi=float(np.percentile(bt[key], 97.5))))
    return pd.DataFrame(rows)


def error_rates(m):
    """Rates of over-/under-assignment of A/A' by the final coding, against each reference,
    with exact (Clopper-Pearson) 95% intervals."""
    def cp(k, n):
        lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
        hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
        return float(lo), float(hi)
    out = {}
    lab = {"A", "A'"}
    fL = m.category_f.isin(lab)
    for ref, col in [("adjudicated", "category_a"), ("adjudicated_alt", "category_a2"), ("human", "category_h")]:
        tL = m[col].isin(lab)
        over_k, over_n = int((fL & ~tL).sum()), int(fL.sum())     # final says L, reference not L
        under_k, under_n = int((~fL & tL).sum()), int((~fL).sum())  # final not L, reference L
        se_n = int(tL.sum()); sp_n = int((~tL).sum())
        se = float((fL & tL).sum() / se_n); sp = float((~fL & ~tL).sum() / sp_n)
        p_obs = float(fL.mean())
        out[ref] = dict(over_k=over_k, over_n=over_n, over_rate=over_k / over_n, over_ci=cp(over_k, over_n),
                        under_k=under_k, under_n=under_n, under_rate=under_k / under_n,
                        under_ci=cp(under_k, under_n), sensitivity=se, specificity=sp)
    return out


def rogan_gladen(p_obs, se, sp):
    d = se + sp - 1
    return float("nan") if d <= 0 else min(1.0, max(0.0, (p_obs + sp - 1) / d))


def tex_num(x, d=2):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$")


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    m, f = load()
    ag = agreement(m, rng)
    ag.to_csv(HERE / "agreement.csv", index=False, float_format="%.4f")
    cor = correction(m, f, rng)
    cor.to_csv(OUT / "correction.csv", index=False, float_format="%.4f")
    er = error_rates(m)
    # Rogan-Gladen on all 161 papers (binary: A/A' vs other), human as reference.
    p_all = float(f.category.str.strip().isin({"A", "A'"}).mean())
    rg = {ref: rogan_gladen(p_all, er[ref]["sensitivity"], er[ref]["specificity"]) for ref in er}
    res = dict(n_sample=len(m), verdicts=m.loc[m.verdict != "", ["paper_id", "verdict"]].values.tolist(),
               error_rates=er, rogan_gladen_all161=rg, p_obs_all161=p_all)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    write_tex(ag, cor, er, m)
    print(ag.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(cor.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(json.dumps(er, indent=1)); print("RG all161:", rg, "p_obs", p_all)


def write_tex(ag, cor, er, m):
    L = ["% Generated by paper/p1/exploratory/human_check/hc_analysis.py; do not edit.",
         "% EXPLORATORY (not pre-registered): human check of the systematic review (32-paper random sample).",
         "% Adjudication verdicts were confirmed by the owner on 2026-10-29 (ADJUDICATION.md).",
         "% \\ExHc<Var><Pair><Stat>: Var Cat/Cov/Fs/Mc/Met; Pair HF (human vs final), HA (human vs adjudicated),",
         "% FA (final vs adjudicated), HAalt/FAalt (alternative adjudication); Stat Kappa/KappaLo/KappaHi/Raw/RawLo/RawHi/Agree.",
         "% \\ExHcCorr<Ref><Qty>[Lo|Hi]: corrected few-shot counts; Ref Adj/AdjAlt/Hum; Qty K (A+A'), N (stated), Pct, A, Ap."]
    def src_of(name):
        """Per-macro provenance comment (m12-10)."""
        if name.startswith("Corr"):
            return "out/correction.csv (corrected few-shot counts and shares; bootstrap 95% CI over the sample)"
        if name.startswith(("Over", "Under")):
            return "error rates of the final coding against the reference (Clopper-Pearson 95% upper bound for Hi)"
        if name.startswith("Worst"):
            return "worst case: upper 95% bound of the over-assignment rate applied to the few-shot A/A' papers"
        if name.startswith("Verdict"):
            return "data/adjudication.csv (adjudication verdicts, counts)"
        if name in ("N", "NDisagree", "NDisCat", "NDisCov"):
            return "data/human_check_form.csv vs data/final_coding.csv (sample size and disagreement counts)"
        return "agreement.csv (raw agreement and Cohen's kappa with bootstrap 95% CI)"

    def put(name, val):
        L.append(f"\\newcommand{{\\ExHc{name}}}{{{val}}} % EXPLORATORY; source: hc_analysis.py, {src_of(name)}")
    put("N", len(m))
    put("NDisagree", int(((m.category_h != m.category_f) | (m.coverage_forced_h != m.coverage_forced_f)).sum()))
    put("NDisCat", int((m.category_h != m.category_f).sum()))
    put("NDisCov", int((m.coverage_forced_h != m.coverage_forced_f).sum()))
    for vd, key in [("FinalRight", "final right"), ("HumanRight", "human right"),
                    ("BothDef", "both defensible"), ("Ambig", "codebook ambiguous")]:
        put(f"Verdict{vd}", int((m.verdict == key).sum()))
    vn = {"category": "Cat", "coverage_forced": "Cov", "few_shot": "Fs", "missed_class_reported": "Mc", "main_metric": "Met"}
    pn = {"human_vs_final": "HF", "human_vs_adjudicated": "HA", "final_vs_adjudicated": "FA",
          "human_vs_adjudicated_alt": "HAalt", "final_vs_adjudicated_alt": "FAalt"}
    for _, r in ag.iterrows():
        p = vn[r.variable] + pn[r.pair]
        put(p + "Agree", int(r.agree))
        for s in ["raw", "raw_lo", "raw_hi", "kappa", "kappa_lo", "kappa_hi"]:
            nm = {"raw": "Raw", "raw_lo": "RawLo", "raw_hi": "RawHi", "kappa": "Kappa",
                  "kappa_lo": "KappaLo", "kappa_hi": "KappaHi"}[s]
            put(p + nm, "n/a" if np.isnan(r[s]) else tex_num(r[s]))
    rn = {"adjudicated": "Adj", "adjudicated_alt": "AdjAlt", "human": "Hum"}
    qn = {"K": "K", "Nstated": "N", "share_stated": "Pct", "A": "A", "Ap": "Ap"}
    for _, r in cor[cor.frame == "fewshot"].iterrows():
        base = "Corr" + rn[r.reference] + qn[r.quantity]
        if r.quantity == "share_stated":
            put(base, f"{100 * r.corrected:.0f}\\%"); put(base + "Lo", f"{100 * r.lo:.0f}\\%")
            put(base + "Hi", f"{100 * r.hi:.0f}\\%")
        else:
            put(base, f"{r.corrected:.0f}"); put(base + "Lo", f"{r.lo:.0f}"); put(base + "Hi", f"{r.hi:.0f}")
    for ref in ["adjudicated", "human"]:
        e = er[ref]
        put(f"Over{rn[ref]}K", e["over_k"]); put(f"Over{rn[ref]}N", e["over_n"])
        put(f"Over{rn[ref]}Hi", f"{100 * e['over_ci'][1]:.0f}\\%")
        put(f"Under{rn[ref]}K", e["under_k"]); put(f"Under{rn[ref]}N", e["under_n"])
        put(f"Under{rn[ref]}Hi", f"{100 * e['under_ci'][1]:.0f}\\%")
    # Worst case allowed by the sample when no over-assignment is observed (adjudicated reference):
    # the upper 95% bound of the over-assignment rate applied to the 73 few-shot A/A' papers,
    # with the misassigned papers moved to B/C (share falls most) or to D/E (they leave the frame).
    obs = cor[(cor.reference == "adjudicated") & (cor.frame == "fewshot")].set_index("quantity").observed
    k0, n0 = obs["K"], obs["Nstated"]
    ub = er["adjudicated"]["over_ci"][1]
    k_lo = k0 * (1 - ub)
    put("WorstKLo", f"{k_lo:.0f}")
    put("WorstPctToS", f"{100 * k_lo / n0:.0f}\\%")
    put("WorstPctToU", f"{100 * k_lo / (n0 - k0 * ub):.0f}\\%")
    (OUT / "hc_numbers.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
