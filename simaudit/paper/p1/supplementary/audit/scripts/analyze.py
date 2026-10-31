"""Analysis for the systematic review (protocol section 9). Every number in REPORT.md comes from this script.
Usage: python -I scripts/analyze.py  (run from the review folder)
Writes analysis_output.json and analysis_output.md."""
import json, math, os, sys
from collections import Counter
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def share(k, n):
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "pct": round(100 * k / n, 1) if n else None,
            "wilson_lo": round(100 * lo, 1), "wilson_hi": round(100 * hi, 1)}


def kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"n": n, "raw_agreement": round(po, 3), "kappa": round(k, 3) if k == k else None}


def norm_cat(x):
    x = str(x).strip().replace("’", "'").replace("′", "'")
    return x.split("/")[0].split(" ")[0] if x else x


def main():
    out = {}
    # PRISMA
    idc = json.load(open(P("screening", "identification_counts.json")))
    s1 = pd.read_csv(P("screening", "stage1_decisions_all.csv")).fillna("")
    s2 = pd.read_csv(P("screening", "stage2_decisions.csv")).fillna("")
    out["prisma"] = {
        "identification": idc,
        "stage1_screened": len(s1),
        "stage1_excluded": int((s1.decision == "exclude").sum()),
        "stage1_excluded_by_reason": s1[s1.decision == "exclude"].reason.value_counts().to_dict(),
        "stage1_included": int((s1.decision == "include").sum()),
        "fulltext_sought": len(s2),
        "fulltext_not_accessible": int((s2.decision == "not_accessible").sum()),
        "fulltext_assessed": int((s2.decision != "not_accessible").sum()),
        "fulltext_excluded": int((s2.decision == "exclude").sum()),
        "fulltext_excluded_by_reason": s2[s2.decision == "exclude"].reason.value_counts().to_dict(),
        "included": int((s2.decision == "include").sum()),
        "included_from_prior_audit_only": int(((s2.decision == "include") & (s2.sources == "prior_audit")).sum()),
    }

    # Agreement
    c1 = pd.read_csv(P("coding", "coder1_coding.csv")).fillna("")
    c2 = pd.read_csv(P("coding", "coder2_coding.csv")).fillna("")
    m = c1.merge(c2, on="paper_id", suffixes=("_1", "_2"))
    agr = {}
    for v in ["category", "coverage_forced", "few_shot", "missed_class_reported", "budget_unit", "main_metric"]:
        f = norm_cat if v == "category" else (lambda x: str(x).strip().lower())
        agr[v] = kappa(m[v + "_1"].map(f), m[v + "_2"].map(f))
    out["agreement"] = agr
    out["n_coded_by_both"] = len(m)

    # Final coding
    fc = pd.read_csv(P("coding", "final_coding.csv")).fillna("")
    fc["cat"] = fc.category.map(norm_cat)
    fc["forced"] = fc.cat.isin(["A", "A'"])
    fc["fs"] = fc.few_shot.str.lower().eq("yes")
    meta = s2[["rec_id", "sources", "year", "venue_type"]].rename(columns={"rec_id": "paper_id"})
    fc = fc.merge(meta, on="paper_id", how="left")
    fc["search_found"] = fc.sources != "prior_audit"

    def block(df):
        n = len(df)
        fs = df[df.fs]
        return {
            "n_reduced_label": n,
            "n_few_shot": len(fs),
            "category_counts_all": df.cat.value_counts().to_dict(),
            "category_counts_few_shot": fs.cat.value_counts().to_dict(),
            "AorAprime_few_shot": share(fs.forced.sum(), len(fs)),
            "AorAprime_all": share(df.forced.sum(), n),
            "A_only_few_shot": share((fs.cat == "A").sum(), len(fs)),
            "A_only_all": share((df.cat == "A").sum(), n),
            "coverage_forced_yes_few_shot": share((fs.coverage_forced.str.lower() == "yes").sum(), len(fs)),
            "coverage_forced_yes_all": share((df.coverage_forced.str.lower() == "yes").sum(), n),
            "coverage_forced_counts_all": df.coverage_forced.str.lower().value_counts().to_dict(),
            "missed_class_reported_all": share((df.missed_class_reported.str.lower() == "yes").sum(), n),
            "missed_class_reported_few_shot": share((fs.missed_class_reported.str.lower() == "yes").sum(), len(fs)),
            # among papers whose draw is stated (exclude D/E): share that uses the labels
            "AorAprime_among_stated_draw_few_shot": share(fs.forced.sum(), fs.cat.isin(["A", "A'", "B", "C"]).sum()),
            "AorAprime_among_stated_draw_all": share(df.forced.sum(), df.cat.isin(["A", "A'", "B", "C"]).sum()),
        }

    out["main"] = block(fc)
    out["sensitivity_search_only"] = block(fc[fc.search_found])
    # Post-hoc sensitivity analyses (not in the protocol; disclosed as post hoc in REPORT.md)
    sl = pd.read_csv(P("coding", "sensitivity_lists.csv"), dtype=str)
    anom = set(sl[sl.list == "anomaly_or_PU"].paper_id)
    inh = set(sl[sl.list == "inherited_A"].paper_id)
    out["posthoc_excluding_anomaly_PU"] = block(fc[~fc.paper_id.isin(anom)])
    fi = fc.copy()
    fi.loc[fi.paper_id.isin(inh), "cat"] = "A"
    fi["forced"] = fi.cat.isin(["A", "A'"])
    out["posthoc_inherited_recoded_A"] = block(fi)

    # Metrics
    def metric_list(s):
        return [x.strip().lower() for x in str(s).split(";") if x.strip()]
    any_use = Counter()
    for s in fc.metrics:
        for x in set(metric_list(s)):
            any_use[x] += 1
    out["metrics_any_use"] = {k: share(v, len(fc)) for k, v in any_use.most_common()}
    out["main_metric"] = {k: share(v, len(fc)) for k, v in fc.main_metric.str.lower().value_counts().items()}
    fsd = fc[fc.fs]
    any_fs = Counter()
    for s in fsd.metrics:
        for x in set(metric_list(s)):
            any_fs[x] += 1
    out["metrics_any_use_few_shot"] = {k: share(v, len(fsd)) for k, v in any_fs.most_common()}
    out["main_metric_few_shot"] = {k: share(v, len(fsd)) for k, v in fsd.main_metric.str.lower().value_counts().items()}
    out["unseen_class_scoring_stated"] = fc[~fc.unseen_class_scoring.str.lower().isin(["", "not stated", "none"])][
        ["paper_id", "unseen_class_scoring"]].to_dict("records")
    out["budget_unit"] = fc.budget_unit.str.lower().value_counts().to_dict()
    out["n_seeds_stated"] = share((~fc.n_seeds.astype(str).str.lower().isin(["", "not stated"])).sum(), len(fc))

    # Year and venue type
    fc["year_bin"] = pd.cut(pd.to_numeric(fc.year, errors="coerce"), [2018, 2021, 2023, 2024, 2025, 2026],
                            labels=["2019-21", "2022-23", "2024", "2025", "2026"])
    by = {}
    for col in ["year_bin", "venue_type"]:
        by[col] = {}
        for g, df in fc.groupby(col, observed=True):
            fs = df[df.fs]
            by[col][str(g)] = {"n": len(df), "AorAprime_all": share(df.forced.sum(), len(df)),
                               "n_few_shot": len(fs), "AorAprime_few_shot": share(fs.forced.sum(), len(fs))}
    out["breakdown"] = by

    json.dump(out, open(P("analysis_output.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
