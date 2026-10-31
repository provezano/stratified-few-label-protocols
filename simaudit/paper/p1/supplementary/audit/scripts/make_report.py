"""Generate REPORT.md from analysis_output.json and the CSVs. Numbers are never typed by hand:
the narrative uses placeholders filled from the analysis output.
Run after analyze.py:  python -I scripts/make_report.py"""
import json, os, re
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)
o = json.load(open(P("analysis_output.json")))
pr, ag, mn = o["prisma"], o["agreement"], o["main"]
idc = pr["identification"]


def s(d):
    return f"{d['k']}/{d['n']} = {d['pct']}% (Wilson 95% CI: {d['wilson_lo']}–{d['wilson_hi']}%)"


def sp(d):
    return f"{d['pct']:.0f}%"


fin = pd.read_csv(P("coding", "final_coding.csv"), dtype=str).fillna("")
inc = pd.read_csv(P("coding", "included_papers.csv"), dtype=str).fillna("")
s2 = pd.read_csv(P("screening", "stage2_decisions.csv"), dtype=str).fillna("")
adj = pd.concat([pd.read_csv(P("coding", f), dtype=str) for f in sorted(os.listdir(P("coding"))) if f.startswith("adjudication_log_")]).fillna("")
adj_main = adj[~adj.reason.str.startswith("POST-CHECK")]
cmp_old = pd.read_csv(P("coding", "comparison_old_audit.csv"), dtype=str).fillna("")
hs = pd.read_csv(P("coding", "human_check_sample.csv"), dtype=str)
na = s2[s2.decision == "not_accessible"]

# worst/best case for the not-accessible records (all reduced-label studies; none / all A or A')
k_all, n_all, n_na = mn["AorAprime_all"]["k"], mn["AorAprime_all"]["n"], len(na)
worst = 100 * k_all / (n_all + n_na)
best = 100 * (k_all + n_na) / (n_all + n_na)

V = dict(
    arx_raw=idc["arxiv_raw_hits"], arx_u=idc["arxiv_unique"], s2_raw=idc["s2_raw_hits"], s2_in=idc["s2_in_scope"],
    oa_raw=idc["openalex_raw_hits"], oa_in=idc["openalex_in_scope"], sb_raw=idc["snowball_raw_citations"],
    sb_term=idc["snowball_with_L_term_in_window"], sb_in=idc["snowball_in_scope"], pa=idc["prior_audit_records"],
    before=idc["records_before_dedup"], after=idc["records_after_dedup"], dup=idc["duplicates_removed"],
    pa_only=idc["only_prior_audit"], scr=pr["stage1_screened"], ex1=pr["stage1_excluded"],
    ex1r="; ".join(f"{k} {v}" for k, v in pr["stage1_excluded_by_reason"].items()), in1=pr["stage1_included"],
    ft=pr["fulltext_sought"], na=pr["fulltext_not_accessible"], fta=pr["fulltext_assessed"], ex2=pr["fulltext_excluded"],
    ex2r="; ".join(f"{k} {v}" for k, v in pr["fulltext_excluded_by_reason"].items()), inc=pr["included"],
    inc_pa=pr["included_from_prior_audit_only"], inc_search=pr["included"] - pr["included_from_prior_audit_only"],
    nfs=mn["n_few_shot"],
    k_cat=ag["category"]["kappa"], a_cat=ag["category"]["raw_agreement"], k_cov=ag["coverage_forced"]["kappa"],
    a_cov=ag["coverage_forced"]["raw_agreement"], k_fs=ag["few_shot"]["kappa"], a_fs=ag["few_shot"]["raw_agreement"],
    a_mc=ag["missed_class_reported"]["raw_agreement"], k_bu=ag["budget_unit"]["kappa"], a_bu=ag["budget_unit"]["raw_agreement"],
    k_mm=ag["main_metric"]["kappa"], a_mm=ag["main_metric"]["raw_agreement"],
    n_dis=len(pd.read_csv(P("coding", "disagreements.csv"))), n_dis_p=pd.read_csv(P("coding", "disagreements.csv")).paper_id.nunique(),
    adj_c1=int((adj_main.final == adj_main.coder1).sum()), adj_c2=int((adj_main.final == adj_main.coder2).sum()),
    adj_nei=int(((adj_main.final != adj_main.coder1) & (adj_main.final != adj_main.coder2)).sum()),
    fs_forced=s(mn["AorAprime_few_shot"]), all_forced=s(mn["AorAprime_all"]), fs_A=s(mn["A_only_few_shot"]), all_A=s(mn["A_only_all"]),
    fs_cov=s(mn["coverage_forced_yes_few_shot"]), all_cov=s(mn["coverage_forced_yes_all"]),
    fs_stated=s(mn["AorAprime_among_stated_draw_few_shot"]), all_stated=s(mn["AorAprime_among_stated_draw_all"]),
    mc_all=s(mn["missed_class_reported_all"]), mc_fs=s(mn["missed_class_reported_few_shot"]),
    fs_forced_p=sp(mn["AorAprime_few_shot"]), fs_stated_p=sp(mn["AorAprime_among_stated_draw_few_shot"]),
    fs_lo=mn["AorAprime_few_shot"]["wilson_lo"], fs_hi=mn["AorAprime_few_shot"]["wilson_hi"],
    fs_k=mn["AorAprime_few_shot"]["k"], fs_stated_n=mn["AorAprime_among_stated_draw_few_shot"]["n"],
    fsE=mn["category_counts_few_shot"].get("E", 0), fsD=mn["category_counts_few_shot"].get("D", 0),
    so_fs=s(o["sensitivity_search_only"]["AorAprime_few_shot"]), so_all=s(o["sensitivity_search_only"]["AorAprime_all"]),
    so_n=o["sensitivity_search_only"]["n_reduced_label"], so_nfs=o["sensitivity_search_only"]["n_few_shot"],
    an_fs=s(o["posthoc_excluding_anomaly_PU"]["AorAprime_few_shot"]), an_st=s(o["posthoc_excluding_anomaly_PU"]["AorAprime_among_stated_draw_few_shot"]),
    ih_fs=s(o["posthoc_inherited_recoded_A"]["AorAprime_few_shot"]), ih_st=s(o["posthoc_inherited_recoded_A"]["AorAprime_among_stated_draw_few_shot"]),
    worst=f"{worst:.1f}", best=f"{best:.1f}",
    seeds=s(o["n_seeds_stated"]), nq_ver=int((fin.quote_check == "verified_in_text").sum()),
    nq_code=int((fin.quote_check == "code_quote_from_repo").sum()), nq_none=int((fin.quote_check == "none_found").sum()),
    n_code=int((fin.code_checked == "yes").sum()), n_hs=len(hs),
    mm_fs_auc=s(o["main_metric_few_shot"].get("auc")), mm_fs_acc=s(o["main_metric_few_shot"].get("accuracy")),
    mm_fs_mf1=s(o["main_metric_few_shot"].get("macro-f1")),
    any_mf1=s(o["metrics_any_use"].get("macro-f1")), any_acc=s(o["metrics_any_use"].get("accuracy")), any_auc=s(o["metrics_any_use"].get("auc")),
    n_unseen=len(o["unseen_class_scoring_stated"]),
)


def table_counts(d, title):
    rows = [f"| {k} | {v['k']} | {v['pct']}% | {v['wilson_lo']}–{v['wilson_hi']}% |" for k, v in d.items()]
    return f"| {title} | n | % | 95% CI |\n|---|---|---|---|\n" + "\n".join(rows)


def cat_table():
    a, f = mn["category_counts_all"], mn["category_counts_few_shot"]
    rows = [f"| {c} | {a.get(c, 0)} | {f.get(c, 0)} |" for c in ["A", "A'", "B", "C", "D", "E"]]
    return "| Category | All (n = %d) | Few-shot (n = %d) |\n|---|---|---|\n" % (mn["n_reduced_label"], mn["n_few_shot"]) + "\n".join(rows)


def breakdown(col):
    rows = []
    for g, v in o["breakdown"][col].items():
        rows.append(f"| {g} | {v['n']} | {v['AorAprime_all']['k']} ({v['AorAprime_all']['pct']}%; {v['AorAprime_all']['wilson_lo']}–{v['AorAprime_all']['wilson_hi']}) "
                    f"| {v['n_few_shot']} | {v['AorAprime_few_shot']['k']} ({v['AorAprime_few_shot']['pct']}%; {v['AorAprime_few_shot']['wilson_lo']}–{v['AorAprime_few_shot']['wilson_hi']}) |")
    return f"| {col} | n | A+A′ (95% CI) | n few-shot | A+A′ few-shot (95% CI) |\n|---|---|---|---|---|\n" + "\n".join(rows)


def esc(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


meta = s2.set_index("rec_id")
rows = []
for i, r in enumerate(fin.merge(inc[["paper_id", "title", "link"]], on="paper_id").itertuples(), 1):
    m = meta.loc[r.paper_id]
    venue = m.venue_from_text or m.venues or ("arXiv" if r.paper_id.startswith("AX") else "")
    rows.append(f"| {i} | [{esc(r.title[:70])}]({r.link}) | {m.year} | {esc(venue[:40])} ({m.venue_type}) | {r.few_shot} | **{r.category}** | {r.coverage_forced} | "
                f"{r.main_metric} | \"{esc(r.quote)}\" ({esc(r.quote_location)}) [{r.quote_check}{'; code' if r.code_checked == 'yes' else ''}] |")
inc_table = ("| # | Paper | Year | Venue (type) | Few-shot | Cat. | Forced coverage | Main metric | Quotation (location) [check] |\n"
             "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))

old_rows = "\n".join(f"| {r.name} | {r.old_category} | {r.category or '—'} | {r.coverage_forced or '—'} | {r.old_fewshot} | {r.few_shot or '—'} | {r.status} |"
                     for r in cmp_old.itertuples())
na_rows = "\n".join(f"| {r.rec_id} | {esc(re.sub(r'<[^>]+>', '', r.title)[:80])} | {esc(r.venues[:60])} |" for r in na.itertuples())
adj_rows = "\n".join(f"| {r.paper_id} | {r.variable} | {esc(r.coder1)} | {esc(r.coder2)} | **{esc(r.final)}** | {esc(r.reason)} |"
                     for r in adj.itertuples())
hs_rows = ", ".join(hs.paper_id)

T = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.txt")).read()
out = T.format(**V, cat_table=cat_table(), mm_table=table_counts(o["main_metric_few_shot"], "Main metric (few-shot)"),
               mm_all_table=table_counts(o["main_metric"], "Main metric (all)"),
               any_table=table_counts(o["metrics_any_use"], "Metric used (any use, all)"),
               year_table=breakdown("year_bin"), venue_table=breakdown("venue_type"),
               inc_table=inc_table, old_rows=old_rows, na_rows=na_rows, adj_rows=adj_rows, hs_rows=hs_rows,
               unseen="; ".join(f"{d['paper_id']}: {d['unseen_class_scoring']}" for d in o["unseen_class_scoring_stated"]))
open(P("REPORT.md"), "w").write(out)
print("REPORT.md written,", len(out), "chars")
