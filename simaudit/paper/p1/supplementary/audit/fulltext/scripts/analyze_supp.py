"""Combine stage-1/2 decisions and coding; quote check; PRISMA counts; distributions with Wilson CIs."""
import csv, json, os, re, math, random, collections
H = os.path.dirname(os.path.abspath(__file__))
cand = {r["rec_id"]: r for r in csv.DictReader(open(os.path.join(H, "candidates.csv")))}
s1 = {}
for b in (1, 2, 3):
    for r in csv.DictReader(open(os.path.join(H, f"stage1/stage1_decisions_batch{b}.csv"))): s1[r["rec_id"]] = r
ft = {r["rec_id"]: r for r in csv.DictReader(open(os.path.join(H, "fulltext_list.csv")))}
s2, cod = {}, {}
for b in range(1, 7):
    p = os.path.join(H, f"stage2/screening_batch{b}.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)): s2[r["rec_id"]] = r
    p = os.path.join(H, f"stage2/coding_batch{b}.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)): cod[r["rec_id"]] = r
# overrides by review lead (logged)
ovp = os.path.join(H, "stage2/review_lead_overrides.csv")
ov = list(csv.DictReader(open(ovp))) if os.path.exists(ovp) else []
import copy
s2_raw, cod_raw = copy.deepcopy(s2), copy.deepcopy(cod)
for o in ov:
    tgt = s2 if o["table"] == "screening" else cod
    if o["rec_id"] in tgt: tgt[o["rec_id"]][o["field"]] = o["new"]
for i, r in s2.items():
    if r["decision"] != "include": cod.pop(i, None)
# combined screening.csv
rows = []
for i, c in sorted(cand.items()):
    st1 = s1[i]
    row = dict(rec_id=i, stratum=c["stratum"], title=c["title"], arxiv_id=c["arxiv_id"], sources=c["sources"],
               stage1_decision=st1["decision"], stage1_reason=st1["reason"], frame=st1.get("frame", ""), stage1_note=st1.get("note", ""),
               sampled_for_fulltext="yes" if i in ft else ("no" if st1["decision"] == "include" else ""), stage2_decision="", stage2_reason="",
               evidence="", evidence_location="", stage2_note="")
    if i == "H3e067ca8":
        row.update(stage2_decision="not_accessible", stage2_reason="not_accessible", stage2_note="ACM OA PDF returned HTTP 403 to scripted fetch; no arXiv version found")
    elif i in s2:
        r = s2[i]; row.update(stage2_decision=r["decision"], stage2_reason=r["reason"], evidence=r["evidence"],
                              evidence_location=r["evidence_location"], stage2_note=r["note"])
    rows.append(row)
with open(os.path.join(H, "screening.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# quote check
def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
for i, r in cod.items():
    t = norm(open(os.path.join(H, "downloads/text", i + ".txt")).read())
    q = r["quote"]
    r["quote_words"] = len(q.split())
    r["quote_check"] = "none_found" if q.strip().lower() == "none found" else ("verified_in_text" if norm(q) and norm(q) in t else "NOT_FOUND")
    r["stratum"] = cand[i]["stratum"]; r["title"] = cand[i]["title"]; r["arxiv_id"] = cand[i]["arxiv_id"]; r["frame"] = s1[i].get("frame", "")
# cap at 60 (protocol 6)
inc = sorted(cod)
coded_ids = inc
if len(inc) > 60:
    C_ = [i for i in inc if cand[i]["stratum"] == "C"]; R_ = [i for i in inc if cand[i]["stratum"] == "R"]
    coded_ids = sorted(C_ + random.Random(20261029).sample(R_, 60 - len(C_)))
F = ["rec_id", "stratum", "frame", "arxiv_id", "title", "category", "category_secondary", "coverage_forced", "few_shot", "missed_class_reported", "metrics",
     "main_metric", "unseen_class_scoring", "budget_unit", "smallest_budget", "n_seeds", "code_checked", "quote", "quote_location", "notes", "quote_check", "quote_words"]
with open(os.path.join(H, "coding.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=F, extrasaction="ignore"); w.writeheader(); w.writerows(cod[i] for i in coded_ids)
def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))
counts = json.load(open(os.path.join(H, "counts.json")))
s1c = collections.Counter(r["reason"] for r in s1.values() if r["decision"] == "exclude")
s1inc = [i for i, r in s1.items() if r["decision"] == "include"]
prisma = dict(identification=counts,
              stage1_screened=len(s1), stage1_excluded=sum(s1c.values()), stage1_excluded_by_reason=dict(s1c),
              stage1_included=len(s1inc), stage1_included_C=sum(cand[i]["stratum"] == "C" for i in s1inc),
              stage1_included_R=sum(cand[i]["stratum"] == "R" for i in s1inc),
              R_sampled=sum(1 for r in ft.values() if r["stratum"] == "R"), fulltext_sought=len(ft), not_accessible=1,
              fulltext_assessed=len(s2), stage2_excluded=sum(r["decision"] != "include" for r in s2.values()),
              stage2_excluded_by_reason=dict(collections.Counter(r["reason"] for r in s2.values() if r["decision"] != "include")),
              included=len(inc), included_C=sum(cand[i]["stratum"] == "C" for i in inc), included_R=sum(cand[i]["stratum"] == "R" for i in inc),
              coded=len(coded_ids), review_lead_overrides=len(ov))
def dist(ids):
    n = len(ids); out = {"n": n}
    for v in ["category", "coverage_forced", "missed_class_reported", "main_metric", "few_shot", "budget_unit", "code_checked", "quote_check"]:
        out[v] = dict(collections.Counter(cod[i].get(v, "") for i in ids))
    k = sum(cod[i]["category"] in ("A", "A'") for i in ids); out["A_or_Aprime"] = [k, wilson(k, n)]
    k = sum(cod[i]["coverage_forced"] == "yes" for i in ids); out["coverage_forced_yes"] = [k, wilson(k, n)]
    k = sum(cod[i]["missed_class_reported"] == "yes" for i in ids); out["missed_class_reported_yes"] = [k, wilson(k, n)]
    k = sum(cod[i]["category"] in ("B", "D", "E") for i in ids); out["B_D_E_draw_not_using_labels_or_unstated"] = [k, wilson(k, n)]
    k = sum(cod[i]["category"] == "E" for i in ids); out["E_unspecified"] = [k, wilson(k, n)]
    mu = collections.Counter(m.strip() for i in ids for m in cod[i]["metrics"].split(";") if m.strip()); out["metric_any_use"] = dict(mu)
    fs = [i for i in ids if cod[i]["few_shot"] == "yes"]; out["few_shot_subset_n"] = len(fs)
    out["few_shot_subset_category"] = dict(collections.Counter(cod[i]["category"] for i in fs))
    return out
res = dict(prisma=prisma, all=dist(coded_ids), census=dist([i for i in coded_ids if cand[i]["stratum"] == "C"]),
           sample=dist([i for i in coded_ids if cand[i]["stratum"] == "R"]))
inc_raw = sorted(i for i in cod_raw if s2_raw[i]["decision"] == "include")
cod_bak = cod; cod = cod_raw
res["sensitivity_no_review_lead_overrides"] = dist(inc_raw)
cod = cod_bak
json.dump(res, open(os.path.join(H, "analysis_supp.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
