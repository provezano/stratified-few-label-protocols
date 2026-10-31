"""Supplementary search: title filters, scope filter, dedup, removal of main-review records -> candidates.csv, counts.json."""
import csv, json, os, re, sys, hashlib
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "raw")
MAIN = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(MAIN, "scripts"))
from merge_dedup import in_scope_venue, norm_title, strip_ver
TITLE_TERMS = ["foundation model", "in-context", "in context", "pretrain", "pre-train", "prior-fitted", "prior fitted", "pfn", "benchmark"]
def title_ok(t): t = (t or "").lower(); return any(x in t for x in TITLE_TERMS)
def tab_ok(t): t = (t or "").lower(); return "tabular" in t or "table" in t
def in_window(year, date):
    try: y = int(year or 0)
    except: y = 0
    return 2022 <= y <= 2026 and (not date or date <= "2026-10-27")
C = {}; recs = []
def s2rec(p, src):
    ex = p.get("externalIds") or {}
    return dict(src=src, arxiv_id=strip_ver(ex.get("ArXiv") or ""), doi=(ex.get("DOI") or "").lower(), title=p.get("title") or "",
                year=p.get("year") or "", date=p.get("publicationDate") or "", venue=p.get("venue") or "",
                abstract=p.get("abstract") or "", oa_pdf=((p.get("openAccessPdf") or {}).get("url") or ""))
# S1
s1 = json.load(open(os.path.join(RAW, "s1_citations_raw.json")))
for k, rows in s1.items():
    C[f"S1_{k}_raw"] = len(rows)
    kept = [r for r in rows if r.get("title") and title_ok(r["title"]) and in_window(r.get("year"), r.get("publicationDate"))]
    C[f"S1_{k}_title_filter_in_window"] = len(kept)
    recs += [s2rec(r, "S1_cites_" + k) for r in kept]
# S3 S2 bulk
b = json.load(open(os.path.join(RAW, "s3_s2bulk_raw.json")))
C["S3_s2bulk_raw"] = len(b)
kept = [r for r in b if title_ok(r.get("title")) and tab_ok(r.get("title")) and in_window(r.get("year"), r.get("publicationDate"))]
C["S3_s2bulk_title_filter_in_window"] = len(kept)
recs += [s2rec(r, "S3_s2bulk") for r in kept]
# S3 OpenAlex
oa = json.load(open(os.path.join(RAW, "s3_openalex_raw.json")))
def inv(ii):
    if not ii: return ""
    pos = {i: w for w, l in ii.items() for i in l}; return " ".join(pos[i] for i in sorted(pos))
n = 0
for q, rows in oa.items():
    n += len(rows)
    for w in rows:
        blob = " ".join((l.get("landing_page_url") or "") + " " + (l.get("pdf_url") or "") for l in (w.get("locations") or [])) + " " + (w.get("doi") or "")
        m = re.search(r"arxiv(?:\.org/(?:abs|pdf)/|\.)(\d{4}\.\d{4,5})", blob, re.I)
        venues = sorted({(l.get("source") or {}).get("display_name") or "" for l in (w.get("locations") or [])} - {""})
        pdfs = [l.get("pdf_url") for l in (w.get("locations") or []) if l.get("pdf_url")]
        recs.append(dict(src="S3_openalex", arxiv_id=m.group(1) if m else "", doi=(w.get("doi") or "").replace("https://doi.org/", "").lower(),
                         title=w.get("display_name") or "", year=w.get("publication_year") or "", date=w.get("publication_date") or "",
                         venue=" | ".join(venues), abstract=inv(w.get("abstract_inverted_index")), oa_pdf=pdfs[0] if pdfs else ""))
C["S3_openalex_raw"] = n
# S2 model lists + benchmark papers, S4 seeds (hand-entered from TabArena/TALENT repos, 2026-10-28)
manual = [
 ("2207.01848", "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second", "S4_seed;S2_list_talent"),
 ("", "Accurate predictions on small data with a tabular foundation model", "S4_seed;S2_list_talent;S2_list_tabarena"),
 ("2506.16791", "TabArena: A Living Benchmark for Machine Learning on Tabular Data", "S2_benchmark"),
 ("2606.30410", "Beyond IID: How General Are Tabular Foundation Models, Really?", "S2_benchmark"),
 ("2407.00956", "A Closer Look at Deep Learning Methods on Tabular Datasets", "S2_benchmark"),
 ("2407.04057", "TALENT: A Tabular Analytics and Learning Toolbox", "S2_benchmark"),
 ("2402.14335", "HyperFast", "S2_list_talent"), ("2311.00055", "TabPTM", "S2_list_talent"),
 ("2502.05564", "TabICL", "S2_list_talent;S2_list_tabarena"), ("2510.21204", "Mitra", "S2_list_talent;S2_list_tabarena"),
 ("2509.03505", "LimiX", "S2_list_talent;S2_list_tabarena"), ("2507.03971", "Real-TabPFN", "S2_list_talent"),
 ("2605.13986", "TabPFN-3", "S2_list_talent;S2_list_tabarena"), ("2602.11139", "TabICLv2", "S2_list_talent"),
 ("2511.08667", "TabPFN-2.5", "S2_list_talent;S2_list_tabarena"), ("2410.18164", "TabDPT", "S2_list_talent;S2_list_tabarena"),
 ("2609.37959", "TabFM", "S2_list_talent;S2_list_tabarena"), ("2609.17895", "TabPFN-3.5", "S2_list_talent;S2_list_tabarena"),
 ("2609.04540", "Mitra-v2", "S2_list_talent;S2_list_tabarena"), ("2609.22866", "Causilo", "S2_list_talent;S2_list_tabarena"),
 ("2609.03880", "Xiaomi-TabLDM", "S2_list_talent;S2_list_tabarena"), ("2608.25774", "EXAONE Tabular", "S2_list_talent;S2_list_tabarena"),
 ("2609.17488", "LimiX-2", "S2_list_talent;S2_list_tabarena"), ("2510.06162", "TabPFN-Wide", "S2_list_tabarena"),
 ("2511.15941", "iLTM", "S2_list_tabarena"), ("2610.25558", "FTFM", "S2_list_tabarena"), ("2511.02818", "Orion-MSP", "S2_list_tabarena"),
 ("2606.07345", "TabSwift", "S2_list_tabarena"), ("2505.18125", "TabSTAR", "S2_list_tabarena"), ("2506.10707", "ConTextTab", "S2_list_tabarena"),
]
C["S2_S4_manual_records"] = len(manual)
C["S2_models_without_paper"] = "Kumo Tabular (blog), Nori (repo), LightPFN (repo)"
for a, t, s in manual:
    recs.append(dict(src=s, arxiv_id=a, doi="10.1038/s41586-024-08328-6" if not a else "", title=t, year="", date="", venue="Nature" if not a else "",
                     abstract="", oa_pdf="", manual=True))
C["records_after_title_filters_total"] = len(recs)
# dedup (union-find)
parent = list(range(len(recs)))
def find(i):
    while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
    return i
km = {}
for i, r in enumerate(recs):
    keys = []
    if r["arxiv_id"]: keys += ["ax:" + r["arxiv_id"], "doi:10.48550/arxiv." + r["arxiv_id"].lower()]
    if r["doi"]: keys.append("doi:" + r["doi"])
    if not r.get("manual") and len(norm_title(r["title"])) > 15: keys.append("t:" + norm_title(r["title"]))
    for k in keys:
        if k in km: parent[find(i)] = find(km[k])
        else: km[k] = i
g = defaultdict(list)
for i, r in enumerate(recs): g[find(i)].append(r)
merged = []
for grp in g.values():
    best = max(grp, key=lambda r: (not r.get("manual"), len(r["abstract"])))
    venues = sorted({v for r in grp for v in (r["venue"] or "").split(" | ") if v})
    years = [int(r["year"]) for r in grp if str(r["year"]).isdigit()]
    srcs = sorted({s for r in grp for s in r["src"].split(";")})
    merged.append(dict(sources=";".join(srcs), arxiv_id=next((r["arxiv_id"] for r in grp if r["arxiv_id"]), ""),
                       doi=next((r["doi"] for r in grp if r["doi"]), ""), title=best["title"], year=min(years) if years else "",
                       venues=" | ".join(venues), oa_pdf=next((r["oa_pdf"] for r in grp if r["oa_pdf"]), ""),
                       abstract=best["abstract"], manual=any(r.get("manual") for r in grp), n_raw=len(grp)))
C["records_after_dedup"] = len(merged); C["duplicates_removed"] = len(recs) - len(merged)
# scope filter (arXiv id or in-scope venue); manual records always in scope
sc = [m for m in merged if m["manual"] or m["arxiv_id"] or any(in_scope_venue(v) for v in m["venues"].split(" | "))]
C["out_of_scope_removed"] = len(merged) - len(sc); C["in_scope"] = len(sc)
# remove main-review records
mr = list(csv.DictReader(open(os.path.join(MAIN, "screening", "records_merged.csv"))))
max_ = {strip_ver(r["arxiv_id"]) for r in mr if r["arxiv_id"]}
mt = {norm_title(r["title"]) for r in mr if len(norm_title(r["title"])) > 15}
out = []; removed = []
for m in sc:
    if (m["arxiv_id"] and m["arxiv_id"] in max_) or norm_title(m["title"]) in mt:
        removed.append(m)
    else: out.append(m)
C["already_in_main_review_removed"] = len(removed); C["candidates"] = len(out)
for m in out:
    m["rec_id"] = ("AX" + m["arxiv_id"]) if m["arxiv_id"] else "H" + hashlib.sha1(norm_title(m["title"]).encode()).hexdigest()[:8]
    m["stratum"] = "C" if m["manual"] else "R"
out.sort(key=lambda m: m["rec_id"])
F = ["rec_id", "stratum", "sources", "arxiv_id", "doi", "title", "year", "venues", "oa_pdf", "n_raw", "abstract"]
with open(os.path.join(HERE, "candidates.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=F, extrasaction="ignore"); w.writeheader(); w.writerows(out)
with open(os.path.join(HERE, "removed_already_in_main_review.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["arxiv_id", "title", "sources"], extrasaction="ignore"); w.writeheader(); w.writerows(removed)
C["candidates_by_stratum"] = {s: sum(1 for m in out if m["stratum"] == s) for s in "CR"}
json.dump(C, open(os.path.join(HERE, "counts.json"), "w"), indent=1); print(json.dumps(C, indent=1))
