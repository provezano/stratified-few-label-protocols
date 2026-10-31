# Protocol — systematic review of label-drawing protocols in few-label tabular classification

**Protocol fixed at: 2026-10-27T22:37Z (UTC), before any search was run.**
Paper: P1 (TMLR submission). Language: English.

Pre-protocol actions (disclosed): connectivity probes only, at 2026-10-27T22:33Z. They returned HTTP status codes; no hit list was read.
- arXiv API (export.arxiv.org): reachable (HTTP 200 over https).
- Semantic Scholar Graph API: intermittent HTTP 429 without a key; works with delays.
- DBLP search API (dblp.org and dblp.uni-trier.de): returns a bot-challenge HTML page ("Making sure you're not a bot!"), so scripted access is blocked.
- OpenAlex API: reachable (HTTP 200).

This protocol must not change after searching. Any change goes in a dated "Deviation" section at the end.

## 1. Research question

Among papers that evaluate tabular **classification** methods with a **reduced labelled set**, what share draws the labelled set **with the help of the class labels**? "With the help of the labels" means the same number per class, proportional stratification, at least one per class, a one-per-class seed, or rejecting seeds that miss a class (categories A and A′ below).

Secondary questions:
- How often do these papers report how often a class got no label (the missed-class rate)?
- Which metrics do they report (accuracy, AUC, macro-F1, balanced accuracy, other)? How do they score classes that never appear in the labelled set?
- What budget unit do they use (per class, total, fraction), and how many seeds?

## 2. Population and scope

- Papers that are **arXiv preprints** (any arXiv category), or papers published in the **in-scope venues** of §4.
- Applied-domain journals outside the venue list are out of scope unless the paper is also on arXiv. This keeps the review on the ML methods literature that sets evaluation practice.
- Date window: **2019-01-01 to 2026-10-27** (first public version for arXiv; publication year for venues).
  - Justification: 2019 is the start of the modern deep-tabular and tabular-SSL line (TabNet 2019, VIME 2020, then SCARF, SAINT, TabPFN, STUNT, TabLLM). It covers every paper of the prior 39-paper audit (earliest 2020). The upper bound is the search date. Earlier work (pre-2019 SSL on UCI tables) uses different models, so older practice says little about the current field.

## 3. Sources

1. **arXiv API** (`https://export.arxiv.org/api/query`), title + abstract fields, date filter on `submittedDate`.
2. **Semantic Scholar Graph API**, bulk search (`/graph/v1/paper/search/bulk`), which matches title and abstract; filters `year=2019-2026`, `fieldsOfStudy=Computer Science`.
3. **DBLP venue search**: planned. It was blocked at the probe. It will be retried once at search time. If still blocked, it is recorded as "not accessible", and venue coverage relies on sources 2 and 4.
4. **OpenAlex API** (`/works`, `filter=title_and_abstract.search:<query>,from_publication_date:2019-01-01,to_publication_date:2026-10-27`). This substitutes for DBLP for venue coverage.
5. **Forward snowballing** (one round, Semantic Scholar citations endpoint) from four seed papers: VIME (NeurIPS 2020), SCARF (ICLR 2022), STUNT (ICLR 2023), TabLLM (AISTATS 2023). A citing paper is kept for screening only if its title or abstract contains at least one label-scarcity term of §4 (the word "tabular" is **not** required here; this catches papers that say "structured data" or "tables").
6. **Prior audit** (`<prior audit file>`, 39 papers). These enter as "records from other sources" in the PRISMA flow. Each goes through the same eligibility rule. Those that meet it are included and **re-coded blind** (the coders never see the old codes).

Not used: Sci-Hub, LibGen or any pirated source; browsers; OpenReview by script (it returns a 403 challenge). If a paper is only on OpenReview, it is recorded as "not accessible".

## 4. Search strings

Tabular term **T** = `tabular`.

Label-scarcity terms **L** (each phrase searched as a phrase where the API supports it):
`few-shot`, `few shot`, `low-shot`, `few labeled`, `few labelled`, `few labels`, `limited labeled`, `limited labelled`, `limited labels`, `limited labeling`, `label scarcity`, `scarce labels`, `label-scarce`, `scarce labeled`, `semi-supervised`, `semi supervised`, `low-data`, `low data`, `small labeled`, `labeling budget`, `labelling budget`, `annotation budget`, `label budget`, `label-efficient`, `cold-start`, `cold start`, `active learning`.

Query = **T AND (any L)**, run per source as follows.

- **arXiv**: one query per L term:
  `(ti:tabular OR abs:tabular) AND (ti:"<L>" OR abs:"<L>") AND submittedDate:[201901010000 TO 202610272359]`
  Paged 200 at a time, 3 s between calls.
- **Semantic Scholar bulk**: one query, using its boolean syntax:
  `tabular + ("few-shot" | "few shot" | "low-shot" | ... | "active learning")` (all L terms, OR-ed), year 2019–2026, field Computer Science. Paged with the continuation token. Delay ≥ 3 s, with retry and back-off on 429.
- **OpenAlex**: one query per L term: `title_and_abstract.search:tabular AND "<L>"`, dates as in §2, 200 per page, 1 s between calls, `mailto` omitted.
- **DBLP** (if reachable): `tabular <L>` per L term, then keep hits in in-scope venues.

**In-scope venues** (matched on the venue / source name, case-insensitive): NeurIPS (incl. its workshops and Datasets & Benchmarks), ICML (incl. workshops), ICLR (incl. workshops), AISTATS, UAI, COLT is excluded (theory), TMLR, JMLR, AAAI, IJCAI, KDD, The Web Conference (WWW), CIKM, WSDM, IEEE ICDM, SIAM SDM, ECML-PKDD, ACL, EMNLP, NAACL, EACL, COLM, IEEE TPAMI, IEEE TKDE, ACM TKDD, Machine Learning (Springer), Data Mining and Knowledge Discovery, Nature, Nature Machine Intelligence, Nature Communications, PMLR (any volume).

**Identification-stage scope filter (automatic, by script):** a record from Semantic Scholar, OpenAlex or DBLP is kept if it has an arXiv id **or** its venue matches the in-scope list. All raw hits are saved before this filter.

Raw hit lists (source, query, id, title, year, venue, plus abstract when available) are saved as CSV/JSON in `raw/`.

## 5. Deduplication

Records are merged by, in order: arXiv id (version stripped); DOI; normalized title (lower case, alphanumerics only). Versions of the same work (preprint + venue version, v1/v2) count as one record. The venue version is the one coded when both exist; the arXiv text is used if the venue text is not accessible.

## 6. Screening

**Stage 1 — title and abstract.** Include for full text if the record plausibly meets all three:
(a) tabular data (rows × features; not images, text, graphs, time series or tables-as-text for QA);
(b) a classification task (binary or multiclass) is evaluated;
(c) an experiment with a reduced labelled set (few-shot, few labels, semi-supervised with a labelled fraction, active learning with an initial labelled pool, cold start, label budget).
When in doubt, the record goes to full text.

Stage-1 exclusion reasons (one per record, first that applies): `not_tabular`, `table_QA_or_text2sql` (tables used as text for QA, reasoning, or SQL), `recsys_cold_start`, `not_classification` (regression-only, generation, imputation, anomaly detection without labels, clustering), `no_reduced_label_experiment` (clear from abstract), `survey_or_position` (no experiments), `out_of_window`, `not_a_paper` (dataset card, thesis chapter duplicate, erratum).

**Stage 2 — full text.** Read the methods and experiments sections (and appendix if needed). Include if:
1. at least one experiment trains or adapts a model on tabular data for classification, and
2. that experiment uses a **reduced labelled set**: a few-shot setting (≤ ~100 labelled rows, or framed as few-shot / k-shot / cold start), **or** a labelled fraction / subsample of the training labels chosen to study label scarcity (semi-supervised, active learning initial pool, label budget, learning curves over labelled-set size).

Exclude if: regression-only; no reduced-label experiment (only full training sets, or the training size varies only for latency/scaling of compute); non-tabular; survey, position paper or no experiments; duplicate or version of an included record; zero-shot only (no labelled rows at all); text not accessible (recorded as `not_accessible`, not as excluded on merit).

Screening is done by the review lead. Stage 1 may be split into batches across language-model sessions that apply the same written rules; each decision is saved with its reason. Counts are kept for a PRISMA flow: identified per source, after scope filter, duplicates removed, screened, excluded at stage 1 (by reason), full text sought, not accessible, full text assessed, excluded at stage 2 (by reason), included (search vs. other sources).

## 7. Coding manual (variables per included paper)

Code the paper's **main reduced-label protocol**: the one behind its headline few-label / low-label results. If several protocols are used, code the main one and note the others in `notes`. For active learning, code how the **initial** labelled pool is drawn. If the whole labelled set is chosen by an algorithm with no random initial set, use C.

Categories (same definitions as the prior audit, §0):
- **A**: m labels per class, or a total of k balanced across classes. Assumes the labels.
- **A′**: another draw that uses the labels to guarantee coverage: proportional stratification, a constraint of ≥ 1 per class, a seed of 1 per class, or discarding seeds that miss a class.
- **B**: uniform draw of a total budget (no use of labels).
- **C**: selected budget (active or structured selection, e.g. by recency, clustering, uncertainty).
- **D**: a fraction of the training split, without saying whether it is stratified.
- **E**: not specified, or no experiment.

Variables (one row per paper in the coding CSV):

| column | values | rule |
|---|---|---|
| `paper_id` | as given | |
| `category` | A, A′ (write `A'`), B, C, D, E | main protocol |
| `category_secondary` | same or empty | other protocol used in the paper, if any |
| `coverage_forced` | yes / no / unclear | **yes** if the stated draw guarantees ≥ 1 labelled row per class by construction or by rejection (per-class draw with m ≥ 1, explicit ≥ 1 constraint, 1-per-class seed, seed rejection). **no** if a draw without any such guarantee is stated (uniform total, stated non-stratified fraction, selection with no class constraint). **unclear** if the draw is not described, or if labels are used but the guarantee is not stated (e.g. proportional stratification with tiny k and unstated rounding). |
| `few_shot` | yes / no | **yes** if the paper frames the setting as few-shot / k-shot / cold start, or any main reduced-label experiment uses ≤ 100 labelled rows in total or ≤ 10 per class. Otherwise no. |
| `missed_class_reported` | yes / no | yes only if the paper reports how often (or in how many runs/datasets) a class had no labelled row. A mention that it can happen, without a number, is `no` (note it). |
| `metrics` | semicolon list from: accuracy, AUC, macro-F1, F1-other (binary/weighted/micro/unspecified), balanced-accuracy, log-loss, other | metrics reported for the reduced-label classification results |
| `main_metric` | one of the above | the metric of the headline table/figure |
| `unseen_class_scoring` | free text or `not stated` | how classes absent from the labelled set are scored, if stated |
| `budget_unit` | per-class / total / fraction / mixed | unit in which the labelled budget is set |
| `smallest_budget` | free text | e.g. "1 per class", "4 total", "1%" |
| `n_seeds` | integer or `not stated` | repetitions of the label draw for the main protocol |
| `code_checked` | yes / no | yes only if the coder read the official code (read-only) to decide the draw |
| `quote` | ≤ 15 words, verbatim | the sentence that best shows the draw |
| `quote_location` | section / page / table | where the quote is |
| `notes` | free text | ambiguities, other protocols |

Decision aids:
- "randomly sample k examples" with nothing else → B (coverage `no`), unless the code (if checked) shows otherwise.
- "k-shot" defined as k per class → A. "k-shot" defined as k total and drawn class-balanced → A. "k total, stratified" (proportional) → A′.
- "We use 10% of the training data as labelled" with no draw described → D. "stratified 10%" → A′. "random 10% (not stratified)" stated explicitly → B.
- Learning curves over training sizes with the draw unstated → D if sizes are fractions, E if sizes are counts with nothing about the draw. Counts with "random subsample" → B.
- Train/test split stratification does **not** count; only the draw of the labelled set does.
- If the protocol is inherited by citation ("following STUNT"), code the inherited protocol and note it.

Quotations: only text actually read in the source; ≤ 15 words; with location. If the coder cannot find a sentence about the draw, write `none found` and code E (or D for fractions).

## 8. Two independent coders and agreement

- Two language-model coders (coder 1, coder 2) code every included paper. They receive this manual and the list of included papers (ids, titles, links, local text files). They never receive the other coder's codes, the prior audit's codes, or P1. Large lists may be split into batches; each batch for coder 1 is run by a fresh coder-1 session with the same instructions (and likewise for coder 2).
- Agreement: Cohen's kappa and raw agreement for `category` and `coverage_forced` (primary). Raw agreement (and kappa where defined) for `few_shot`, `missed_class_reported`, `budget_unit`, `main_metric`.
- **Disagreements** on any coded variable used in the analysis are resolved by a third adjudication pass (the review lead or a third coder) that reads the source text again. Each resolution is logged (paper, variable, coder 1, coder 2, final, reason, quote).
- The final coding CSV (`coding/final_coding.csv`) holds the adjudicated values.
- Disclosure: all coders are runs of a large language model. A human check of a random 20% subset (seeded draw recorded in the report) is pending.

## 9. Analysis (script, no hand-typed numbers)

`scripts/analyze.py` reads `coding/final_coding.csv` and the screening table, and prints:
- PRISMA counts.
- Share with A + A′ among few-shot studies, and among all reduced-label studies; share A alone; share `coverage_forced = yes`; share `missed_class_reported = yes`. Each with a Wilson 95% interval.
- Distribution of `main_metric` and of any-use of each metric (macro-F1 vs accuracy vs AUC in particular).
- Breakdown by year and venue type (peer-reviewed conference / journal / workshop / preprint only).
- Sensitivity: the same shares on the search-only set (excluding papers that entered only via the prior audit).
- Kappa and raw agreement from the two coder CSVs.

## 10. Stopping rule

- The searches in §4 are run once, with the exact strings above. No query is refined after seeing results.
- One round of forward snowballing (§3.5), no further rounds.
- Every identified, in-scope, de-duplicated record is screened. No saturation-based early stop.
- Every stage-1 inclusion is sought in full text. A paper that cannot be read after trying arXiv, the proceedings site and the Semantic Scholar open-access link is recorded as `not_accessible`.

## 11. Known limitations (stated in advance)

- Papers whose abstract never mentions tabular data and label scarcity (e.g. a foundation-model paper with a learning-curve appendix) are hard to find by search. The prior-audit papers partly cover this; the report gives the share both with and without them.
- Scope excludes applied-domain journals not on arXiv.
- Language-model coders; human check pending.

---

## Deviations (added after searching; the sections above are unchanged)

**2026-10-27, ~23:00–23:20Z**
1. **OpenAlex paging.** Cursor-based paging returned HTTP 401 (the key-less shared budget does not cover cursor pages). The script switched to `page=` paging with the same filter strings. Run logs of the failed attempts are kept (`raw/openalex_search_run1_401.log`, `raw/openalex_search_run2_cursor401.log`). No query string changed.
2. **DBLP.** Still blocked at search time (`raw/dblp_status.json`). Recorded as not accessible, as planned in §3.
3. **Snowball seed id.** The first run used a wrong Semantic Scholar id for VIME (VIME is not on arXiv; the id pointed to a 4-citation paper). Fixed by looking VIME up by exact title; the first run is kept as `raw/s2_snowball_run1_wrong_vime_seed.json` and not used.
4. **arXiv ids parsed as numbers.** A pandas step dropped trailing zeros from three arXiv ids, so their downloads failed. Fixed (ids read as strings) and re-downloaded. No screening decision was affected.
5. **Withdrawn preprints.** Two records (arXiv 2208.14834, 2311.18356) are withdrawn on arXiv; only v1 is available. The protocol is silent on this. They are assessed on the v1 text and flagged.
6. **Stage-2 screening in batches.** §6 says screening is done by the review lead, with stage 1 splittable into batches. Stage 2 was also split across five language-model sessions applying the same written rules, because of volume (207 full texts).
7. **Stage-2 edge rules made explicit** (written before stage 2 started, after stage 1): (a) semi-supervised anomaly detection that trains with a few labelled anomalies on tabular data counts as binary classification with reduced labels; (b) subsampling the whole dataset to a small n to study small-data performance counts as a reduced labelled set; (c) semi-supervised feature selection evaluated by classification accuracy counts. These follow from §6 but were not spelled out.
8. **Extra source for full text.** NeurIPS 2024 PDFs were found through the public NeurIPS proceedings index; one Springer open-access PDF was fetched from link.springer.com. Both are "proceedings site / publisher OA" routes allowed by §10.

**2026-10-27, ~23:50Z**
9. **Wrong full texts in the first download run.** Because of item 4, 17 arXiv ids ending in "0" were requested without the trailing zero, and arXiv served a *different* paper for 15 of them (e.g. 2602.2228 → 2602.02228). Stage-2 screeners caught 11; an automatic title check then found all 17. All 17 PDFs were deleted, re-downloaded with the correct id, title-checked, and re-screened by a fresh screener session (`screening/stage2_decisions_batch6_refetch.csv`, which supersedes the earlier rows).
10. **Eligibility consistency rule (review lead, at stage-2 adjudication).** The five stage-2 batches treated "naturally small datasets used whole (no label subsampling, not framed as few-shot)" differently. The review lead applied one rule: such papers are **excluded** (`no_reduced_label_experiment`), because §6 requires a few-shot setting or a deliberate reduction of the labels. Overrides are logged with reasons in `screening/stage2_adjudication_overrides.csv` (9 rows, including two multimodal papers without a tabular-only experiment and one parameter-estimation paper).
