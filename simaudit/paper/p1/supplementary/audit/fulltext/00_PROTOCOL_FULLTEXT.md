# Protocol: supplementary full-text search (P1, systematic review of label-drawing protocols)

**Protocol fixed at: 2026-10-28T01:40Z (UTC), before any supplementary search was run.**
Parent protocol: `simaudit/paper/p1/supplementary/audit/00_PROTOCOL.md` (main review, fixed 2026-10-27T22:37Z) and `01_CODING_MANUAL.md`. This file does not change the main review. Any change after searching goes in a dated "Deviation" section at the end.
Outputs: `supplementary_fulltext/` next to this file (`candidates.csv`, `screening.csv`, `coding.csv`, raw hit files, logs, `counts.json`).

## 1. Why

The main review searched titles and abstracts and required "tabular" AND a label-scarcity term (main protocol §4). Its own §11 states the known gap: papers that report a few-label experiment only in the body (typically tabular foundation-model papers with learning curves over small training sizes, and large tabular benchmarks with small-n subsampling) are hard to find that way. This supplementary search samples that gap and applies the main review's eligibility rule and coding manual to the full text.

Research question: the same as the main review (§1), restricted to the frame below. Main outcomes: category distribution (A/A'/B/C/D/E), share `coverage_forced = yes`, share `missed_class_reported = yes`, main metrics.

## 2. Frame

Papers with first public version (arXiv) or publication year between **2022-01-01 and 2026-10-27** that are either
- (F1) **tabular foundation-model / in-context-learning / pretrained tabular model papers**: they propose, adapt, fine-tune, or evaluate a model pretrained across tables (incl. prior-fitted networks, TabPFN-family, LLM-based row-serialisation classifiers, cross-table pretraining), or
- (F2) **tabular benchmark / large empirical comparison papers** that evaluate classifiers on many tabular datasets.

Scope filter: the main review's (§2, §4): the record has an arXiv id **or** its venue matches the main in-scope venue list. Applied-domain journals not on arXiv are out of scope.

## 3. Sources (identification)

- **S1. Forward citations** of TabPFN v1 (Hollmann et al., ICLR 2023, arXiv 2207.01848) and TabPFN v2 (Hollmann et al., Nature 2025, "Accurate predictions on small data with a tabular foundation model"), from the Semantic Scholar Graph API citations endpoint (all pages the API returns). A citing record is kept if its **title** contains (case-insensitive) at least one of: `foundation model`, `in-context`, `in context`, `pretrain`, `pre-train`, `prior-fitted`, `prior fitted`, `PFN`, `benchmark`.
- **S2. Benchmark model lists.** The models evaluated in **TabArena** (Erickson et al., arXiv 2506.16791) and **TALENT** (Ye et al., "A Closer Look at Deep Learning Methods on Tabular Datasets", arXiv 2407.00956; and the TALENT toolbox paper, arXiv 2407.04057), taken from the papers or their public repository READMEs. Each model is mapped to its originating paper; only papers in the 2022–2026 window that are in the F1/F2 frame are kept (classical GBDTs and pre-2022 deep models drop out by the window). The two benchmark papers themselves are also records.
- **S3. Database queries** (title + abstract), window as in §2:
  - Semantic Scholar bulk search, one query: `tabular + ("foundation model" | "in-context learning" | "prior-fitted" | "pretrained" | "pre-trained" | "benchmark")`, fields of study Computer Science. Because this can be large, only records whose **title** contains one of the S1 title terms AND (`tabular` or `table`) are kept.
  - OpenAlex `/works`, `title.search` queries (title only, to bound volume): `tabular foundation model`, `tabular in-context learning`, `prior-fitted network`, `TabPFN`, `pretrained tabular`, `tabular benchmark`.
- **S4. Seed papers** entered directly (other sources): TabPFN v1, TabPFN v2.

All raw hits are saved before filtering. arXiv API is not queried separately (S2/S3 carry arXiv ids). No browser; OpenReview not fetched by script (as in the main review).

## 4. De-duplication and removal of already-handled records

Merge by arXiv id (version stripped), then DOI, then normalised title (lower case, alphanumerics only). Then **remove every record already in the main review's frame**, i.e. any record in `screening/records_merged.csv` of the main review (which includes all screened and all coded papers, `final_coding.csv` being a subset), matched by arXiv id or normalised title. Counts of removals are kept.

## 5. Stage 1: title/abstract screen into the frame

Include if plausibly all of: (a) tabular data (rows × features; not table QA, text-to-SQL, table understanding/retrieval, images, time series forecasting, graphs); (b) classification is evaluated (or plausibly evaluated: a general tabular model or benchmark counts); (c) the paper is in F1 or F2; (d) it has experiments (not survey/position). Unlike the main review, a label-scarcity term is **not** required. Exclusion reasons as in the main review §6, plus `not_in_frame` (tabular paper but neither F1 nor F2). In doubt, include.

## 6. Strata and sampling (fixed now)

- **Census stratum (C)**: S2 model papers + the two benchmark papers + S4 seeds that pass stage 1. All go to full text.
- **Sample stratum (R)**: all other stage-1 inclusions. If more than **80**, a simple random sample of 80 is drawn with `random.Random(20261028).sample(sorted(rec_ids), 80)`. The sampling fraction is reported, and shares in R are reported separately from C.
- If the number of included papers (after stage 2) exceeds **60**, all C inclusions are coded and R inclusions are reduced to fill 60 with `random.Random(20261029).sample(sorted(ids), k)`.

## 7. Stage 2: full-text eligibility (main review rule, applied to the full text)

Main protocol §6 stage 2, including deviations 7 and 10. Include if:
1. at least one experiment trains, adapts or conditions (in-context) a model on tabular data for **classification**; and
2. that experiment uses a **reduced labelled set**: a few-shot setting (≤ ~100 labelled rows, or framed as few-shot / k-shot / cold start), or a labelled subsample of the training labels chosen to study label scarcity or small-data performance (semi-supervised fraction, active-learning initial pool, label budget, **learning curves over training/labelled-set size**, subsampling a dataset to small n).

Exclude: regression only; no reduced-label experiment (only full training sets; training size varied only for runtime/compute scaling; naturally small datasets used whole and not framed as few-shot, per deviation 10); non-tabular; survey/position; zero-shot only; duplicate. Full text not readable: `not_accessible` (not an exclusion on merit).

**Note on in-context learning:** for ICL models, the context set plays the role of the labelled training set; subsampling the context/training rows to small n counts as a reduced labelled set.

Full texts only from legal open sources: arXiv, OpenReview (manual open page only if needed, not scripted), PMLR / NeurIPS / ICLR proceedings, publisher open access (e.g. Nature OA). No Sci-Hub, no LibGen. Text extracted with pypdf with page markers (main review's `extract_text.py`).

## 8. Coding

Same manual (`01_CODING_MANUAL.md`, main protocol §7), same variables and spellings, same decision aids, same quotation rule (≤ 15 words, verbatim, with location). Code the paper's main reduced-label protocol: here, the experiment that qualifies the paper under §7 (e.g. the small-n learning curve or few-shot table); others go in `notes`. Official code is read only under the manual's rule (only when the text leaves the draw ambiguous; read-only; never executed), with `code_checked = yes`. Extra columns: `stratum` (C/R), `source`, `frame` (F1/F2).

Coders: one coding pass per paper, done by language-model sessions applying the manual (batches allowed, as in the main review); every quote is then checked automatically against the extracted text. **No second independent coder** in this supplementary pass (stated as a limitation).

## 9. Analysis

Counts for a PRISMA-style flow (identified per source, after title filter, after scope filter, duplicates, already in main review, stage-1 screened/excluded by reason, sampled, full text sought, not accessible, assessed, excluded by reason, included, coded). Distribution of category; share `coverage_forced = yes`; share `missed_class_reported = yes`; distribution of `main_metric`; by stratum. Wilson 95% intervals. Computed by script from the CSVs.

## 10. Stopping rule

Each query and citation pull is run once with the strings above; no refinement after seeing results. No further snowballing. Every sampled stage-1 inclusion is sought in full text.

## 11. Limitations stated in advance

- Frame is defined by citations of TabPFN, two benchmark model lists and title queries; FM papers that cite neither and use other wording are missed.
- Sampling within stratum R (if triggered).
- Single coding pass, no inter-coder agreement; language-model screeners/coders.
- Title filters in S1/S3 are crude by design.

---

## Deviations (added after searching; sections above unchanged)

**2026-10-28, ~01:45–03:30Z** (searches run once, 01:41–01:50Z; strings unchanged)
1. **Order of scope filter.** §4 fixes dedup then removal; §2 fixes the scope filter but not its position. It was applied **after** dedup: a merged record is in scope if any member has an arXiv id or an in-scope venue. Manual (S2/S4) records are always in scope.
2. **Model-list entries without a paper.** Kumo Tabular (blog only), Nori and LightPFN (repositories only) have no originating paper and enter no record. 8 of the 30 manual records were already in the main review and were removed (§4), leaving 22 in stratum C.
3. **Full-text routes.** EquiTabPFN (an OpenAlex record without arXiv id) was fetched from its arXiv version (2502.06684), found by exact-title search on the arXiv API. StablePFN (ACM, DOI 10.1145/3770855.3818030, listed CC-BY open access) returned HTTP 403 to a scripted fetch on both doi.org and dl.acm.org and is recorded `not_accessible`. TabPFN v2 was fetched from the Nature open-access PDF.
4. **Batches.** Stage 1 was split across 3 language-model screener sessions and stage 2 + coding across 6 sessions, all with the same written rules (`stage2_coder_instructions.md`). The stage-2 instructions did not define the `frame` column; the frame of each record is therefore taken from stage 1.
5. **Review-lead consistency overrides (stage 2), written after seeing the batch outputs.** Batches differed on fixed context budgets. One rule was applied: a context/training subset of fixed size drawn from a large, fully labelled pool for compute, latency or context-window reasons, or as a stress test of another component, is **not** a reduced labelled set (as already applied by the screeners to BAPS, TabDPT-Turbo, HyperFast, Mitra-v2). Experiments whose only reduced-label data are images or bags of image embeddings are `not_tabular` (main protocol rule (a)). 7 includes became excludes, plus 1 `few_shot` correction; all logged in `supplementary_fulltext/stage2/review_lead_overrides.csv`. Results without the overrides are reported as a sensitivity analysis (`analysis_supp.json`, key `sensitivity_no_review_lead_overrides`).
6. **Cap of 60** (§6) was not triggered; all included papers were coded.
7. **Tooling.** Text extraction used the main review's `extract_text.py` with pypdf in a scratch virtual environment.
