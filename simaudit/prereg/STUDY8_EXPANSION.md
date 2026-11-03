# Study 8: stratified dataset expansion for the evaluation paper

Status: FROZEN 2026-10-21. Committed before any candidate dataset was downloaded
and before any model was run on it. Changes only by dated amendments at the end.
The pool is built by `simaudit build-pool-ext`
(`src/simaudit/runner/build_pool_ext.py`); the rule below is what that code does.

## 1. Why

The DEV set (20 OpenML datasets) is narrow:
- 14 of 20 are binary;
- the median imbalance ratio is 1.6;
- n lies between 100 and 1000.

LUCoS used 67 CC18 datasets. The protocol bias we study (stratified "m per class" against a realistic total budget B = m·C) should be largest with many classes and strong imbalance (`reports/COVERAGE_MODEL.md`), and DEV barely covers those settings. Study 8 adds about 72 datasets, stratified by the number of classes C and the imbalance ratio IR (majority size / minority size), drawn by the rule below before any result on them exists.

## 2. Sources

All candidates come from one query of OpenML's dataset list (`list_datasets`, status `active`) and its precomputed qualities.

Curated suites (tier 1) are resolved by suite ID at query time:

| Key | OpenML suite | Role |
|---|---|---|
| `cc18` | 99, OpenML-CC18 | the benchmark LUCoS used |
| `amlb_classification` | 271, AutoML Benchmark classification | the curated successor to CC18 |
| `tabarena_v0.1` | 457, TabArena v0.1 (the lite variant uses the same tasks with fewer repeats) | the current FM benchmark |

- We know of no OpenML suite published under the name "OpenML-CC21". The AMLB classification suite takes the role of a curated CC18 successor.
- If any suite fails to resolve, the build stops; it does not continue without that suite.

Tier 2: every other OpenML dataset that meets Sec. 3. Suite membership only orders candidates within a stratum (Sec. 5). It never makes a dataset eligible.

## 3. Eligibility (OpenML metadata only, before download)

Rules are checked in this order, and the audit records the first rule a dataset fails:
1. status `active`;
2. format not sparse;
3. classification: NumberOfClasses and the class-size qualities are present;
4. 2 ≤ C ≤ 10, the class limit shared by TabPFN v2, TabPFN v3.5, TabICL and Kumo;
5. 100 ≤ n ≤ 5000 (see the cap note below);
6. 2 ≤ d ≤ 100 features, as in the pilot;
7. all features numeric (NumberOfSymbolicFeatures = 1, only the target);
8. no missing values;
9. minority class ≥ 11, so that m = 10 per class still leaves unlabeled rows.

**Encodable as numeric.** This means stored as numeric by OpenML, as the existing loader requires. Binary and ordinal features stored as numbers qualify. Categorical (symbolic) features are not admitted. One-hot encoding would add a preprocessing choice that DEV does not have, and it changes the distance geometry that the label-free selection strategies use.

**The n cap at 5000, and why.** Study 4 measured the cost of the four FMs: all four learners over the full Study 4 grid took about 7.9 wall-hours at 4 workers on DEV, where Σn = 8363 (Sec. 9).
- Cost grows roughly linearly with n, because each fit predicts every unselected row.
- The distance-based selectors (k-medoids, ProbCover, Σ-optimality) need n × n matrices: 200 MB at n = 5000 and 800 MB at n = 10 000, per worker.
- At n = 5000, one dataset costs about 19 worker-hours on that grid. A cap of 5000 keeps the largest dataset under a day of wall time at 4 workers, and keeps every n within the range that TabPFN v2 recommends.

**Exclusions against DEV and HELD-OUT.** The protected set is the 52 DEV and HELD-OUT entries of `data/pool_i2.yaml`: IDs, names and SHA-256 content hashes. These are non-sealed metadata. The sealed copy in `../sealed_heldout/` is not opened. A candidate is excluded if any of the following holds:
10. its ID is a DEV or HELD-OUT ID;
11. its name key is a DEV or HELD-OUT name key (lower case, non-alphanumerics removed);
12. its family key is a DEV or HELD-OUT family key (the name key truncated at its first digit, the pilot's family rule);
13. its signature (n, C, majority size, minority size) equals that of a DEV or HELD-OUT dataset, taken from OpenML metadata for those IDs. This catches re-uploads, re-versions and feature views of the same rows;
14. after download, its (X, y) SHA-256 equals a DEV or HELD-OUT hash, or that of an already accepted candidate.

**No near-duplicate versions inside the new pool.** Within the candidates that pass rules 1 to 13, the following are applied in sequence. In each step, ties go to the suite member first, then to the lowest version, then to the lowest ID:
- one dataset per name key;
- one per family key;
- one per signature.

**PRIORSEL IDs are not excluded.** The pilot excluded the 124 PRIORSEL IDs because those datasets were chosen for a graph method's advantage. That selection is unrelated to the contrasts here, which compare label-sampling protocols and selectors. CC18 overlaps heavily with PRIORSEL. Each pooled dataset therefore carries a `priorsel_overlap` flag, and Sec. 7 reports a sensitivity analysis without the flagged datasets.

## 4. Strata and quotas

| | IR < 2 | 2 ≤ IR ≤ 5 | IR > 5 |
|---|---|---|---|
| C = 2 | 8 | 8 | 8 |
| C = 3–4 | 8 | 8 | 8 |
| C = 5–10 | 8 | 8 | 8 |

- The target is 72 datasets.
- IR and C come from OpenML metadata when the dataset is drawn. After download, they are recomputed from the loaded y.
- If the loaded data falls in a different stratum, the dataset is rejected ("stratum mismatch") and the next candidate in the same stratum is tried.

## 5. Sampling rule

**Seed.** `numpy.random.default_rng([20261020, 8])`, where 8 is the Study 8 namespace.

**Queues.** For each stratum in the fixed order (C bins ascending, then IR bins ascending), the tier 1 IDs (sorted, then permuted) come first and the tier 2 IDs (sorted, then permuted) after them. The permutations are drawn in that order from the one generator.

**Primary pass.** Candidates are taken from the front of each queue. Each is downloaded with the existing loader (`simaudit.data.openml_loader.load`, into the existing `data/cache`) and post-load checked:
- n, d, C and minimum class size against Sec. 3;
- a unique content hash;
- a matching stratum.

A failure is recorded and the next candidate is taken. Each stratum stops at its quota of 8.

**A short stratum.**
- It contributes everything it has.
- The shortfall is reallocated so that the total still reaches 72. The order is (C 5–10, IR > 5), (C 5–10, 2–5), (C 5–10, < 2), (C 3–4, > 5), …, (C 2, < 2): the strata where the bias should be largest come first.
- One more dataset is taken from each stratum in that order that still has candidates, cycling until the total is 72 or every queue is empty.
- The final per-stratum counts are what the analysis uses. The moderator analysis (Sec. 7) uses C and IR as measured, so unequal cells reduce power but cause no bias.

**Downloads.** These run sequentially, one at a time, since the order of the draw is the rule. Downloaded files are only parsed by the OpenML client and the loader; nothing downloaded is executed.

**Freezing.** The output is frozen to `data/pool_ext.yaml`:
- the seed, quotas and suite sizes;
- per-stratum counts (eligible, tier 1, sampled);
- for every sampled dataset: ID, name, version, SHA-256, stratum, n, d, C, IR, tier, suites and the PRIORSEL flag.

A transient download failure is a recorded rejection. The pool is not re-drawn afterwards, which is the pilot's precedent.

## 6. Exclusion audit

`data/pool_ext_audit.csv` lists every OpenML dataset that is not in the pool, with columns did, name, stage and reason:
- stage `metadata`: the first failing rule of Sec. 3, or the de-duplication step;
- stage `post-load`: load errors, post-load checks, hash duplicates and stratum mismatches;
- stage `not drawn`: eligible, but its stratum's quota was met.

`pool_ext.yaml` also holds the counts per reason.

## 7. Confirmatory role and analysis

**An independent confirmation set.** The expansion pool is a second confirmation set for Study 7's confirmatory contrasts, in addition to HELD-OUT.
- It is analysed once, with the Study 7 analysis exactly as frozen before Study 7's DEV run: the same contrasts, metric, budgets, seeds rule, tests and multiplicity correction.
- No arm, hyperparameter or contrast may be chosen or changed after any expansion result is seen.
- Study 7 is extended to this pool only after its DEV run, with its analysis frozen. No model runs on this pool before then; Study 8 runs none.

**Preregistered moderators.** The unit is the dataset, with the score averaged over repetitions.
- **M1 (the class count).** The protocol gap Δ_ST−RR (stratified − realistic random, macro-F1, FM5) grows with C. Test: one-sided Kendall τ between the per-dataset Δ and C, at each budget multiplier mult ∈ {1, 3, 5, 10}.
- **M2 (the imbalance).** Δ_ST−RR grows with IR. Test: one-sided Kendall τ between Δ and log IR, at each mult.
- **Dose-response.** The per-dataset Δ is regressed on log C and log IR (OLS, HC3 errors), at each mult. Both slopes are predicted positive.
  - The interaction term and the trend over the three C bins and the three IR bins (Jonckheere–Terpstra) are reported as secondary.
  - The same regression is fitted for each Study 7 confirmatory contrast. Prediction: the selector advantage over realistic random also grows with C and IR, because coverage failures of random sampling grow with both.
- **Multiplicity and power.** Holm across the 2 moderators × 4 budgets = 8 primary tests. A moderator is confirmed if its τ survives Holm at α = 0.05 for mult = 1 and has the same sign at every mult. With about 72 datasets, a τ of about 0.27 is detectable at 80% power (one-sided, α = 0.05/8; null variance of Kendall τ).
- **Coverage model.** The analytic coverage model (`reports/COVERAGE_MODEL.md`), applied unchanged, predicts each dataset's gap from its class counts alone. Its correlation with the observed gap is reported as an out-of-sample test of that model.

**Sensitivity, descriptive.** Each of the following is reported:
- the analysis without the `priorsel_overlap` datasets;
- tier 1 only;
- n ≤ 1000, the DEV range, against n > 1000.

## 8. Threats

- **The sampling frame is OpenML.** Many OpenML datasets are derived or synthetic. The family and signature rules remove copies, but not every derivative. The audit lists what was kept.
- **Numeric-only features under-represent categorical tables.** This is a deliberate trade for preprocessing parity with DEV.
- **Imbalance and n are correlated with C**, since a minority class of 11 or more caps IR at n/11. The dose-response regression includes both log C and log IR. We also report it with log n as a covariate (secondary).
- **Metadata errors** (a wrong default target, stale qualities) are caught by the post-load stratum check, never corrected by hand.

## 9. Compute estimate (no run is part of Study 8)

**Measured on DEV.** All four FMs (FM1, FM2, FM3 and FM5) over the Study 4 grid (4 budgets × 10 strategies; 10 reps for random and 5 for the rest) took:
- FM5 ≈ 2.1 h, FM1 ≈ 1.8 h, FM2 ≈ 1.0 h and FM3 ≈ 3.0 h of wall time at 4 workers;
- 7.9 h in total for Σn = 8363 rows;
- that is ≈ 0.95 wall-hours per 1000 rows at 4 workers, or ≈ 3.8 worker-hours per 1000 rows.

**Expansion.** 72 datasets. Σn is known only after the draw and is recorded in the freeze note (Sec. 10).
- If the mean n is 1500 (Σn ≈ 108 000), the same grid costs about 100 wall-hours at 4 workers, or about 50 at 8.
- Cost scales linearly with the size of Study 7's grid relative to Study 4's.
- If Study 7's grid is smaller (fewer strategies or FMs), the cost scales down proportionally.

**Download.** The datasets are small (≤ 5000 × 100); the expected cache growth is well under 1 GB.

## 10. Amendments and freeze note

### Amendment 1 (2026-10-21): Tier 2 and Tier 3, a second official source

Committed before any expansion data was downloaded, from OpenML or from PMLB, and before any model was run on either. The only data read so far is metadata: PMLB's `all_summary_stats.tsv` at the pinned commit below (SHA-256 `d658820a844a555776273a2c1b6766eba983bb078e3bcc539141b1ac2cc33960`). The OpenML REST API has been unreachable since 2026-10-21 (openml/openml.org issue #423), so nothing has been drawn.

**Why.** Study 9 (`prereg/STUDY9_HOW_MANY_LABELS.md`) estimates the typical label-saturation point per dataset. The authors want clearly more than 72 datasets for that estimate.

**Tier 1 is unchanged.** It is exactly the rule of Sec. 2–9: 72 OpenML datasets in 9 strata, the full Study 7 grid, and the confirmatory role of Sec. 7. Nothing below changes which datasets enter Tier 1, nor their order. Study 8's own confirmatory analysis (Sec. 7) stays on Tier 1 only.

**Naming.** Sec. 2 and 5 call suite members "tier 1" and other OpenML datasets "tier 2". From this amendment on:
- those are *source* classes: `source: suite` or `source: openml`;
- **Tier** means the expansion tier defined here.

In `pool_ext.yaml`, every dataset carries `tier` (1, 2 or 3), `subtier` (`2a` or `2b` for Tier 2) and `source` (`suite`, `openml` or `pmlb`). The per-stratum key `tier1` (eligible suite members) becomes `suite_members`.

#### A1.1 Sources

| Tier | Source | Rule |
|---|---|---|
| 1 | OpenML (Sec. 2–5) | unchanged, 72 datasets |
| 2a | PMLB, numeric-only | every eligible dataset (A1.2), minus duplicates (A1.3) |
| 2b | OpenML, the remainder | eligible under Sec. 3 and not drawn into Tier 1, minus duplicates, balanced quotas, up to the cap (A1.4) |
| 3 | PMLB, integer-coded categoricals | every eligible dataset with a categorical feature, minus duplicates; **secondary only** (A1.5) |

- **PMLB.** The Penn Machine Learning Benchmarks (EpistasisLab/pmlb, MIT licence for the repository; each dataset keeps the licence of its original source). Pinned commit `7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68` (2025-02-25). Files are fetched by plain HTTPS from `raw.githubusercontent.com/EpistasisLab/pmlb/<commit>/datasets/<name>/<name>.tsv.gz` into `data/cache/pmlb/`, with no PMLB package and no executed code. The SHA-256 of every file is recorded in the pool.
- **Not used.**
  - Unofficial mirrors of OpenML (e.g. Hugging Face copies of CC18): their provenance cannot be verified.
  - TabArena/BeyondArena on Hugging Face: official, but its licence is "other" and its datasets are mostly larger than n = 5000. Out of scope.
- **Independence from timing.** Every tier is defined by content, not by when a source is reachable. Duplicates are resolved by a fixed priority, whatever the order of the builds: DEV/HELD-OUT > Tier 1 > Tier 2a > Tier 3 > Tier 2b.
  - PMLB tiers (2a, 3) can be built now.
  - Tier 1 and Tier 2b are built when OpenML returns. At that point, any PMLB dataset that duplicates a Tier 1 dataset or HELD-OUT is removed from its PMLB tier (A1.3). Any OpenML candidate that duplicates a PMLB dataset is removed from the Tier 2b queue.
  - Results already computed on a removed dataset are discarded from every analysis.

#### A1.2 PMLB eligibility (metadata at the pinned commit, then post-load)
Rules, checked in order; the audit records the first rule a dataset fails:
1. `task == classification`, name not starting with `_deprecated_`;
2. 2 ≤ C ≤ 10, 100 ≤ n ≤ 5000, 2 ≤ d ≤ 100 (the Sec. 3 ranges);
3. **numeric-only (Tier 2a):** `n_categorical_features == 0` in PMLB's summary. This is PMLB's analogue of Sec. 3 rule 7. PMLB stores every feature as a number and flags integer-coded categoricals in this column. Binary features (0/1) are admitted, as Sec. 3 admits binary features stored as numbers. Datasets with `n_categorical_features > 0` go to Tier 3 (A1.5), not to Tier 2a.

Post-load, on the downloaded file:
- the target column is `target`; y is encoded as in `openml_loader.encode` (sorted strings);
- the Sec. 3 ranges and minority class ≥ 11 are rechecked;
- the data must have no missing value and no non-numeric column.

C and IR are computed from the loaded y; PMLB has no class-size metadata. The stratum is therefore set after loading, so a stratum mismatch cannot occur.

#### A1.3 Duplicates (PMLB against DEV, HELD-OUT, Tier 1 and itself)
Many PMLB datasets are copies of OpenML or UCI tables. A PMLB dataset is excluded if any of the following holds:
- its name key or family key (Sec. 3, rules 11–12) equals that of a DEV or HELD-OUT dataset;
- its content hash (`content_hash` of the loaded X, y) equals a DEV or HELD-OUT hash;
- its signature (n, C, majority size, minority size), computed from the loaded data, equals that of a DEV or HELD-OUT dataset.
  - **DEV signatures** come from the cached DEV data (non-sealed).
  - **HELD-OUT signatures** need OpenML's qualities for the HELD-OUT IDs. `sealed_heldout/` is never opened. Until OpenML returns, the PMLB pool records `heldout_signature_check: pending`, and no PMLB dataset is analysed before that check has passed.
- when Tier 1 exists: its name key, family key, hash or signature equals that of a Tier 1 dataset (Tier 1 wins).

Within PMLB (2a and 3 together), one dataset is kept per name key, then per family key, then per signature, in this order. Ties go to Tier 2a over Tier 3, then to the alphabetically first PMLB name. The PMLB names carry no versions.

*Expected effect (inference from names and summary sizes only, not a result):* several numeric PMLB candidates repeat DEV or HELD-OUT tables:
- iris, vehicle, prnn_synth, spectf, the waveform family;
- breast_cancer_wisconsin_diagnostic, which is wdbc;
- appendicitis.

The mfeat views, led7/led24 and the Hill_Valley pair share signatures. Of the 40 numeric metadata candidates, about 25–30 should remain.

#### A1.4 Tier 2b and the cap
- **Cap.** |Tier 1| + |Tier 2| ≤ **200**, so Tier 2 ≤ 128 if Tier 1 reaches 72. Tier 3 is outside the cap (A1.5).
- **Order.** Tier 2a enters in full. Tier 2b fills the rest of the cap.
- **Balanced quotas for Tier 2b.** The per-stratum target for Tier 1 + Tier 2 is 22 (≈ 200 / 9). In stratum s, Tier 2b takes up to max(0, 22 − |Tier 1 ∩ s| − |Tier 2a ∩ s|) datasets. Any shortfall is reallocated by Sec. 5's rule: one more dataset from each stratum in `REALLOC_ORDER` that still has candidates, cycling until the cap is reached or every queue is empty.
- **Queues.** These are the Sec. 5 queues (same seed, same order), continued after the Tier 1 draw. The candidates Tier 1 popped (accepted or rejected) are not revisited. OpenML candidates that duplicate a Tier 2a or Tier 3 dataset (name, family or signature, by metadata; hash after download) are removed first. The post-load checks are those of Sec. 5.
- **Why 200 (compute).** These are Study 9 costs; Study 7's grid on Tier 1 is separate (Sec. 9).
  - The Study 4 timings (Sec. 9) give, per 1000 evaluated rows per fit at mean mult ≈ 5: FM2 ≈ 7.8 s, FM1 ≈ 14 s, FM5 ≈ 16 s and FM3 ≈ 23 s of worker time.
  - Study 9's grid is about 189 fits per dataset and learner; its contexts are larger (mean m ≈ 10), counted as ×1.5.
  - Study 9 caps the evaluated rows at 2000 per dataset. With an assumed mean of 900 evaluated rows, that is ≈ 4.4 worker-hours per Tier 1 dataset (4 FMs) and ≈ 1.5 per Tier 2 or Tier 3 dataset (FM1 + FM2).
  - Totals: 72 × 4.4 + 128 × 1.5 + 60 × 1.5 ≈ 600 worker-hours, about 6 days at 4 workers.
  - A Tier 2 above 128 would add more than a day of wall time for each further 60 datasets, for a gain in precision that A1.6 shows is small.
  - The DEV dry-run of Study 9 measures the real cost. The freeze note records it, with the actual Σn.

#### A1.5 Tier 3 (secondary)
PMLB datasets that pass A1.2 except that `n_categorical_features > 0` (integer-coded categoricals), after A1.3.
- **Why secondary.** Integer codes impose an arbitrary order and distance on nominal values. That is the preprocessing choice Sec. 3 avoids for DEV parity.
- **Use.** Tier 3 enters only the analyses that Study 9 marks as Tier 3 sensitivity (the saturation distribution and the coverage budget). It never enters a confirmatory test of Study 7, 8 or 9.
- **Size.** No cap; the metadata lists 58 such candidates before deduplication.

#### A1.6 Expected counts and shortfalls (stated before any download)
- **PMLB numeric (2a).** By metadata, before deduplication:
  - 26 binary, 7 with C = 3, 3 with C = 4, and 4 with C ≥ 5;
  - the multiclass PMLB tables with C ≥ 6 (yeast, wine_quality_*, ecoli) mostly fail the minority ≥ 11 rule.
  - Tier 2a will be dominated by C = 2 with IR < 2.
- **OpenML remainder (2b).** Unknown until OpenML returns.
  - The pilot's frame (n ≤ 1000, PRIORSEL IDs excluded) had 72 metadata-eligible datasets, of which 52 were valid and unique.
  - The Study 8 frame is wider: n ≤ 5000, PRIORSEL IDs included, suite members.
  - A plausible eligible remainder after Tier 1 is 50–150 datasets. It is concentrated in C = 2 and IR < 2.
- **Expected shortfalls.**
  - Strata C 5–10 × IR > 5 and C 3–4 × IR > 5 will almost surely miss the per-stratum target of 22. A total of 3–8 datasets each across all tiers is a realistic expectation.
  - The total of Tier 1 + Tier 2 may fall below 200. The realistic range is 110–200, plus up to about 45 in Tier 3.
  - The final counts per stratum and tier go into the freeze note. Study 9's analyses use C and IR as measured, so unequal cells cost power, not validity.
- **Precision gained.** The half-width of a 95% CI for a share (e.g. P(m* ≤ 10)) near 0.5 is ≈ 0.115 at N = 72, 0.08 at N = 150 and 0.07 at N = 200.

#### A1.7 Outputs
- `simaudit build-pool-pmlb` writes `data/pool_ext_pmlb.yaml` and `data/pool_ext_pmlb_audit.csv`, which hold Tiers 2a and 3 with:
  - the commit and the summary file's hash;
  - for every candidate: file SHA-256, content hash, n, d, C, IR, stratum, tier and the first failed rule.
- `simaudit build-pool-ext` (OpenML) then writes `data/pool_ext.yaml`, which holds every tier with the fields above. It applies A1.3 against Tier 1 and HELD-OUT, and removes the PMLB entries that fail. The audit file gains the stages `tier-2b not drawn` and `duplicate of PMLB`.
- Both files are committed with a freeze note before any model runs on them.

**Unchanged.** Sec. 3's eligibility, exclusion and deduplication rules; the seed; the Tier 1 quotas, queues and reallocation; and Sec. 7 (Tier 1 only).

### Freeze note, PMLB part (2026-10-21)
`simaudit build-pool-pmlb` was run once, after Amendment 1 was committed (R039) and after the Git LFS fix (the files at the commit are LFS pointers; the content now comes from GitHub's media host and is checked against the pointer's SHA-256). The first attempt stored only the pointers and rejected every dataset. Those files were deleted before any parse, and the build was re-run with no other change.

**Result.** `data/pool_ext_pmlb.yaml` and `data/pool_ext_pmlb_audit.csv`.

| Stratum | Tier 2a | Tier 3 |
|---|---|---|
| C2, IR<2 | 13 | 10 |
| C2, IR 2–5 | 4 | 3 |
| C2, IR>5 | 1 | 4 |
| C3–4, IR<2 | 2 | 2 |
| C3–4, IR 2–5 | 2 | 2 |
| C3–4, IR>5 | **0** | 4 |
| C5–10, IR<2 | 2 | 2 |
| C5–10, IR 2–5 | **0** | 2 |
| C5–10, IR>5 | 2 | 1 |
| **Total** | **26** (Σn = 21 413) | **30** (Σn = 44 643) |

**Exclusions after metadata.**
- *DEV/HELD-OUT content:*
  - by hash: analcatdata_creditscore, hayes_roth, wine_recognition;
  - by signature against DEV: appendicitis, breast_cancer_wisconsin_diagnostic (= wdbc), credit_approval_germany.
- *Missing values:* 12.
- *Minority < 11:* allhyper, lymphography, wine_quality_red, wine_quality_white.
- *Within-PMLB deduplication:* 11, among them led7, the mfeat views and the GAMETES and monk families.

**Still pending.** `heldout_signature_check` and `tier1_duplicate_check` both read `pending`. They run in `build-pool-ext` when OpenML returns, and no PMLB dataset is analysed before then.

**Shortfall, as expected in A1.6.** Tier 2a has no dataset in C3–4 × IR>5 or C5–10 × IR 2–5. Tier 3 is more multiclass and more imbalanced than Tier 2a, but it is secondary.

### Amendment 2 (2026-10-21): Tier 4 (UCI), the OpenML-free confirmation set T1′, HELD-OUT checks without OpenML, and data-integrity screens

Committed before any Tier 4 data file was downloaded and before any model was run on any expansion dataset (Tier 1, 2a, 2b, 3 or 4). OpenML is still unreachable (openml/openml.org issue #423).

**What had been read when this was written.** Only metadata:
- the UCI API dataset list and per-dataset metadata, frozen as `data/uci_api_snapshot.json.gz` (SHA-256 `1f523469b00b0eaec11695f8b437131fcb638eb0c955dd8f194ec0bd5683de46`, written by `simaudit snapshot-uci`);
- the licence line of each UCI dataset page (stored in the snapshot);
- the card and file list of TALENT and of inria-soda/tabular-benchmark on Hugging Face (A2.1).

Also already known: the PMLB Tiers 2a and 3 (Amendment 1 freeze note), whose class counts were computed at that build. No result of any model on any expansion dataset exists.

**Why.** Studies 7, 9 and 10 need an independent confirmation set besides HELD-OUT, and Tier 1 cannot be built while OpenML is down. This amendment adds a second official source and fixes now, independently of any result, which set replaces Tier 1 if OpenML is still down when confirmation is due.

#### A2.1 Sources evaluated (2026-10-21)

| Source | Official? | Licence | Provenance | Fit to Sec. 3 | Decision |
|---|---|---|---|---|---|
| **UCI ML Repository**, official API (`archive.ics.uci.edu/api`) | yes | CC BY 4.0 on 203 of the 208 API datasets; 5 say "see linked dataset" | per-dataset DOI, creators, last-updated date; the data file is served by UCI itself | 23 pass the metadata rules of A2.2 | **adopted as Tier 4** |
| **TALENT** (LAMDA-Tabular) | yes (authors' GitHub, MIT; data on Google Drive and HF `LAMDA-Tabular/TALENT`, card licence MIT) | MIT on the repackaging; the original licences of the OpenML/UCI/Kaggle sources are not listed per dataset | pre-split `.npy` repackagings (train/val/test) of tables from other repositories, with no source ID per dataset | the metadata (`info.json`) is only inside 216–269 MB zips, so nothing can be selected before download | **rejected**: a derived copy whose per-dataset provenance and licence cannot be verified, and it would duplicate OpenML/UCI tables without a source key |
| **inria-soda/tabular-benchmark** (Grinsztajn et al., HF, commit `8d0ff9103525b7e3579b180230fddb3186258301`) | yes | none stated on the card | each config names its OpenML origin | 23 classification configs, all binary and balanced by construction; only Bioresponse (n = 3434, d = 419) and compas-two-years (n = 4966, categorical features) have n ≤ 5000. Without subsampling, **0** are eligible | **rejected** |
| Unofficial mirrors (e.g. Hugging Face copies of OpenML suites) | no | — | not verifiable | — | **rejected** (as in A1.1) |

**UCI scope.** The API lists 689 datasets. It serves 209 of them as a single `data.csv` file with variable roles: the set that the official `ucimlrepo` client can import. One of these (ID 462) returns 404, which leaves 208. The other 480 are distributed only as archives in heterogeneous formats, where choosing the target and parsing the file would need a per-dataset judgement. They are excluded. The `ucimlrepo` package is not used: the loader reads the same `data.csv` and the same variable roles directly (`src/simaudit/runner/build_pool_uci.py`).

#### A2.2 Tier 4 eligibility (snapshot metadata, then post-load)

Rules are checked in this order, and the audit records the first rule a dataset fails:
1. the API serves a data file under `https://archive.ics.uci.edu/static/public/`;
2. `tasks` includes Classification;
3. the dataset page links a Creative Commons licence;
4. exactly one target column;
5. 100 ≤ `num_instances` ≤ 5000;
6. 2 ≤ d ≤ 100, where d counts the variables whose role is Feature. Columns with role ID or Other are dropped and never used as features;
7. every Feature variable has type Continuous, Integer or Binary. This is the analogue of Sec. 3 rule 7: Categorical or Date features exclude the dataset, and Tier 4 has no categorical sub-tier;
8. `has_missing_values` is "no";
9. the name rules of A2.3.

**Post-load,** on the downloaded `data.csv`:
- the Feature-role columns form X and the target column forms y. They are encoded with `openml_loader.encode` (numeric dtype required, no missing value, y in sorted string order);
- the Sec. 3 ranges and minority class ≥ 11 are rechecked;
- C and IR are computed from the loaded y, and the stratum is set from them, as for PMLB;
- no subsampling, and no change to rows or columns other than the A2.7 drop rule.

**Metadata counts (snapshot of 208).** n outside the range: 49; non-numeric features: 45; not classification: 35; not exactly one target: 28; DEV/HELD-OUT name (A2.3): 9; missing values: 8; PMLB name: 4; no Creative Commons licence: 4; d outside the range: 3. **23 pass.**

#### A2.3 Duplicates

**Priority.** DEV/HELD-OUT > Tier 1 > Tier 2a > Tier 3 > **Tier 4** > Tier 2b. Under the trigger of A2.5, T1′ members take precedence over Tier 1.

**Against DEV and HELD-OUT, by name (metadata, conservative).** The rule is stricter than A1.3, because the HELD-OUT signature check is still pending (A2.6) and UCI hosts the originals of several HELD-OUT tables. A candidate is excluded if its name and a DEV or HELD-OUT name satisfy any of:
- equal name keys or family keys (Sec. 3, rules 11–12);
- one name or family key (≥ 4 characters) lies inside the other;
- the protected name's head token (its first alphanumeric token of ≥ 4 characters that is not one of: data, dataset, database, test, clean, numeric) is a token of the candidate's name;
- the protected name key is a token of the candidate's name.

On the snapshot, this excludes Hayes-Roth, Iris, SPECTF Heart, Waveform (v1), Wine, Wine Quality, Statlog (Vehicle Silhouettes), Blood Transfusion Service Center and ILPD. Over-exclusion only costs datasets; the rule is fixed before any data is read.

**Against DEV and HELD-OUT, by content.**
- The (X, y) content hash is checked against the 52 DEV and HELD-OUT hashes in `data/pool_i2.yaml`.
- The signature is checked against DEV, using the cached DEV data.
- The signature check against HELD-OUT follows A2.6.

**Against PMLB (Tiers 2a and 3, frozen).**
- Exact name key and family key, on metadata.
- After loading:
  - the (X, y) hash;
  - the X-only hash (row-sorted features, which catches the same rows under another target);
  - the signature (n, C, majority size, minority size) of every PMLB pool dataset.

**Within Tier 4.** One dataset per name key, then per family key, then per signature, then per X-only hash. The lowest UCI ID wins.

**Against Tier 1, when it exists.** The rule is that of A1.3 (name, family, hash, signature). Tier 1 wins, except under the trigger of A2.5.

#### A2.4 Tier 4 size

- Tier 4 enters in full: every dataset that survives A2.2–A2.3 and A2.7. This is the Tier 2a precedent; the source is too small for a quota within the tier.
- Tier 1's quotas, seed rule and reallocation apply when T1′ is drawn (A2.5).
- **If the trigger of A2.5 does not fire,** Tier 4 is used as Tier 2a is: in Study 9's EXP (FM1 and FM2), and inside the Tier 1 + Tier 2 cap of 200, before Tier 2b.

#### A2.5 The OpenML-free confirmation set T1′

**Definition.** T1′ is drawn from Tier 2a ∪ Tier 3 ∪ Tier 4, after A1.3, A2.3 and A2.7, with Tier 1's quotas and rules (Sec. 4–5):
- **Queues.** For each stratum in the fixed order, the numeric-only datasets (Tiers 2a and 4) come first, then Tier 3. Within each group, the keys are sorted and then permuted with `numpy.random.default_rng([20261020, 8, 2])` (Study 8 namespace, sub-stream 2). The permutations are drawn in stratum order from that one generator.
- **Draw.** At most 8 per stratum. Any shortfall is reallocated in `REALLOC_ORDER`, one dataset at a time, until 72 or every queue is empty. This is the code of Sec. 5 (`build_pool_ext.draw`) with an always-accepting check, since every candidate is already loaded and checked.
- **Balanced core.** The first 8 of each queue, i.e. the primary pass without reallocation.
- **Output.** `simaudit build-t1prime` writes `data/pool_t1prime.yaml` (`src/simaudit/runner/t1prime.py`).

**Why Tier 3 is in T1′.** Without it, the numeric-only pool (about 26 + 12 datasets) cannot approach Tier 1's 72. It would also leave C3–4 × IR>5 and C5–10 × IR 2–5 nearly empty, and these are the strata where the protocol bias should be largest (Sec. 1). Tier 3's integer-coded categoricals are the reason A1.5 made Tier 3 secondary. This amendment changes A1.5 only for T1′ under the trigger. The numeric-only sensitivity below isolates their effect.

**Trigger (fixed now; it depends on no result).**
- Let T* be the date of the first of these two events:
  - the Study 7 DEV report is committed with its analysis frozen, which is the condition of Sec. 7 for extending Study 7;
  - the Study 9 freeze note of its Sec. 10 is committed.
- On T*, a dated note in this file records whether `data/pool_ext.yaml` (Tier 1) has been committed with its freeze note.
- **If it has not, the trigger fires.** T1′ then replaces Tier 1 in every confirmatory use:
  - Study 8 Sec. 7 (the moderators and the Study 7 extension);
  - Study 9's EXP (EXP′ = T1′, with all four FMs on every member, which is Tier 1's learner set);
  - Study 10's primary set (HO ∪ T1′).
- The choice is made once and is irreversible. A Tier 1 that is built later does not undo it.
- **If it has, the trigger does not fire.** T1′ is not used, and the preregistered Tier 1 roles stand.

**Tier 1 after a trigger.**
- When OpenML returns, Tier 1 is still built by Sec. 2–5, unchanged. For Study 9, Tier 2b is built as well.
- Any Tier 1 or Tier 2b dataset that duplicates a T1′ member (A1.3 rules) is removed from Tier 1 or 2b, not from T1′.
- Tier 1 (for Study 9, Tier 1 ∪ Tier 2b) is then analysed once, with the same frozen code, as a **replication**. It is reported next to the confirmatory T1′ result, never pooled into it, and it does not change any decision.

**Pre-specified sensitivity on T1′.** Each of the following is reported, with no confirmatory weight:
- the numeric-only subset (Tiers 2a and 4);
- the balanced core;
- each source separately;
- equal weight per stratum (post-stratified).

**Study 10's T2a arm.** Under the trigger, T2a ⊂ T1′, so Study 10's optional secondary T2a arm (its Sec. 3.1) is void.

#### A2.6 HELD-OUT checks without OpenML

**What is available outside the sealed folder.** `data/pool_i2.yaml` lists, for the 32 HELD-OUT datasets, the ID, name, version and SHA-256 content hash only; it has no n or class sizes. No HELD-OUT file is in the non-sealed caches: `data/cache/npz` and `data/cache/openml` hold no HELD-OUT ID (checked 2026-10-21). `sealed_heldout/` was not opened.

**Done now, for Tiers 2a, 3 and 4:**
- the name rules (A1.3 for PMLB; the stricter A2.3 rule for Tier 4);
- the (X, y) hash against the 32 HELD-OUT hashes.

**Pending: the signature check (n, C, majority size, minority size) against HELD-OUT.** It runs at the earliest of:
- (a) OpenML returning: the HELD-OUT qualities are metadata, as in A1.3;
- (b) the first opening of HELD-OUT by any study;
- (c) if the authors authorise it: a labels-only extraction. A script, committed before it is run, reads only the HELD-OUT target vectors and writes the 32 signatures. No feature and no model is involved, and the event is logged as a HELD-OUT access.

**Consequence.**
- Any T1′ member (or Tier 2a/3/4 dataset) whose signature or hash equals a HELD-OUT dataset's is removed from T1′ **before any T1′ confirmatory analysis is computed**. It is not replaced, and the removal is recorded in the freeze note.
- Runs on T1′ may proceed before the check, but no T1′ result is analysed until the check has run.
- The pool files keep `heldout_signature_check: pending` until then.

#### A2.7 Data-integrity screens (every expansion tier)

**Why.** The DEV audit (`research_notes/Systematic discovery 2026-10/22_row_order_leakage_fix.md`, Sec. 7) found three problems:
- rows stored sorted by class;
- a counter column whose 1-NN recovers the class (OpenML 694);
- duplicate rows with conflicting labels.

**Scope and timing.** The screens below apply to every expansion dataset: Tiers 2a, 3 and 4 now, and Tier 1 and Tier 2b when they are built. They run on the stored row order, before any model runs. DEV and HELD-OUT are not re-screened here.

**Code.** `src/simaudit/data/integrity.py` (thresholds as module constants) and `src/simaudit/runner/screen_integrity.py`.

**Results.** `data/pool_ext_integrity_audit.csv`, one row per dataset with:
- κ;
- the sorted flag;
- the ID-like and dropped columns;
- the duplicate and conflict counts and the share;
- the decision.

**S1, rows sorted by class (flag only).**
- κ = (P(same class | adjacent rows) − P(same class | random pair)) / (1 − P(same class | random pair)).
- The dataset is flagged if κ ≥ 0.5.
- No exclusion follows. Every runner on expansion data applies a fixed row permutation: Study 9 namespace 13, Study 10 namespace 14. For the Study 7 extension, the flag is reported with the permuted-row sensitivity of Study 7 Addendum 1.

**S2, ID-like or counter columns.**
- A column is **ID-like** if either:
  - |Spearman ρ| between it and the stored row index is ≥ 0.99; or
  - it is integer-valued with at least 95% unique values.
- An ID-like column is **leaky** if leave-one-out 1-NN on that column alone (ties go to the lower value) reaches a balanced accuracy ≥ 0.90.
- **Leaky columns are dropped** by every runner. The pool lists them as `drop_columns`. If fewer than 2 columns remain, the dataset is **excluded**.
- ID-like columns that are not leaky are kept and reported.
- *Calibration:* DEV 694's counter has ρ = 1.00, 310 unique values in 310 rows, and a leave-one-out 1-NN accuracy of 1.00.

**S3, duplicate rows with conflicting labels.**
- Count the rows in groups of identical X that carry more than one label.
- If they exceed **5% of the rows**, the dataset is **excluded**. Otherwise, the count is reported.

**Outcome.** These decisions are mechanical. The T1′ draw (A2.5) skips excluded datasets before queueing.

#### A2.8 Expected counts and shortfalls (stated before any Tier 4 download)

**Tier 4.** 23 datasets pass metadata.

*Expected effect, an inference from names and sizes only, not a result:*
- several should fall at post-load or deduplication:
  - Breast Cancer Wisconsin (Diagnostic) is wdbc, so the DEV signature;
  - Glass has a minority class of 9;
  - Haberman, Image Segmentation and its Statlog copy, SPECT, Sonar and MONK's are likely PMLB content;
  - Absenteeism has C > 10;
- about 10–14 should remain;
- they should be mostly binary, plus a few with C = 3–5.

**T1′.**
- Tier 2a (26) + Tier 3 (30) + Tier 4 (≈ 12) ≈ 68 datasets before A2.6 and A2.7. That is below the target of 72, so every eligible dataset enters T1′, and the stratum counts are those of the union.
- The union is dominated by C2 × IR<2, at about 40% of T1′ against 11% in Tier 1's design.
- The balanced core (≤ 8 per stratum) is the pre-specified check of that imbalance.

**Expected shortfalls against Tier 1's quotas of 8.** C5–10 × IR 2–5 (about 2), C5–10 × IR>5 (about 3), and C3–4 × IR>5, C3–4 × IR<2 and C5–10 × IR<2 (about 4–5 each).

#### A2.9 Compute (no run is part of Study 8)

From the Sec. 9 and A1.4 unit costs:
- **Study 9 on T1′** (4 FMs, about 4.4 worker-hours per dataset): about 68 × 4.4 ≈ 300 worker-hours.
- **The Study 7 extension on the full Study 4 grid** (≈ 3.8 worker-hours per 1000 rows): for Σn ≈ 85 000, about 320 worker-hours.

The freeze note records the actual Σn.

#### A2.10 Outputs and order of work

1. Commit this amendment (done before any Tier 4 download).
2. `simaudit build-pool-uci` writes `data/pool_ext_uci.yaml` and `data/pool_ext_uci_audit.csv`. Each file records:
   - the snapshot hash;
   - per dataset: UCI ID, DOI, licence, last-updated date, file SHA-256, content hash, n, d, C, IR and stratum;
   - `heldout_signature_check: pending`.
3. `simaudit screen-integrity` writes `data/pool_ext_integrity_audit.csv` for Tiers 2a, 3 and 4.
4. `simaudit build-t1prime` writes `data/pool_t1prime.yaml`.
5. All are committed with a freeze note below, before any model runs on them.

**Unchanged.** Tier 1's rules (Sec. 2–9); Amendment 1, except A1.5 as stated in A2.5; and the frozen PMLB pool file. A2.3, A2.6 and A2.7 act through new files and through exclusions recorded in them; the PMLB pool file is not rewritten.

#### A2.3a Addendum (2026-10-21, after the Tier 4 download, before any screen or model run): row-subset duplicates

**What prompted it.** The Tier 4 member list (names, n and class sizes only) contained two row subsets of frozen PMLB datasets. The hash, X-only hash and signature rules of A2.3 cannot detect subsets:
- UCI 50 Image Segmentation (n = 210), the training part of the 2310 rows of PMLB `segmentation`;
- UCI 70 MONK's Problems (n = 432), the test part of PMLB `monk1` (n = 556).

No integrity screen and no model had been run on any expansion dataset when this was written.

**Rule.** Two datasets with the same d are duplicates if at least 50% of the unique feature rows of either one occur exactly in the other. The check is the larger of the two directions (`integrity.row_overlap`, `OVERLAP = 0.5`).
- It applies to every T1′ candidate (Tiers 2a, 3 and 4) against the DEV datasets (cached) and against each other.
- **Priority** is Tier 2a > Tier 3 > Tier 4, then the key in alphabetical order. A candidate is compared only with DEV and with the higher-priority candidates already kept.
- An excluded candidate is recorded in `data/pool_ext_integrity_audit.csv` as `exclude: row overlap … with …`, and the T1′ draw skips it.
- Against HELD-OUT, the check needs the HELD-OUT features. It runs when HELD-OUT is first opened (A2.6 (b)), with the same consequence as A2.6.
- When Tier 1 is built, it applies to Tier 1 against these tiers, with A2.3's priority.

#### A2.3b Addendum (2026-10-21, after a first screen run, before any model run): how row overlap is measured

**What happened.** `screen-integrity` and `build-t1prime` were run once under A2.3a. No model was run. Their output was not committed, and is replaced by the re-run below. Three defects showed up:
1. **The code dropped the exclusions.** The `build-t1prime` command did not pass the integrity CSV, so T1′ ignored every exclusion.
2. **Exact matching missed the two cases that motivated A2.3a.**
   - UCI 50 against PMLB `segmentation`: 0.17 by exact float match, because the two sources print the floats differently. It is 0.99 after rounding.
   - UCI MONK's against PMLB `monk1`: the same grid with a different integer coding, 0.06 by exact match.
3. **The two-directional check over-excluded.** A large discrete grid that merely contains a small one was excluded: for example `car_evaluation` and `monk1` because they contain the 64 unique rows of `corral`.

**Rule, replacing A2.3a's measure.** Everything else in A2.3a is unchanged: the priority, the comparison only with DEV and with higher-priority candidates already kept, the 0.5 threshold, the HELD-OUT timing and the Tier 1 rule.
- The overlap is **the candidate's** share of unique rows found in the kept dataset. It is one-directional: the candidate's rows inside the other.
- That share is computed twice, and the larger value counts (`integrity.overlap_share`):
  - (i) after rounding every value to 6 significant digits;
  - (ii) after replacing each column by the dense rank of its values, which makes the check invariant to integer recodings of the same grid.
- *Known consequence:* datasets drawn on the same small Boolean grid with different target functions (e.g. the `parity`, `mux`, `threeOf9` and `xd6` families, or the thyroid family `allbp`/`allhypo`/`allrep`/`dis`) count as duplicates. That is the A2.3 principle "the same rows under another target".

### Freeze note, Amendment 2 (2026-10-21)

**Order of work.**
1. Amendment 2 was committed (R040).
2. `simaudit build-pool-uci` was run once.
3. A2.3a was committed with the code (R041).
4. A first screen run was made. It was not committed (A2.3b).
5. A2.3b was committed (R042).
6. `screen-integrity` and `build-t1prime` were run once.

No model has been run on any expansion dataset.

**Tier 4** (`data/pool_ext_uci.yaml`, `data/pool_ext_uci_audit.csv`): **14** datasets, Σn = 18 131. Every one is CC BY 4.0.

*Post-load exclusions:*
- Breast Cancer Wisconsin (Diagnostic): the DEV signature, i.e. wdbc.
- Glass: minority class of 9.
- Absenteeism: C = 19.
- Haberman and SPECT Heart: PMLB data hash.
- Statlog (Image Segmentation) and Sonar: PMLB signature.
- Parkinsons and Diabetic Retinopathy Debrecen: rejected by the loader. Their API variable lists repeat a column name (e.g. two `MDVP:Jitter`), so the Feature columns cannot be matched to the file one-to-one. These are recorded rejections, not re-drawn (Sec. 5 precedent).

**Integrity screens** (`data/pool_ext_integrity_audit.csv`, 70 datasets: Tiers 2a, 3 and 4):
- **S1.** 30 of 70 are sorted by class (κ ≥ 0.5). They are flagged only.
- **S2.** No leaky ID or counter column was found; no column is dropped.
- **S3.** 9 datasets are excluded for conflicting duplicates (share > 5%): spect (2a), tae (2a), analcatdata_germangss, contraceptive_method, calendarDOW, flags and analcatdata_dmft (all Tier 3), and UCI 863 Maternal Health Risk and UCI 379 Website Phishing.
- **Row overlap (A2.3a/b).** 9 datasets are excluded:
  - xd6, mux6 and parity5+5 (2a), against threeOf9, corral and mofn_3_7_10 (same Boolean grids);
  - dis, allhypo and allrep, against allbp (the same thyroid rows);
  - UCI 50 Image Segmentation, against PMLB segmentation (0.99);
  - monk1 and UCI 70 MONK's, against car_evaluation (0.75, by dense ranks).
  - *Known over-exclusion:* monk1 and car_evaluation are different tables whose small discrete grids coincide under dense ranking. Both MONK copies leave, and the rule is kept as committed.

**T1′** (`data/pool_t1prime.yaml`): **52 datasets, Σn = 63 418.** By source: Tier 2a 21 (Σn 18 770), Tier 3 21 (Σn 29 526), Tier 4 10 (Σn 15 122). The numeric-only subset has 31 datasets (Σn 33 892); the balanced core has 36 (Σn 52 393).

| Stratum | 2a | 3 | 4 | T1′ | Tier 1 quota |
|---|---|---|---|---|---|
| C2, IR<2 | 11 | 9 | 4 | 24 | 8 |
| C2, IR 2–5 | 2 | 3 | 1 | 6 | 8 |
| C2, IR>5 | 1 | 3 | 1 | 5 | 8 |
| C3–4, IR<2 | 1 | 0 | 0 | **1** | 8 |
| C3–4, IR 2–5 | 2 | 2 | 2 | 6 | 8 |
| C3–4, IR>5 | 0 | 2 | 0 | **2** | 8 |
| C5–10, IR<2 | 2 | 1 | 1 | 4 | 8 |
| C5–10, IR 2–5 | 0 | 0 | 0 | **0** | 8 |
| C5–10, IR>5 | 2 | 1 | 1 | 4 | 8 |
| **Total** | 21 | 21 | 10 | **52** | 72 |

**Shortfall.**
- T1′ is 20 short of 72.
- Eight of the nine strata are below quota. Only C2 × IR<2 exceeds it: 24, or 46% of T1′, against 11% in Tier 1's design.
- The multiclass strata are thin: C3–4 × IR<2 has 1, C3–4 × IR>5 has 2, C5–10 × IR 2–5 has 0.
- The balanced core and the post-stratified analysis (A2.5) are the pre-specified checks.

**Still pending.**
- The HELD-OUT signature check, and the HELD-OUT row-overlap check (A2.6, A2.3a).
- The trigger note at T* (A2.5).
- Tier 1, when OpenML returns.

**Compute (A2.9, with the actual Σn).**
- Study 9 on T1′ with 4 FMs: about 52 × 4.4 ≈ 230 worker-hours.
- The Study 7 extension on the full Study 4 grid: about 3.8 × 63.4 ≈ 240 worker-hours.
- Each is about 60 wall-hours at 4 workers.

### Trigger note at T* (A2.5), 2026-10-21
- **T\*.** The first of the two events of A2.5 is the Study 9 Sec. 10 freeze note: commit R043, 2026-10-21 20:26:15 −0300. By then no Study 7 DEV report had been committed (`reports/STUDY7_REPORT.md` is not in the repository; no Study 7 report commit exists).
- **Tier 1 at T\*.** `data/pool_ext.yaml` had not been committed, and Tier 1 has no freeze note. OpenML was unreachable during the whole 3-hour poll (Amendment 1 freeze note).
- **Outcome: the trigger fires.** T1′ (`data/pool_t1prime.yaml`, frozen in R003, 52 datasets) replaces Tier 1 in every confirmatory use:
  - Study 8 Sec. 7;
  - Study 9's EXP, which becomes EXP′ = T1′ with FM1, FM2, FM3 and FM5 on every member (Study 9 Amendment 1);
  - Study 10's primary set (HO ∪ T1′).
- The choice is irreversible (A2.5). This note changes no rule and was written before any Study 9 run on T1′. The HELD-OUT signature and row-overlap checks for T1′ (A2.6, A2.3a) are still pending, and no T1′ result is analysed before they have run.

### HELD-OUT check note for T1′ (A2.6, A2.3a/b), 2026-10-27
Written after the check and before any T1′ result was analysed. Study 9's runs on T1′ (52 datasets × 4 FMs) were finished; none of their outcomes had been read.
- **Event.** A2.6 (b), the first opening of HELD-OUT, limited to what the checks need. Script `tools/heldout_dedup_t1prime.py`, committed before it was run (R004). Output `data/t1prime_heldout_checks.yaml`; access log `data/t1prime_heldout_access_log.jsonl`.
- **Access.** `sealed_heldout/MANIFEST.json` was read, and all 32 npz files matched their manifest SHA-256 (file bytes hashed, not parsed). Then only the `X` array of each npz was read. No HELD-OUT label (`y`) and no other field was read, and no model was run.
- **Hash (A2.6).** No T1′ member's (X, y) content hash equals a HELD-OUT hash.
- **Row overlap (A2.3a/b).** The largest share of a member's unique rows found in a HELD-OUT dataset is 0.004 (pmlb:analcatdata_lawsuit), far below 0.5. No member is excluded.
- **Signature (A2.6).** Equal signatures need equal n. No T1′ member has the n of any HELD-OUT dataset, so the check passes without HELD-OUT class counts.
- **Outcome.** `heldout_signature_check` and the HELD-OUT row-overlap check: **passed**. T1′ stays at 52 datasets. The A2.7 integrity exclusions were already applied at the draw (18 candidates never entered), and no T1′ member has a dropped column. `pool_t1prime.yaml` is not rewritten.
