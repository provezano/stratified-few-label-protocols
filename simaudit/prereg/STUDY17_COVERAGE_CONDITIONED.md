# Study 17: coverage-conditioned and coverage-aware contrasts beyond one label per class (paper P1)

Status: FROZEN — 2026-11-01. Drafted 2026-11-01; expanded and frozen the same day (decisions O1–O4, Sec. 10.1). No Study 17 fit exists and no FM was run for this draft. The runner refuses real runs until this file is copied to `prereg/STUDY17_COVERAGE_CONDITIONED.md` with `Status: FROZEN` and committed unchanged (`tools/study17/study17.py::check_frozen`). Changes after the freeze only by dated amendments at the end, each committed before the run or analysis it affects.

Code (committed with this draft; tested on synthetic data under pytest `tmp_path` only):
- `tools/study17/study17.py`: draws, common-random-number seeds, guard, resumable job loop. Reuses `runner/study13.py` unchanged (`row_permutation`, `fixed_split`, `fit_record`, `_atomic_write`) and the FM wrappers through `runner/study4._predictor` (pins as Studies 13/15);
- `tools/study17/run_study17.py`: CLI (sharding, workers, `--dummy` smoke mode);
- `tools/study17/analysis.py`: every test and decision of Secs. 3–6;
- `tools/study17/build_frame.py`, `frame_check.py`, `postrelease_specs.jsonl`: frame build and checks;
- `tools/study17/power.py`, `compute_estimate.py`: Sec. 7 and Sec. 8 numbers;
- `tools/study17/test_study17.py`: 12 tests.
SHA-256 values are in Sec. 11.

## 0. Why, and what has been seen

### 0.1 Three anticipated reviewer attacks on P1 (TMLR #13940)
| Attack | What Study 17 tests |
|---|---|
| (1) "The effect is just class coverage." | **Family P:** stratified (S) against random draws *conditioned on covering all C classes* (RC), same budget mC, at m ∈ {3, 5, 10}. Any S − RC gap is a composition (balance) gap, not a coverage gap. **Family E** (TOST) can establish the opposite: that the gap is coverage only. |
| (2) "A fixed-budget random sample is an unrealistic comparator; your own §7 says: sample until every class is covered." | **Family Q:** S against the coverage-aware acquisition arm A ("random until every class is seen, then up to the budget"), with the labels A actually uses, the label overhead and the label-equivalence ratio ρ_A. |
| (5) "No confirmatory evidence at 3C–10C, where coverage is almost complete." | Every primary test averages m ∈ {3, 5, 10} only, on a new frame. m = 1 enters only family L1. |

It also removes the seed coupling flagged as reviewer point 7: in Studies 7–15 the FM seed was derived from the selection seed. Here every arm at (table, m, rep) uses the **same learner seed**, drawn independently of every selection (common random numbers, Sec. 2.3).

### 0.2 Seen before this draft (disclosed)
- **DEV, T1′ (PMLB/UCI), Study 15 second frame:** everything (P1 and its exploratory appendices).
- **HELD-OUT:** the Study 13 joint results. For Sec. 7, `tools/study17/power.py` read the stored arm-T outputs of `results/study13/heldout` (not `sealed_heldout/`, which was neither opened nor listed). Per-table HELD-OUT values were used only in memory, as resampling atoms. Only aggregates were written (`prereg/DRAFTS/study17_power/effects.csv`: n, mean, SD, d_z per set).
- **Exploratory S − redraw at m ≥ 3** (P1 Table "main", column "S−redraw, m≥3", from Study 12; the Study 13 and 15 redraw arms are the same construction). These numbers plan the power. They also set the direction expectations in Sec. 3.3. The pooled values over T1′ + HELD-OUT + second frame (n = 104) are: macro-F1 S − RC = +0.036 (fm1), −0.003 (fm2), +0.010 (fm3), +0.054 (fm5).
- **Study 17 frame:** only what the Study 16 candidate build recorded (metadata, class counts, feature kinds, content hashes, dedup), plus the re-encoding check of Sec. 1.4 (content hashes identical to the candidate file). No model was fitted and no feature–label statistic was computed.

## 1. Data

### 1.1 Source: the Study 16 candidate pool
The 54-table candidate pool `data/pool_study16_candidates.csv` was built on 2026-10-31 for the unfrozen Study 16 draft. It has three parts:
- Part O: OpenML tables outside every used pool, with a nominal feature or a missing value;
- Part U: UCI tables that the T1′ build left out only for nominal or missing features;
- Part P: Zenodo and figshare tables first public after 2026-07-01.

The candidate build already deduplicated this pool against DEV, HELD-OUT (ID list, names, signatures and hash list in memory; no seal opened), T1′, the PMLB/UCI pools, the Study 14 T2 metadata candidates and the Study 15 second frame. Study 17 checks the pool further (Sec. 1.2). The relation to Study 16 is decision O1 (Sec. 10).

### 1.2 Additional exclusions (`tools/study17/frame_check.py` → `prereg/DRAFTS/study17_frame_check.csv`)
Each candidate was checked by ID, record, normalised name and content hash against the frames and pools of the project's other studies (DEV, T1′, HELD-OUT and the second frame among them).

The checks use metadata only. **Hard clashes** are a matching ID, record, normalised name or content hash. Every hard clash is excluded: 16 tables.
- Pools of other studies of the project (by ID and name): OpenML 25, 185, 1412, 6332, 23381, 40966, 42585, 46601, 46630, 46758, 46927, 46940.
- Name only: OpenML 23380 (cjs) and UCI 426, 419, 277.

**Token clashes** come from the exclusion rule, which matches any shared token of 4 or more characters. That rule is over-inclusive, so 45 tables were reviewed by hand:
- One is the same data and is excluded: OpenML 45557 "Mammographic-Mass-Data-Set" is UCI "Mammographic Mass".
- All others are generic words ("survey", "factors", "clinical", "with", "churn" for different churn tables, "chess" for jungle-chess vs kr-vs-kp, "breast" for breast-w vs Breast Cancer Coimbra, "pain" for low-back pain in pregnancy vs the Japanese pain survey). They are kept.

**First part of the frame: 37 tables** from the Study 16 candidates.

### 1.2b Expansion, Part P2 (decision O2, 2026-11-01)
- **Goal.** About 33 more tables, for fm3 power near 0.8. Same eligibility, exclusion, contamination and hash rules. Multiclass and post-release tables first.
- **Search.** New Zenodo (1,596 records) and figshare (928 records) harvests of datasets first public in 2026-07 to 2026-10, CC BY / CC0 only, csv/xlsx only, files < 25 MB. Downloads go only to the new cache `data/cache/study17/raw/<key>/`, each file checked against the repository md5. All-numeric complete OpenML tables were not used: they belong to the DEV/HELD-OUT/T2 frame rule.
- **Specs.** Each table has a reviewed spec (`tools/study17/postrelease_specs_v2.jsonl`): target, columns dropped as identifiers or as target-defining (post-outcome) columns, and a reason from the record or its article. Records without an outcome column, with anonymised variable names, aggregate-only, n > 5,000, n < 100 or > 100 encoded features were rejected before a spec was written.
- **Build.** `tools/study17/build_postrelease17.py` → `prereg/DRAFTS/study17_postrelease_v2.csv`. 29 specs were encoded with the Study 15 rule. Six fail eligibility (smallest class < 11): gbs, depreth, blv, tpexgb, menop, lipedema. Content hashes were checked against DEV, the HELD-OUT hash list (in memory, no seal opened), T1′, Study 15 and the first 37 Study 17 tables, and row overlap (≥ 0.5) against DEV and T1′. No hash match, maximum overlap 0.
- **Clash check.** `tools/study17/merge_expansion.py` → `prereg/DRAFTS/study17_expansion_check.csv`. The 23 eligible records were checked by record ID, normalised name and content hash against the protected sets of the other studies. No clash.
- **Result: 23 tables added (Part P2), 2 of them multiclass.** The honest search did not reach 33. The pre-set rule applies: the frame is frozen with what there is, and the power reached is recorded in Sec. 7.

**Frame: 60 tables** (`data/pool_study17.yaml`).

### 1.3 Composition
| | Tables | Binary | Multiclass | Imbalanced binary (π_min < 0.25) | Σn |
|---|---|---|---|---|---|
| Part O (OpenML) | 14 | 10 | 4 | 2 | 13,336 |
| Part U (UCI) | 1 | 1 | 0 | 0 | 943 |
| Part P (Zenodo/figshare, post 2026-07-01) | 22 | 15 | 7 | 8 | 10,709 |
| Part P2 (Zenodo/figshare expansion, post 2026-07-01) | 23 | 21 | 2 | 7 | 22,970 |
| **Total** | **60** | **47** | **13** | **17** | **47,958** |

- **C:** 2 (47 tables), 3 (6), 4 (4), 5 (1), 6 (1), 9 (1).
- **Imbalance spread:** r = π_min·C has quartiles 0.03 / 0.36 / 0.62 / 0.79 / 1.00, and π_min reaches 0.017.
- **Size:** n from 100 to 4,766; d from 4 to 97. All tables have nominal features and/or missing values, except 4 of Part P.
- **Multiclass share:** 0.22 (13 of 60). Multiclass post-release tables are rare in the harvests. The effect is larger on multiclass tables: pooled S − RC macro-F1 is +0.048 vs +0.032 (fm1) and +0.019 vs +0.006 (fm3).
- **Review flags:** 11 tables (the 9 Study 16 flags plus 2 case-control P2 tables, syphilis and gout). The flags are Likert targets, case-control designs, a preference target, n = 111, 15 positives, and undocumented columns dropped. They carry over to SENS-FLAG (Sec. 5).

### 1.4 Encoding, eligibility, build
- **Encoding.** The Study 15 rule (`second_frame.encode`), with the Study 16 typed-target protection for OpenML/UCI. Every encoded table reproduces the candidate file's content SHA-256 (the build stops otherwise).
- **Eligibility.** The Study 15 rule: 2 ≤ C ≤ 10, smallest class ≥ 11, n ≥ 100; plus n ≤ 5,000 and d ≤ 100 encoded features.
- **Build.** npz files go to the new cache `data/cache/study17/npz/` (not committed). OpenML tables were read from the existing project cache, UCI from the UCI cache, and Part P from `data/cache/study16_candidates/raw/` with the reviewed specs. Part P2 was downloaded into the new cache `data/cache/study17/raw/` (Sec. 1.2b).

### 1.5 Pretraining contamination (per table in `pool_study17.yaml`; Study 16 audit of 2026-10-30)
| FM | Checkpoint | Tables first public after it | Note |
|---|---|---|---|
| fm1 Real-TabPFN (TabPFN v2) | 2025-07-05 | 45 | Synthetic prior + 71 real tables, all > 10k rows. `realtabpfn_71 = no` for every frame table (n ≤ 4,704). |
| fm2 TabICLv2 | 2026-02-12 | 45 | Synthetic. |
| fm3 Kumo-Tabular | undocumented | — | No verifiable status. |
| fm5 TabPFN v3.5 | 2026-09-09 | **33** | Prior tuned on BeyondArena. |

The 33 tables first public after 2026-09-09 form the contamination-safe subset (S-5). An FM win on older tables is never read as evidence against the hypothesis without this subset (rule of 2026-10-30).

## 2. Design

### 2.1 Geometry (fixed test set, same queries for every arm)
- Per table, rows are permuted once (Study 13 namespace 35, `row_permutation`).
- A fixed class-stratified test set T_d is drawn first (Study 13 `fixed_split`, namespace 36): per class max(1, min(⌊0.3 n_c⌋, n_c − 10)) rows.
- Every labelled set is drawn from the pool P = rows outside T_d, and **every fit is scored on T_d**.

This is the Study 13 arm-F geometry, not arm T (complement scoring). The reasons:
- (a) A uses more labels than mC, and with complement scoring its query set would differ from the other arms;
- (b) identical queries make the paired arm differences less noisy.

This is deviation 1 (Sec. 10.2).

### 2.2 Arms (rep r, m labels per class, B = mC)
| Arm | Definition | m | Reps |
|---|---|---|---|
| **S** stratified | m rows of every class from P | 1, 3, 5, 10 | 10 |
| **R** random | first B rows of the rep's random stream (one permutation of P per rep, shared across m, so R(m) sets are nested) | 1, 3, 5, 10, 20 | 20 |
| **RC** coverage-conditioned random | R if R covers all C classes; else rejection sampling of independent uniform B-subsets of P until one covers (≤ 100,000 attempts; infeasible otherwise, recorded). This is exactly uniform over covered B-subsets and keeps the budget B. | 1, 3, 5, 10 | 20 |
| **A** coverage-aware acquisition | stream prefix of length max(B, T_cover), where T_cover is the first prefix length that covers every class: "random until every class is seen, then up to the budget". Labels used = max(B, T_cover). | 1, 3, 5, 10, 20 | 20 |

- m = 20 (R and A only) is a curve point for ρ_A. It is skipped when 20C > |P|.
- A at m equals R whenever B ≥ T_cover, and RC equals R whenever R covers. Identical (rows, learner seed) pairs are fitted once and reused, so the extra arms cost fits only where coverage fails.

### 2.3 Seeds (common random numbers)
- `learner_seed = SeedSequence([base, did, m, rep, 173])`. It is the same for S, R, RC and A at (table, m, rep), and independent of every selection seed:
  - stratified 170;
  - stream 171;
  - RC attempts 172.
- Base seed 20261101. Table IDs:
  - OpenML ID;
  - UCI −(10⁹ + ID);
  - Zenodo 10¹⁰ + record;
  - figshare 2·10¹⁰ + record.
- Pairing: S rep r < 10 shares its learner seed with rep r of R/RC/A. The analysis averages within arm before differencing (Sec. 3.1), so the pairing reduces variance and does not change the estimand.

### 2.4 Learners and metrics
- **FMs:** fm1 (TabPFN v2 / Real-TabPFN, tabpfn 9.2.0 default checkpoint), fm2 (TabICLv2), fm3 (Kumo-Tabular), fm5 (TabPFN v3.5), through `runner.study4._predictor` with the Study 13 pins. CPU, one thread. No classical learner and no fm6–fm8.
- **Seen-class assumption:** a fit sees only the classes in its context (`study9.predict_on`). With one seen class the prediction is constant and no fit is made.
- **Per fit:** the Study 13 panel (`study13.fit_record`): macro-F1 over all C classes, accuracy, balanced accuracy, seen-class metrics, AUC, `pc_*` after p/π̂, κ, MCC, CITL, coverage, probabilities. Also stored: labels used, T_cover, RC attempt and learner seed.

## 3. Hypotheses

### 3.1 Unit and contrasts
- **Unit.** The table. Per (table, FM, arm, m), take the mean over reps (infeasible draws are dropped and counted). Per table, Δ = mean over m ∈ M of (arm₁ − arm₂). M3 = {3, 5, 10}.
- **Primary metric.** Macro-F1 over all C classes.
- **Complete case per FM.** A table is lost for an FM if any needed m is missing.

### 3.2 Confirmatory families (each Holm over the four FMs at α = 0.05)
| Family | Contrast | Metric, budgets | Test | Addresses |
|---|---|---|---|---|
| **P (primary)** | S − RC | macro-F1, M3 | Wilcoxon signed-rank, two-sided | attacks (1), (5) |
| **E** | \|S − RC\| < 0.02 | macro-F1, M3 | TOST (two one-sided signed-rank tests on Δ ± 0.02) | attack (1), the opposite reading |
| **Q** | S − A | macro-F1, M3 | Wilcoxon, two-sided | attack (2) |
| **L1** | \|S − RC\| < 0.02 at m = 1 | macro-F1, m = 1 | TOST | composition remainder at 1C |

- **Key secondary families** (each Holm over 4; reported with verdicts, no claim depends on them alone):
  - **BA:** S − RC on balanced accuracy, M3. Prop. balance (i) predicts S ≥ RC for every composition.
  - **ACC:** S − RC on accuracy, M3. Prop. balance (ii) predicts RC ≥ S once B is large on skewed tables. The test is two-sided, and the prediction is RC > S.
  - **COV:** RC − R on macro-F1, M3, the coverage part.
  - **C1-m3:** S − R on macro-F1, M3, the unconditional contrast of Study 16 family A for these four FMs.
  - **L1-cov:** S − R on macro-F1 at m = 1, the pure coverage gap.
- **No cross-family adjustment.** The four confirmatory families answer different questions, and each statement in Sec. 4 rests on one family.

### 3.3 Expectations, written now (from Sec. 0.2 and theory)
- **P:** replicated for fm1 and fm5; not established for fm3 (small effect); not established or a tie for fm2. Direction S > RC for every FM.
- **E:** equivalence for fm2; not established for fm1, fm5.
- **Q:**
  - At m ≥ 3, A differs from R only when R misses a class, so S − A ≈ S − RC.
  - A uses slightly more labels than RC.
  - Expected: Q replicated for fm1 and fm5, with a mean label overhead of A small at m ≥ 3 and large at m = 1 on imbalanced tables.
- **L1:** equivalence for every FM. At B = C, a covered random draw has exactly one row per class, uniform within the class, which is the stratified distribution. With learner seeds independent of the draw, E[S − RC] = 0 exactly (Prop. onelabel (i), Lemma composition). L1 is therefore a check of the theorem, and of the absence of seed artefacts, with the coupling of Studies 7–15 removed. The whole m = 1 gap is then coverage (L1-cov).
- **BA:** replicated for all four FMs.
- **ACC:** RC ≥ S.

## 4. Decision rules and consequences for P1

### 4.1 Per FM
- **P verdict.**
  - **Replicated:** Holm p < 0.05 and mean Δ > 0.
  - **Reversed:** Holm p < 0.05 and mean Δ < 0.
  - **Not established:** otherwise.
  - **Not evaluable:** more than 10% of the 60 tables lost for the FM (fewer than 54 analysed); a failed job is retried once.
- **E verdict.** "Equivalent" if the Holm-adjusted TOST p < 0.05.
- **Reading per FM (the P verdict takes precedence over E):**

| P | E | Reading |
|---|---|---|
| replicated | any | the macro-F1 gap at 3C–10C is **not only coverage**: a balance (composition) gap of the size estimated remains among covered draws |
| reversed | any | covered random contexts beat stratified ones in macro-F1 at 3C–10C |
| not established | equivalent | at 3C–10C the gap is **coverage** (within ±0.02) |
| not established | not established | inconclusive for that FM (estimate and CI reported) |

- **Q verdict** follows the P rules, with the mean label overhead of A (labels used / mC) and the censored median of ρ_A per m (Sec. 5) reported next to it.
- **L1:** "equivalent" (expected) or "not established". If it is not established with mean |Δ| > 0.02, it is reported as an implementation or theory discrepancy to investigate before any P1 text uses Study 17.

### 4.2 Frame-level statements
- **C-RC (not only coverage) holds** iff P is replicated for ≥ 3 of 4 FMs and none is reversed.
- **C-COV (coverage explains it) holds** iff E is equivalent for ≥ 3 of 4 FMs and P is replicated for none.
- **C-ACQ (stratified beats coverage-aware acquisition) holds** iff Q is replicated for ≥ 3 of 4 FMs and none is reversed.
- **Otherwise, per-FM statements only.** This is the expected outcome (Sec. 3.3, power Sec. 7). Each FM gets its reading from the table above.

### 4.3 What each outcome means for P1 (rebuttal + revision)
| Outcome | Rebuttal (attacks 1/2/5) | Revision of P1 |
|---|---|---|
| P replicated for fm k | "(1)/(5): at 3–10 labels per class, among draws that cover every class and have the same budget, stratified contexts still give higher macro-F1 for fm k (pre-registered, new frame, Δ = x [CI])." | Results: new paragraph "Beyond coverage" with the per-FM table. Limitation "No confirmatory test beyond one label per class" is rewritten to name the FMs for which it is now confirmed. Prop. balance is cited as the mechanism. |
| E equivalent for fm k (P not established) | "(1): for fm k we agree, and we now show it: beyond one label per class the gap is coverage, within ±0.02 (pre-registered TOST)." | Claim (1) is scoped per FM: "for fm k the overstatement at ≥ 3C is a coverage effect; reporting coverage or using covered draws removes it". The abstract mentions the split if it affects ≥ 1 of the 4 FMs. |
| P inconclusive for fm k | "(5): for fm k the 3C–10C contrast is not resolved at n = 60; estimate x [CI]." | Stated as such. No hedge beyond the CI. |
| P reversed for fm k | — | Abstract-level sentence; Sec. 7 recommendation changes for fm k (covered random draws give the higher macro-F1). |
| Q replicated for fm k | "(2): the comparator your review proposes, and our §7 recommends, still scores below stratified for fm k, while using x% more labels." | Sec. 7 keeps the recommendation and adds the measured label overhead and ρ_A. The protocol table gains the A row. |
| Q not established for fm k | "(2): the coverage-aware random protocol closes the gap for fm k at a label cost of x%." | Sec. 7 recommends A as the realistic comparator for fm k and reports its label overhead. |
| BA replicated (expected all) | Supports (1): the balance effect predicted by Prop. balance (i) is present among covered draws. | Added to the "Beyond coverage" paragraph. |
| L1 equivalent | Answers point 7: with independent, common learner seeds the 1C contrast against covered draws is zero, as the lemma says, so the 1C gap is entirely coverage. | Replaces the "approximate for our seeded FMs" caveat in the assumptions table with the Study 17 result. |

- **No re-run, re-seed, re-encoding or added table follows any outcome.**
- **Wording.** P1 wording uses the verdict labels above and never upgrades "not established" to "no effect".

## 5. Secondary (pre-declared; descriptive unless stated)
- **Decomposition per m ∈ {1, 3, 5, 10}** and per metric: S − R = (S − RC) + (RC − R), means per FM.
- **Label overhead of A:** mean labels used / mC per FM and m. **ρ_A(m):**
  - Definition: the labels that A needs to reach S's mean macro-F1 at m, divided by mC.
  - Curve: the per-table A curve over (mean labels used, mean macro-F1) at m ∈ {1, 3, 5, 10, 20}, made monotone, interpolated linearly in log labels.
  - Censoring: censored high if S exceeds the curve, low if S is below it.
  - Summary: the censored median per FM and m, valid if < 50% is censored high.
- **Strata and subsets** (P contrast, mean, CI, unadjusted p):
  - S-1 multiclass (11);
  - S-2 binary (26);
  - S-3 imbalanced binary (10);
  - S-5 contamination-safe (post 2026-09-09, 33);
  - SENS-FLAG (without the 9 flagged tables). A family-P verdict that flips under SENS-FLAG is reported as fragile.
- **Leave-one-table-out** range of the P p-values; the robustness panel of `analysis.study12.summarize` (mean, median, win rate, bootstrap 95% CI).
- **Infeasible RC/A draws:** count per table.

## 6. Exploratory (no test enters any family)
- The P, Q and BA contrasts per m, and with the moderators r = π_min·C, binary and n.
- The coverage of R at each m, against Prop. coverage (i) and Cor. budget.
- `pc_*` metrics (prior-corrected) under each arm.

## 7. Power (`prereg/DRAFTS/study17_power/`; `tools/study17/power.py`, 2,000 simulations)

**Method.**
- Table-level S − RC gaps at M3 were computed for T1′ (52), HELD-OUT (32, aggregates only) and the second frame (20), pooled (n = 104).
- k tables were resampled jointly across the four FMs, keeping the FM correlation, and reweighted to the frame's multiclass share of 0.217 (13 of 60).
- Wilcoxon two-sided, Holm over 4, rejections only in the predicted direction. TOST margin 0.02, Holm over 4.
- Scenarios: "full" (pooled effects) and "half" (each FM's gaps shifted down by half its mean).

**Pooled effects (macro-F1, S − RC, M3):**

| | fm1 | fm2 | fm3 | fm5 |
|---|---|---|---|---|
| Mean (SD) | 0.036 (0.045) | −0.003 (0.034) | 0.010 (0.038) | 0.054 (0.047) |
| d_z | 0.81 | −0.08 | 0.25 | 1.14 |

Balanced accuracy: 0.049 / 0.021 / 0.033 / 0.061 (d_z 0.86 / 0.56 / 0.68 / 0.91). Accuracy: −0.051 / −0.038 / −0.046 / −0.049.

**Power of family P (macro-F1) and E** (rerun 2026-11-01 for the 60-table frame; `power_sim.csv`):

| k | P fm1 | P fm3 | P fm5 | E fm2 (TOST) | E fm3 | P fm1 (half) | P fm5 (half) | P fm3 (half) |
|---|---|---|---|---|---|---|---|---|
| 30 | 0.96 | 0.36 | 1.00 | 0.73 | 0.13 | 0.56 | 0.85 | 0.15 |
| 37 | 0.98 | 0.44 | 1.00 | 0.84 | 0.20 | 0.68 | 0.93 | 0.20 |
| 50 | 1.00 | 0.56 | 1.00 | 0.94 | 0.30 | 0.82 | 0.98 | 0.25 |
| **60 (frame)** | **1.00** | **0.65** | **1.00** | **0.98** | 0.37 | 0.89 | 0.99 | 0.30 |
| 70 | 1.00 | 0.72 | 1.00 | 0.99 | 0.42 | 0.95 | 1.00 | 0.35 |
| 80 | 1.00 | 0.79 | 1.00 | 1.00 | 0.46 | 0.97 | 1.00 | 0.40 |

- **Power reached (O2).**
  - fm1 and fm5 (P): ≥ 0.99, and ≥ 0.89 even at half effects.
  - fm2 (E, ±0.02): 0.98.
  - **fm3 (P): 0.65**, below the 0.8 target. The draft figure (0.73 at k = 60) assumed a multiclass share of 0.30; the frame's real share is 0.22, and the pooled fm3 effect is smaller on binary tables. About 80 tables at this share would give 0.79.
  - fm3 equivalence stays out of reach (0.37).
  - A "not established" for fm3 is not evidence of no effect; its CI is reported.
- **BA family:** power ≥ 0.98 for all four FMs at k = 60 (full effects).
- **C-RC (≥ 3 of 4)** has low power by construction, because the expected outcome is per-FM.
- **Q family:** no stored acquisition arm exists. At m ≥ 3, A = R whenever R covers (most draws), and otherwise A ⊇ a covering prefix, so the Q effects are expected at or slightly below the P effects; the power is planned as for P.

## 8. Host, run plan, compute

### 8.1 Compute (`tools/study17/compute_estimate.py`; `prereg/DRAFTS/study17_power/compute_estimate.csv`)
- **Exact fit counts.** The draws of every table were enumerated, counting distinct (rows, learner seed) pairs with ≥ 2 seen classes: **6,449 fits per FM** (≈ 174 per table).
- **Timing model** from all 22,400 stored Study 15 fits: seconds = a + b_q·|queries| + b_B·budget per FM, least squares; a floor of 0.3 s per fit.
- **Estimate:**

| FM | fm1 | fm2 | fm3 | fm5 | Total |
|---|---|---|---|---|---|
| Worker-hours | 16.6 | 8.1 | 35.1 | 19.0 | **78.8** (≈ 118 with a 1.5× margin) |

- **Fits per FM:** 10,102 for the 60 tables (≈ 168 per table).

- **Smoke test** (`--dummy`, nearest-centroid, 2 tables, 1 thread): about 3 s per table-job for the draws and bookkeeping alone.

### 8.2 Hosts and launch
- Two machines; each FM runs on a single host, so each FM's fits come from one machine and one library install.
- Each process has 1 thread at nice 15. Shards are disjoint over (FM, table) jobs; a restarted shard skips finished jobs and resumes partial ones from their checkpoint (every 25 fits).
- **No interim look.** The analysis (`tools/study17/analysis.py`) runs once, after every job has finished or definitively failed.
- (Machine names, paths and launch commands removed.)

## 9. Threats
- **Frame reuse.** None: Study 16 was absorbed into Study 17 (O1) and will not run on its own.
- **Arm-F geometry.** Fixed test set instead of complement scoring. P1's primary protocol is arm T, so Study 17's level is comparable to P1's arm-F secondary results, not to Table main. The contrast of interest (S vs RC/A) is unaffected.
- **Rejection sampling cost** on very skewed tables (hipfx: 15 positives) is bounded by 100,000 attempts per draw. Infeasibility is recorded, not imputed.
- **ρ_A** depends on the interpolation between grid points (as P1's ρ does) and is descriptive.
- **Contamination.** 27 tables predate fm5 and 15 predate fm1/fm2; Kumo is unknown (S-5).
- **Overlap with other studies.** The token review is manual. A re-encoded copy under a new name would not be caught by metadata checks; content hashes rule out exact copies of another study's frame and of the Study 15 frame.
- **One frame, n = 60.** fm3 underpowered (0.65, Sec. 7). The P2 tables are mostly binary clinical tables, so multiclass is under-represented (0.22).

## 10. Decisions and deviations

### 10.1 Decisions (2026-11-01)
| ID | Decision |
|---|---|
| O1 | Study 16 is **absorbed** into Study 17 and will not run on its own. Its 37 kept candidates are the first part of the frame. The 17 excluded candidates (Sec. 1.2) are recorded in `prereg/DRAFTS/STUDY16_DRAFT.md`. |
| O2 | Expand the frame toward about 70 tables, under the same rules. Result: 60 tables (Sec. 1.2b). fm3 power is 0.65, as recorded in Sec. 7. |
| O3 | Run on two machines. Each FM runs on a single host (Sec. 8.2). |
| O4 | TOST margin **±0.02**. |

### 10.2 Deviations from Studies 13/15 (declared)
1. Arm-F geometry (fixed test set) for every arm.
2. Learner seed independent of the selection and common across arms.
3. New arms RC (rejection over independent subsets, not Study 13's redraw seeds) and A.
4. Reps: S 10, stream arms 20.
5. m = 20 for R and A.
6. Four FMs, no classical learners.
7. The confirmatory families are P, E, Q and L1, each with Holm over 4.

## 11. Code and data hashes (SHA-256, at the freeze)
| File | SHA-256 |
|---|---|
| tools/study17/study17.py | 46936330dc23301939facafc431377e100e3c793c468f893da2fed984735c2be |
| tools/study17/run_study17.py | 4995ad36456f668c4fb0b18dd57b56ff6a2a2f21b6b2197c738d0aa732228386 |
| tools/study17/analysis.py | 39b4ed9125a2280a0d3413a845bc5a9785ded7f5726d9da1e940226c9247f7fd |
| tools/study17/build_frame.py | 7872cb4118e6e60b7b0401609084d8ba6ecc23afc4455a78299b61446df963f7 |
| tools/study17/frame_check.py | eeb1364026b446e3c498a72bd33aadb71ed3a01890e5cdba2b72c73aa805f339 |
| tools/study17/power.py | 21fa85c96d44542552ed85f60cdcd558b709f46e3f721d7f1c9889805f699135 |
| tools/study17/compute_estimate.py | 37fd09b6c976bf0ad0446dc66fd15dab047e73a0601b84bbe4b6759657373d73 |
| tools/study17/test_study17.py | 11839022fcf3793a7ccccb4a8d4305f6470b8b9845b793bd6bc32ca8419d5b50 |
| tools/study17/postrelease_specs.jsonl | 3cf8ee4ed83190949b709b4ee1f86cd619937660eba1c0275a87a3a8ad40d476 |
| tools/study17/build_postrelease17.py | 2b53ed504df7f6b23cf777e059d48d00f855955335fa1839bd2b6d70774961fb |
| tools/study17/merge_expansion.py | 0039f8a736a24d76131b40d99f0c773fd0f7ecdd7d799bf6fb7d9f8a814c2c47 |
| tools/study17/postrelease_specs_v2.jsonl | 882ae1a2d59369eab9a81a31d2115f8bd7fb81760df47f407cbac8d8d5e2c58b |
| data/pool_study17.yaml | 201b750a29c04684d750309edab19bdfc7d2a0a9a5cfe4f4c32b580741570975 |
| prereg/DRAFTS/study17_frame_check.csv | 4e30f127bb1a830fcddbc96d078c7c6b39da2edf81b5e7e07486162a741a45cd |
| prereg/DRAFTS/study17_postrelease_v2.csv | a23f36834d0a428f1e38d66e16844294c58de4089518824b2d45c733706636cc |
| prereg/DRAFTS/study17_expansion_check.csv | 12f2a284076e27345642974c34cfd00acbbdbbc63dd40d337e0f59f36b8e7cc1 |
| prereg/DRAFTS/study17_power/power_sim.csv | 177ed048517c033e9eff209ad724d8ad553cda1dc3e4786ea4d01ace091cd5cb |
| prereg/DRAFTS/study17_power/compute_estimate.csv | 346ab5cb93068ff65ca405d9fc9e581a4a1cd08e0f4e1598b8b08b8a01ab234f |

- The SHA-256 of this file is in `prereg/DRAFTS/STUDY17_COVERAGE_CONDITIONED.md.sha256`. At the freeze, the frozen copy's SHA-256 and the commit hash go into Sec. 12, and the runner records every code hash in `results/study17/provenance_*.json`.
- **Tests:** `OMP_NUM_THREADS=1 nice -n 15 .venv/bin/python -m pytest -q tools/study17/test_study17.py -p no:cacheprovider --no-cov` gives 12 passed. `--no-cov` is required: the project's default pytest options write coverage files into the repository root.

## 12. Log
- **2026-11-01:** draft. The frame was checked (16 hard clashes plus 1 manual exclusion out of 54) and built (37 npz files, every content hash equal to the candidate file). Power and compute were estimated from stored outputs only. No FM was run.
- **2026-11-01 (expansion and freeze):** decisions O1–O4. The frame was expanded with 23 Part P2 tables to 60 (Sec. 1.2b). Power and compute were rerun (Secs. 7, 8). Study 16 was absorbed. **Frozen on 2026-11-01.** No FM was run before the freeze.

## 13. Amendments

### Amendment 1 (2026-11-02, before any analysis)
- **What.** `tools/study17/analysis.py::claim` declared C-COV when E was equivalent for ≥ 3 of 4 FMs, without checking the Sec. 4.2 condition "P is replicated for none". The code now checks both, as Sec. 4.2 says. The text of Sec. 4.2 is unchanged and governs.
- **Test.** `test_claim_ccov_requires_no_replication` added; the test file gives 13 passed.
- **Status at the time of the amendment.** Fits are still running on both hosts. No result file has been read or analysed, and `analysis.py` has never been run on Study 17 outputs.
- **Hashes.** The freeze hash in `STUDY17_COVERAGE_CONDITIONED.md.sha256` is that of the frozen text before this section. `analysis.py` after the fix: sha256 prefix 4389a80c24beaf83.
