# Study 13: confirmatory replication of Study 7 Part (a) on HELD-OUT (paper P1)

Status: FROZEN 2026-10-28T01:37:58Z. Drafted 2026-10-27. No Study 13 run exists and `sealed_heldout/` has not been opened for Study 13. After the freeze, changes only by dated amendments at the end, each committed before the run or analysis it affects.

Code (implemented with this draft, tested on DEV and synthetic fixtures only): `src/simaudit/runner/study13.py` (runner), `src/simaudit/analysis/study13.py` (analysis), tests `tests/unit/test_study13.py`. Study 7, 9 and 12 modules are reused read-only; no existing module changes. Outputs: `results/study13/{set}/{key}_{learner}.parquet`, one file per (dataset, learner group), written atomically, checkpointed and skipped when present (resumable).

## 0. Why, what has been seen, and the conflict with Studies 9 and 10

### 0.1 Why
P1 claims that the stratified "m labels per class" protocol overstates macro-F1 and changes a ranking. The evidence is DEV (Study 7, n = 20, the search set of the whole project) and T1′ (Study 12, n = 52, running; partly not blind, Study 12 Sec. 0). The reviews (internal reviews R2-W1, R2-Q1, R3) ask for confirmation on untouched data. HELD-OUT is the only set that no study, search or exploratory analysis has used: 32 OpenML tables drawn with DEV by the pilot I2 rule (`prereg/PILOT_I2.md` Sec. 2; `data/pool_i2.yaml`, `heldout`).

The systematic audit (research_notes/systematic_review_2026-10/REPORT.md §4) shows that the audited few-shot papers report AUC (37%) and accuracy (29%) as their main metric far more often than macro-F1 (7%). The AUC and accuracy analyses below (S-2, S-3) are therefore pre-declared, not optional.

### 0.2 Seen before this draft (disclosed)
- **DEV:** everything (Study 7 report, exploratory E1–E14 in paper/p1/exploratory/out/). Every prediction below comes from DEV.
- **T1′:** the Study 9 T1′ report (Study 12 Sec. 0). No Study 12 outcome was read for this draft. Only the `seconds` timing column of the finished Study 12 files and the run log's completion lines were read, for the compute estimate (Sec. 8).
- **HELD-OUT:** nothing of its content. The only prior access to `sealed_heldout/` was Study 8 A2.6 (b) on 2026-10-27 (`tools/heldout_dedup_t1prime.py`, log `data/t1prime_heldout_access_log.jsonl`): manifest hashes, then the `X` arrays only; no label, no model. Its outcome (no T1′ overlap) carries no information about any Study 13 contrast. HELD-OUT n, C and class counts are unknown to the authors.

### 0.3 Conflict: two other studies plan to use HELD-OUT (decision needed before freezing)
- **Study 10** (`STUDY10_SELECTOR_CONFIRMATION.md`, DRAFT, not frozen) plans HO ∪ T1′ as its joint primary set, with HELD-OUT "opened **once** for Study 10" after its own freeze (Sec. 3.1). Its gates G4 (T7, DEV dry run), the δ/decision-table confirmation and the compute budget are still open.
- **Study 9** (`STUDY9_HOW_MANY_LABELS.md`, FROZEN) plans one HELD-OUT replication with all four FMs after its EXP (T1′) report (Sec. 9), with Holm across Workstreams that use HELD-OUT.
- **Study 11** does not use HELD-OUT (its Sec. 4: "Not used: HELD-OUT").
- **Rule in force:** research_notes/WORKSTREAMS.md, rule 4: a single confirmation on the sealed HELD-OUT, with Holm across the Workstreams that use it.

**The conflict.** If Study 13 opens HELD-OUT first, Study 10 would be frozen after HELD-OUT data had been restored and Study 13 outcomes on it had been seen. Study 10's arms do not depend on Study 13's question, but "opened once" and "sealed throughout the search" (Study 10 T6) would no longer hold literally.

**Resolution (decision of 2026-10-27, option A; recorded in research_notes/WORKSTREAMS.md, "Joint opening of HELD-OUT"): one shared opening event.**
1. The studies in the event are Study 9 Sec. 9 (FROZEN; its precondition, the committed EXP report `reports/STUDY9_T1PRIME_REPORT.md`, holds), Study 13 (this file, frozen before the opening), and Study 10 **if and only if it is frozen by 2026-10-31 23:59 UTC**.
2. **Deadline rule.** If Study 10 is not frozen by then, the opening goes ahead for Studies 9 and 13 only. Study 10 must then record in its own prereg that HELD-OUT was already opened and which results had been seen. The opening may happen earlier than the deadline if Study 10 freezes sooner.
3. A dated opening note (Sec. 11, mirrored in the other preregs' logs) lists the studies, their confirmatory families and the cross-study Holm family (Sec. 4.4). It is committed before the first runner touches `sealed_heldout/`.
4. Each runner verifies the manifest itself (Sec. 9) and appends to its own access log. Nobody reads `sealed_heldout/` outside the frozen runners. No study's outcome is analysed before every confirmatory cell of every study in the event is complete or has definitively failed.

## 1. Data

- **HO.** The 32 HELD-OUT tables of `data/pool_i2.yaml` (`heldout`): OpenML, all-numeric, no missing values, 100 ≤ n ≤ 1000, 2 ≤ C ≤ 10, smallest class ≥ 11 (pilot eligibility).
- **All 32 tables stay in the primary analysis** (decision of 2026-10-27), as in Studies 7 and 12: no table is excluded and no column is dropped.
- **Integrity screen, flag-only, before any fit** (Study 8 A2.7 rules and thresholds, `src/simaudit/data/integrity.py`, unchanged). At opening, before any selection or fit, the runner screens every HO table in stored row order and writes `results/study13/heldout/integrity_screen.csv` with the A2.7 decision and two flags:
  - `flag_id`: an identifier/counter column, i.e. a leaky ID-like column (1-NN balanced accuracy ≥ 0.90 on the column alone, the A2.7 S2 rule) or an ID-like column monotone in the stored row index (|Spearman ρ| ≥ 0.99; the DEV 694 and T1′ cloud pattern);
  - `flagged`: `flag_id`, or any conflicting-duplicate group, or any A2.7 decision other than `keep` (exclusion for > 5% conflicting duplicates, or a column drop).
  - Class-sorted storage (A2.7 S1) is reported only; the row permutation (Sec. 2.1) neutralises it.
  - The flags are mechanical and are written before any outcome exists.
- **Pre-declared integrity sensitivities** (no confirmatory weight; Holm recomputed within the reduced family; every confirmatory contrast and S-1 to S-9):
  - **SENS-INT:** drop every flagged table.
  - **SENS-ID:** drop only the identifier/counter-column tables (`flag_id`).
  - A confirmatory verdict that changes under SENS-INT is reported as fragile.
  - No column-dropped re-run is made (unlike Study 7 S4); SENS-ID covers the identifier channel by removing those tables.
- **Expected flags (from DEV only; HO is drawn like DEV).** The A2.7 screen run on the 20 DEV tables (smoke test of the Study 13 runner, 2026-10-27, no model) would exclude 5 tables under A2.7 (860, 1063, 40496, 46336 for conflicting duplicates; 782, one column left after its leaky column) and drop a column of 694; 801 and 1071 have conflicting groups below 5%. So SENS-INT may remove about a third of HO (n ≈ 20–24 left), SENS-ID a few tables.
- **Failures.** A job that fails is retried once; a table that still fails for an FM is dropped from the contrasts that need that FM (complete case per contrast), and the count is reported. If more than 10% of the 32 tables is lost for a confirmatory contrast, that contrast is "not evaluable", not "not replicated".
- **Feasibility.** Every class has ≥ 11 rows, so stratified sampling (`data.splits.draw_split`, which needs ≥ m + 1 rows per class) is feasible at every m ∈ {1, 3, 5, 10} in the transductive arm. The fixed split (Sec. 2.4) is defined so that it is feasible too.

## 2. Design

### 2.1 Row permutation and seeds (new namespaces, disjoint from every earlier study)
- **Permutation.** π_d = `default_rng([base_seed, seed_did(did), 35]).permutation(n)` (namespace **35**), applied to X and y before anything else. Stored indices are mapped back to the stored order.
- **Selection and fit seed.** `SeedSequence([base_seed, seed_did(did), m, rep, study4.ORDER.index(strategy), 34, arm])` (namespace **34**; arm = 0 transductive, 2 fixed split). The same seed drives the selection and every learner of the cell.
- **Redraw attempt a ≥ 1.** `SeedSequence([base_seed, seed_did(did), m, rep, a, study4.ORDER.index("random"), 34, 1])`.
- **Fixed split.** `default_rng([base_seed, seed_did(did), 36])` (namespace **36**).
- base_seed = 20261020 (settings). Namespaces 34–36 are unused by Studies 1–12 (checked against every `*_NAMESPACE` constant in src/ and autoresearch/).

### 2.2 Arm T: Study 7 Part (a), transductive (the confirmatory arm)
- **Strategies.** `stratified` (m per class, 5 reps) and `random` (B = m·C uniform, 20 reps), through `runner.study4.select`, unchanged.
- **Budgets.** m ∈ {1, 3, 5, 10}, B = m·C.
- **Evaluation.** The complement of the labeled rows (HO has n ≤ 1000, so Study 12's 2000-row panel is the full complement).
- **Redraw (secondary arm, Study 12 Sec. 2.3).** Rep r's random draw is reused when it covers every class; otherwise attempts a = 1, 2, … redraw uniformly until every class is present, at most 100,000 attempts. **New:** a cell with no covering draw within the cap is recorded as infeasible (no fit, NaN metrics) and reported; it does not fail the job (Study 12 lost a job this way).

### 2.3 Learners
- **FMs (confirmatory):** FM1 TabPFN v2, FM2 TabICL, FM3 Kumo-Tabular, FM5 TabPFN v3.5 (research-only licence; `TABPFN_TOKEN`), via `runner.study4._predictor`, CPU, one thread. Missing classes: fit on the seen classes, probability 0 for unseen ones; one seen class → constant prediction (`runner.study9.predict_on`).
- **fm1_bal, fm5_bal:** the post-hoc p / π̂ transform of the stored probabilities (`analysis.study7.prior_correct`), as Study 12 Sec. 2.2.
- **Classical learners (secondary S-1, E1 specification, `paper/p1/exploratory/e1_classical.py`):** lr = LogisticRegression(C = 1, lbfgs, max_iter 5000); knn1; knn3 with k = min(3, B); gnb = GaussianNB(var_smoothing = 0.1); lda = Gaussian plug-in with Ledoit–Wolf pooled covariance and context prior; hgb = HistGradientBoosting(max_iter 100, max_depth 3, min_samples_leaf 1, lr 0.1, no early stopping). Features z-scored with the mean and SD of all rows of the table (label-free); constant columns get SD 1. Same labeled sets and seeds as the FMs; same missing-class rule. Re-implemented in the runner with identical definitions (the exploratory script is not a package module).

### 2.4 Arm F: a fixed evaluation split per table (secondary S-4; new fits; R3-W4)
- **Definition, fixed before any draw.** Per table and class c with n_c rows, the test set T_d holds t_c = max(1, min(⌊0.3·n_c⌋, n_c − 10)) rows of class c, drawn uniformly within the class with the namespace-36 generator, in permuted order. The labeling pool is P_d = rows ∖ T_d.
  - Class-stratified on purpose: every class is present in T_d with roughly its pool share, which removes the rare-class support problem that the strict intersection of E7 ran into. T_d uses labels only to define the evaluation rows, identically for every protocol and draw; no acquisition step sees them.
  - t_c ≤ n_c − 10 keeps ≥ 10 rows of every class in P_d, so stratified m = 10 is feasible.
- **Draws from P_d.** `stratified`: m rows per class of P_d, uniform within class (5 reps). `random`: B rows uniform from P_d (20 reps). Seeds with arm = 2. No redraw in arm F.
- **Scoring.** Every fit predicts T_d; every metric is computed on T_d. So all draws and both protocols share one evaluation set (an inductive, protocol-independent test set).
- **Learners.** All four FMs and the six classical learners.

### 2.5 Stored per fit, and metrics
- Stored: arm, strategy, m, rep, attempt, seed, labeled rows and query rows (stored order), float32 probability matrix over all C classes, coverage, seen classes, seconds, and the metrics.
- Metrics (Study 12 definitions, `analysis.study12.panel_metrics`): macro-F1 over all C classes, zero_division 0 (**primary**); accuracy; balanced accuracy; log-loss; AUC over seen-class columns (Study 7 convention); `auc_all` (all classes present, unseen class column = 0 ⇒ AUC 0.5; one seen class ⇒ 0.5); `seen_*` variants; and `pc_*` (after p / π̂).

### 2.6 Deviations from Study 7 / Study 12, all declared here
1. New seed namespaces (34–36).
2. Label-free selectors, Parts (b) and (c) are not run (as Study 12).
3. Redraw infeasibility is recorded, not a job failure.
4. Arm F and the classical learners are added (secondary only).
5. No panel cap is active (n ≤ 1000).
6. The integrity screen runs on HO for the first time; it only flags tables for the sensitivities.

## 3. Primary hypotheses (confirmatory family)

**Unit.** The table: mean over reps per (table, learner, strategy, m), then the mean over the m values stated. Arm T. Macro-F1.

| ID | Contrast | Per-table value | Test | Prediction |
|---|---|---|---|---|
| (i)-fm1, fm2, fm3, fm5 | stratified − random, per FM | mean over m ∈ {1, 3, 5, 10} | Wilcoxon signed-rank, two-sided | Δ > 0 for every FM |
| (ii-a) | fm2 − fm5 under random | mean over m ∈ {3, 5, 10} | Wilcoxon, two-sided | Δ > 0 (TabICL ahead) |
| (ii-c) | fm2 − fm5 under stratified | mean over m ∈ {3, 5, 10} | TOST, margin 0.01 (paired t); its p enters Holm | equivalent within ±0.01 |

- **One Holm family of 6** at α = 0.05, exactly as Study 12 Sec. 3.1. Margin 0.01 for TOST and ROPE 0.01 for the Bayesian signed-rank (baycomp), as in Studies 7 and 12.
- **Cross-study Holm (Sec. 4.4)** is applied on top when HELD-OUT is opened jointly.
- **Robustness for every contrast (Study 7 Sec. 4.2 / Study 12 Sec. 3.2):** mean, median, win rate, bootstrap 95% CI (`analysis.report.bootstrap_ci`); TOST; Bayesian signed-rank P(+), P(ROPE), P(−); leave-one-table-out (range of the mean, largest p, sign stability); per-budget versions (unadjusted); the same contrasts in accuracy, balanced accuracy, log-loss, AUC (seen) and the seen-class metrics (secondary, unadjusted).

## 4. Decision rules and replication criteria

### 4.1 Per contrast (identical to Study 12 Sec. 3.1)
- **(i), per FM.** *Replicated:* Holm p < 0.05 and mean Δ > 0. *Reversed:* Holm p < 0.05 and mean Δ < 0. *Not replicated:* otherwise.
- **(ii-a).** Same three outcomes.
- **(ii-c).** *Tie established:* Holm-adjusted TOST p < 0.05. *Difference:* not equivalent and two-sided Wilcoxon p < 0.05 (unadjusted, with its sign). *Unresolved:* otherwise.

### 4.2 Claims
- **C1 (optimism) replicates on HELD-OUT** iff (i) is replicated for ≥ 3 of the 4 FMs **and** no (i) contrast is reversed.
- **Ranking claim.** *Supported:* (ii-a) replicated and (ii-c) tie established. *Half supported:* (ii-a) replicated, (ii-c) not established. *Not supported:* (ii-a) not replicated.
- **Magnitude consistency (descriptive).** For each contrast: is the DEV estimate (Study 7) inside the HO bootstrap 95% CI; the ratio HO mean / DEV mean; the same against T1′ when the Study 12 report exists. A replication with a much smaller effect (ratio < 0.5) is reported as "replicated, attenuated".

### 4.3 What counts as non-replication, and its consequence for P1
- **Non-replication of C1:** fewer than 3 of 4 FMs replicated, or any (i) reversed. P1 then states in the abstract and conclusion that the macro-F1 overstatement is shown on DEV (and T1′ if Study 12 replicates) but **not** on HELD-OUT, with the HO estimates and CIs.
- **Non-replication of the ranking claim:** (ii-a) not replicated. The reversal is then reported as DEV-specific (or DEV/T1′), and P1 drops "changes rankings" from the contributions for this pair; the flip-rate (S-9) is still reported.
- **Disagreement between T1′ and HO** is reported per set; no pooling is used for a decision. A random-effects meta-analysis across DEV, T1′ and HO is reported descriptively (S-6).
- No re-run, re-seed or re-tuning follows any outcome.

### 4.4 Multiplicity across studies sharing the opening (WORKSTREAMS rule 4; decision of 2026-10-27)
- The within-study Holm (6 tests) gives each per-contrast verdict (Sec. 4.1).
- **Cross-study Holm.** The confirmatory p-values of every study in the joint opening (Sec. 0.3) form one Holm family at α = 0.05: Study 13's 6; Study 9's HELD-OUT family as its Sec. 9 defines it; and Study 10's H1 (4 NI tests) if Study 10 joins before the deadline. The opening note (Sec. 11) lists the exact members before the opening.
- **Wording rule.** A claim is "confirmed on HELD-OUT" only if its verdict holds under the cross-study Holm as well. If it holds only under the within-study Holm, it is "replicated (study-wise), not under the cross-study correction". Both adjusted p-values are reported (`analysis.study13.cross_study_holm`).
- Expected impact: (i) and (ii-a) had d ≈ 1.0–1.9 on DEV and keep power ≥ 0.99 at α = 0.05/19. (ii-c) is the last Holm step either way.

## 5. Pre-declared secondary analyses (no confirmatory weight; Holm within each block where a test is stated)

**S-1. Classical learners (E1, R3-W8).** Arm T, the same labeled sets. Per learner ∈ {lr, knn1, knn3, gnb, lda, hgb}: stratified − random, mean over m, macro-F1; Wilcoxon two-sided, Holm over 6.
- *Prediction (DEV E1):* Δ > 0 for all six. Learner-agnostic optimism is claimed only if ≥ 5 of 6 are Holm-significant with Δ > 0.
- Also: accuracy and balanced accuracy; per-budget values; the p / π̂-corrected versions (gnb, lda, lr, hgb). knn3 at m = 1 is degenerate (k = C vote) and is reported but flagged.

**S-2. AUC over all classes (E2; `auc_all`).** Arm T. Stratified − random per FM at m = 1 (Wilcoxon two-sided, Holm over 4) and as the mean over m (reported).
- *Prediction:* Δ > 0 at m = 1 for every FM. The seen-class AUC (Study 7 convention) and its NaN share are reported beside it.

**S-3. Accuracy gap versus pool imbalance (E6, R3-W2).** Arm T. Per table, ΔA = stratified − random accuracy (mean over m) and TV = ½‖π − u‖₁ of the full table's class shares.
- Primary statistic of the block: Spearman ρ(ΔA, TV) for the mean over the 4 FMs, one-sided (ρ < 0). Per FM: ρ and OLS slope with a pairs-bootstrap CI (reported, unadjusted). Also ρ with π_min.
- Binary tables: agreement of the sign of ΔA with the Prop. 2(ii) threshold B* = 1/(π₁π₂τ*²) (as E6; τ* from the stored probabilities as in e6_accuracy_heterogeneity.py), over cells with |ΔA| > 0.01, descriptive.
- *Prediction:* ρ < 0. The mean ΔA is not predicted to differ from 0; heterogeneity is the claim.

**S-4. Fixed evaluation split (arm F; R3-W4, R2-W4).** On T_d:
- (i) per FM, mean over m: Wilcoxon two-sided, Holm over 4; (ii-a) Wilcoxon and (ii-c) TOST, reported unadjusted.
- The paired difference of the (i) gap between arms T and F per FM (Wilcoxon, descriptive), and the T_d class support per table.
- The six classical learners on arm F (reported).
- *Prediction:* the same signs as arm T; the arm T − arm F difference within ±0.01 (TOST reported, no direction claimed).

**S-5. Binary versus multiclass (R3-W13).** Every confirmatory contrast and the (i) contrasts at m = 1, within binary (C = 2) and multiclass (C ≥ 3) tables: mean, CI, Wilcoxon p (unadjusted); Mann–Whitney U on the per-table Δ between groups. *Prediction:* the (i) gap at m = 1 is larger for multiclass tables. No direction for (ii). If a group has fewer than 6 tables, only the descriptive summaries are reported.

**S-6. Random-effects meta-analysis and prediction interval (E9, R2-W5).** Per FM and m (and the mean over m), macro-F1 and accuracy: REML pooled mean μ with 95% CI, τ, I² and the 95% prediction interval (t, k − 2 df). The within-table variance is var(stratified reps)/n_S + var(random reps)/n_R.
- *Expectation from DEV (descriptive, not a test):* for macro-F1, mean over m, the PI lies above 0 for TabPFN v2, Kumo and TabPFN v3.5 and straddles 0 for TabICL; for accuracy, every PI straddles 0.
- Also: the same model over DEV + T1′ + HO with the set as a moderator (descriptive).

**S-7. FM gap minus classical gap (E13).** D_d = Δ_FM − mean of the classical Δ over {lr, knn1, gnb, lda, hgb} (knn3 excluded as in E13), mean over m, macro-F1; Wilcoxon two-sided, Holm over 4 FMs.
- *Predictions (DEV E13):* D > 0 for TabPFN v3.5; D < 0 for TabICL. No direction for TabPFN v2 and Kumo.

**S-8. Redraw protocol (Study 12 S-1).** Stratified − redraw per FM (mean over m ∈ {3, 5, 10}; Holm over 4; *prediction:* Δ > 0 for TabPFN v2 and v3.5, none for TabICL and Kumo); fm2 − fm5 under redraw (*prediction:* Δ > 0); the covered share of random draws per m; the share of the (i) gap that redraw closes; infeasible redraw cells.

**S-9. Ranking flip rate (Study 12 S-3).** Learners {fm1, fm2, fm3, fm5, fm1_bal, fm5_bal}; decisive pairs (|Δ| > 0.01 under both protocols) per (table, m); flip rate with a table-cluster bootstrap CI against the random-halves noise baseline. *Prediction:* protocol flip rate > baseline (CI of the difference excludes 0).

Anything else is labelled exploratory.

## 6. Power (primary n = 32; sensitivities n ≈ 20–24)
- Wilcoxon two-sided, normal approximation with ARE 0.955: at α = 0.05/6 and n = 32, power 0.55 at d = 0.5, 0.96 at d = 0.8, 0.998 at d = 1.0. DEV effects for (i) and (ii-a) were d ≈ 1.0–1.9, so they stay well powered in the sensitivity sets too (n ≈ 20–24).
- (ii-c) TOST, true Δ = 0, SD 0.022 (DEV): power ≈ 0.62 at α = 0.05 with n = 32 (its Holm step when the five others reject; under the cross-study Holm it is also the last step if every other test rejects), ≈ 0.40 with n = 24 (SENS-INT), and ≈ 0.03 at α = 0.05/6. "Unresolved" is a likely outcome and is stated now.
- S-3 (ρ): the minimum detectable ρ at 80% power, one-sided α = 0.05, is ≈ 0.43 (n = 32) and ≈ 0.49 (n = 24, SENS-INT); DEV gave ρ ≈ −0.51.
- S-5: HO is drawn like DEV (DEV: 14 binary, 6 multiclass), so the multiclass group is expected to be small.

## 7. Run plan
- Order: fm2 and fm5 first (they serve (ii)), then fm1, fm3, then the classical group.
- One torch/BLAS thread per worker, `nice -n 10`, background, logged to `logs/study13_*.log` (the machine is shared).
- Launch (after the freeze commit): `cd simaudit && P1_HELDOUT_GO=1 OMP_NUM_THREADS=1 nohup nice -n 10 uv run python -m simaudit.runner.study13 --set heldout --workers 4 > logs/study13_heldout.log 2>&1 &`. The DEV dry run is the same command with `--set dev`, without the variable.
- No outcome is analysed before every confirmatory job (arm T, four FMs) has finished or definitively failed. There is no interim look.
- Unit tests (synthetic data): the HELD-OUT guard; the manifest and pool-hash checks; the fixed split (feasibility, class presence, determinism); redraw infeasibility; namespaces; resumable outputs; the classical learners; the analysis family and decision rules; the random-effects estimator.

## 8. Compute estimate
**Sources.** The `seconds` column of the Study 12 files finished by 2026-10-27 (fits that were not reused): TabICL 6,034 fits on 50 tables, TabPFN v3.5 667 fits on 6 tables. On T1′ tables with n ≤ 1000 (HO's range): **TabICL 2.2 s per fit** (33 tables), **TabPFN v3.5 3.7 s** (5 tables). TabPFN v2 and Kumo have no Study 12 file yet; the Study 12 timing probe gives ratios to TabICL of 1.6–2.1 (TabPFN v2) and 3.6–4.2 (Kumo), so ≈ 4.4 s and ≈ 8.8 s. Study 12 also needed ≈ 120 non-reused fits per (table, FM) for 100 stratified/random cells plus redraws (n ≤ 1000: 118).

**Grid per (table, FM).** Arm T ≈ 120 fits (100 + redraw refits); arm F = 100 fits, each predicting ≈ 30% of the rows (assumed 0.6× the time of an arm-T fit).

**HO size.** Unknown until opening; assumed DEV-like (DEV: Σn = 8,363, mean n = 418, mean C = 3.2).

| Item | Fits | Worker-hours |
|---|---|---|
| Arm T, 4 FMs | 32 × 4 × 120 ≈ 15,400 | 32 × 120 × 19.1 s ≈ 20.4 |
| Arm F, 4 FMs | 32 × 4 × 100 = 12,800 | 32 × 100 × 19.1 s × 0.6 ≈ 10.2 |
| Classical, both arms | 32 × 6 × 220 ≈ 42,000 (sub-second) | ≈ 1 |
| **Total** | ≈ 28,000 FM fits | **≈ 32**, ≈ 45–50 with a 1.5× margin for the shared machine |

At 4 workers ≈ 8–12 wall-hours; at 2 workers ≈ 16–24. For the joint opening: Study 9 HELD-OUT ≈ 75 worker-hours (Study 9 Freeze note 1), Study 10 HO ≈ 85 (Study 10 Sec. 11), so ≈ 190–210 worker-hours in total.

## 9. Access procedure (HELD-OUT)
1. **Who.** Only the authors authorise the opening. The runner is launched only after that authorisation; nobody reads `sealed_heldout/` by hand.
2. **Preconditions enforced by the runner** (`runner/study13.check_heldout_go`); any failure raises before any sealed file is touched:
   - environment variable `P1_HELDOUT_GO=1`;
   - `prereg/STUDY13_HELDOUT_CONFIRMATION.md` exists, its status line reads `Status: FROZEN`, it carries no DRAFT marker, and `prereg/DRAFT_STUDY13_HELDOUT_CONFIRMATION.md` no longer exists;
   - the frozen file is tracked by git and identical to `HEAD` (no uncommitted change).
3. **Hash check at run time, by the runner.** It reads `sealed_heldout/MANIFEST.json` (`npz_sha256`), requires its IDs to equal the 32 `pool_i2.yaml` HELD-OUT IDs, and hashes every npz file's bytes against the manifest **before** parsing any of them. A mismatch aborts the run with nothing parsed. Each loaded table's (X, y) content hash must then equal its `pool_i2.yaml` `sha256`.
4. **Logging.** Every access (manifest read, file hash, npz load) is appended with a UTC timestamp to `results/study13/heldout/access_log.jsonl`. The provenance file records the git HEAD, the SHA-256 of the frozen prereg and of `runner/study13.py` and `analysis/study13.py`, the base seed and the namespaces.
5. **Integrity screen** runs and its flags are written (Sec. 1) before any selection or fit; it excludes nothing.
6. **One execution.** A crash may be resumed (outputs are per table × learner group), but nothing may be re-run with changed code or settings. The access log and the screen file are committed with the report.

## 10. Threats
- **Same population as DEV.** HO was drawn with DEV from one frame (small all-numeric OpenML tables, n ≤ 1000). Replication on HO shows that the DEV results were not a sample accident; it does not show generality beyond that frame. T1′ (PMLB/UCI, up to n = 5000) covers a different frame.
- **Small n.** 32 tables (fewer in SENS-INT); (ii-c) and the multiclass stratum are underpowered (Sec. 6).
- **Shared use.** Studies 9 and 10 plan to use HO (Sec. 0.3); the cross-study Holm addresses multiplicity, the joint opening addresses contamination.
- **Fixed split uses labels to define T_d.** Identical for every protocol and draw, so it cannot favour either; it does change the evaluation population slightly (class-stratified rather than uniform).
- **Learners.** The fm*_bal transform is post hoc (exact for TabPFN per Study 7 Part (b)). TabPFN v3.5 weights are gated and research-only.
- **Shared machine.** Timing does not affect results.

## 11. Opening note and amendments
- **2026-10-27:** DRAFT written; runner, analysis and tests committed; unit tests on synthetic data (28 pass). DEV smoke test without any model (integrity screen, fixed split, selections; output in a temporary folder, not in results/): 15 of 20 DEV tables kept; on every kept table the fixed split is feasible and every class is in T_d (smallest T_d class count 5, table 694); no redraw cell is infeasible on DEV. `sealed_heldout/` not opened for Study 13. Open before the freeze: the choice between options A and B of Sec. 0.3; the cross-study Holm rule (Sec. 4.4); the compute budget (Sec. 8).
- **2026-10-27, decisions (recorded in research_notes/WORKSTREAMS.md):**
  1. **Option A, joint opening** (Sec. 0.3): Studies 9 and 13, plus Study 10 if it is frozen by 2026-10-31 23:59 UTC; otherwise Studies 9 and 13 only, and Study 10 records that HELD-OUT was already opened and what was seen. Cross-study Holm under WORKSTREAMS rule 4 (Sec. 4.4).
  2. **All tables stay in the primary analysis;** the A2.7 screen only defines the sensitivities SENS-INT and SENS-ID (Sec. 1). Power statements updated (Sec. 6). The runner was changed accordingly (flag-only screen); 29 unit tests pass.
  3. The fixed evaluation split (S-4) and the compute budget (Sec. 8) are approved.
- **Study 9 HELD-OUT runner status (checked 2026-10-27): not ready.** `runner/study9.run_job` accepts a `loader`, but `runner/study9.run` does not pass one, and the CLI deliberately does not expose HELD-OUT (`cli._study9_jobs`). There is no FROZEN/go guard and no manifest check. The default `openml_loader.load` would download the HELD-OUT tables from OpenML into `data/cache/`, **bypassing the seal**. Study 9's runner and analysis are hash-frozen (Study 9 Freeze note 1), so they must not be edited. A separate thin driver is needed before the joint opening. It would call `study9.run_job(..., loader=study13.SealedLoader(...))` after `study13.check_heldout_go`-style checks (Study 9's own prereg and go variable) and `study13.verify_manifest`, and it should be committed (with a dated Study 9 amendment, if one is needed) before the opening. Not written here.

## Amendment 1 (2026-10-28, before any HELD-OUT access)

**Scope and timing.** This amendment was written and committed on 2026-10-28, before the opening note (Sec. 11), before the 2026-10-31 deadline, and before any Study 13 run. `sealed_heldout/` has not been opened for Study 13 or for this amendment. No HELD-OUT data, label, metadata or manifest content was read, and no integrity screen was run on HELD-OUT. The frozen text above is unchanged. **The primary hypotheses (Sec. 3), the decision rules (Sec. 4), the within-study Holm family of 6 and the cross-study Holm family (Sec. 4.4) are unchanged.** Every analysis added here is secondary. Each has a stated direction, test and Holm family, fixed before any HELD-OUT outcome exists: in that sense it is *exploratory-confirmatory*. None carries confirmatory weight, none enters the cross-study Holm family, and none may be called "confirmed on HELD-OUT".

**Seen for this amendment (disclosed).**
- The frozen text above, the Study 13 and Study 9 HELD-OUT code, `prereg/STUDY9_HOW_MANY_LABELS.md`, `docs/papers/P1_P2_BOUNDARY.md` and the joint-opening section of `research_notes/WORKSTREAMS.md`.
- The E17 report (`paper/p1/exploratory/e17_metrics/e17_report.md`) and its independent verification (`paper/p1/exploratory/verify/`, caveats in `research_notes/FINDINGS_LOG_P1.md`, "Independent verification").
- The DEV Study 7 Part (a) outputs of fm6 and fm7 (`paper/p1/exploratory/fm_extra/out/s7a`, `s7a_sel`), read only for adapter checks (B.6) and fit times (D). The fm_extra smoke summary (2 DEV tables, `out/smoke_summary.csv`) was also displayed. No setting was chosen or changed on any of these outputs.
- The DEV raw OpenML caches (`data/cache/openml`, not sealed), used to check the label-reading code on the 20 DEV tables (A.3).
- The T1′ label problem of UCI 257 (an internal to-do note).

### A. Label-normalisation integrity step

**A.1 Why.** In T1′ table UCI 257 ("User Knowledge Modeling"), "Very Low" and "very_low" are stored as two classes (5 classes instead of 4). A spurious split of this kind changes C, the class counts, the stratified draws (m rows for each spelling) and every class-averaged metric. HELD-OUT may contain the same problem, and it must not be looked for before the opening. So the check is a mechanical step declared now and applied only inside the single confirmatory run.

**A.2 The rule.** It is applied to every raw label string of a table:
1. A label that is a plain number after stripping (regex `^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$`) is left unchanged.
2. Every other label is lowercased. Each run of whitespace, `_`, `-` and `.` becomes one space, and the result is stripped. So "Very Low", "very_low", "VERY-low." and "very  low" all become "very low".
3. Labels whose normalised strings are equal are merged into one class.
4. The merged class takes the smallest old code of its members, and the codes are then made contiguous (0..K′−1) in their old order. Without any merge the mapping is the identity, so the table is used exactly as stored.

**A.3 Where the raw strings come from.**
- The sealed npz files store integer codes only. `openml_loader.encode` codes a label as the rank of its string among the table's sorted distinct strings.
- Inside the run, the runner reads the raw strings from the table's raw OpenML cache in `sealed_heldout/`, which was moved there with the npz files on 2026-10-20 (Study 3 log). It reads `description.xml` (only `default_target_attribute`) and the target column of `dataset_{did}.pq`. It does not call the `openml` package, use the network or unpickle anything.
- The strings are used only if they re-encode, row by row, to exactly the stored codes. That check runs after the manifest and content-hash checks of Sec. 9.3, which still run on the stored codes.
- If the cache folder is missing or ambiguous, or the strings do not reproduce the codes, the table keeps its stored labels. It is recorded as unverified (`raw labels not found` or `raw labels do not reproduce the stored codes`).
- **DEV check of this code (2026-10-28):** on all 20 DEV tables, the raw strings in the non-sealed DEV cache reproduced the stored codes exactly. No DEV table has a merge.

**A.4 When, and what is logged.**
- The step is applied once per table at load time, inside the confirmatory run. It comes after the guard, the manifest check and the content-hash check, and before the integrity screen, any selection and any fit.
- Before the screen, the runner writes `results/study13/heldout/label_normalisation.csv`, one row per table: key, status, the raw strings, the normalised strings, the old→new mapping, whether a merge happened and the merged groups.
- Every load also appends the reads of the two raw files and a `label normalisation` entry with the same mapping to `access_log.jsonl`.
- Both files are committed with the report (Sec. 9.6). The run's provenance (`provenance_amendment1.json`) records that the step was on and the SHA-256 of `data/labels.py`.

**A.5 Consequences.**
- C, the class counts, the integrity screen, the fixed split, the draws and all metrics use the normalised labels.
- n is unchanged, so the row permutation and every seed are unchanged.
- A merge can only raise class counts, so the eligibility facts of Sec. 1 (2 ≤ C ≤ 10, smallest class ≥ 11) still hold, and C can only go down.

**A.6 Analysis.**
- The **primary analysis uses the normalised labels**.
- **SENS-LABEL (pre-declared):** the confirmatory family of Sec. 3, recomputed without the tables where a merge happened, with Holm within the reduced family. A verdict that changes is reported as "label-fragile".
- If some table is unverified, **SENS-LABEL+** also drops the unverified tables.
- If no merge happens, SENS-LABEL equals the primary analysis, and the report says so.
- The number of merged and of unverified tables is reported in every case.

**A.7 Study 9 HELD-OUT.**
- **The step also applies to Study 9's HELD-OUT run**, because the driver `runner/study9_heldout.py` loads through the same `SealedLoader`. Both studies of the joint opening must use the same labels.
- The driver now uses `SealedLoader(normalise_labels=True)` and writes `results/study9/heldout/label_normalisation.csv` before any selection.
- Study 9 Amendment 2 (appended to `prereg/STUDY9_HOW_MANY_LABELS.md` in the same commit) records this.
- The driver's pin check (`PINNED`) covers only `runner/study9.py` and `analysis/study9.py`. Neither changes, and both still match their Freeze-note-1 hashes. The loader (`runner/study13.py`), `data/labels.py` and the driver itself are not pinned; the driver writes its own SHA-256 into its provenance.

**A.8 Code.**
- `src/simaudit/data/labels.py` (rule, raw-label reading, verification, mapping).
- `runner/study13.SealedLoader(normalise_labels=...)` (off by default) and `runner/study13.label_log`. `main()` turns the step on for `--set heldout`.
- DEV runs use `openml_loader.load`, which does not apply the step.

### B. New FMs as secondary learners: fm6 TabDPT v1.3 and fm7 Mitra-v2

**B.1 Learners and pins** (exactly `paper/p1/exploratory/fm_extra/out/pins.json`; a unit test checks the copy in the worker):

| ID | Model | Venv | Packages | Weights (HF repo @ revision, file, SHA-256) | Licence |
|---|---|---|---|---|---|
| fm6 | TabDPT v1.3 | `.venv_fm_extra` | tabdpt 1.3.1, torch 2.14.1, scikit-learn 1.9.1, numpy 2.5.4 | Layer6/TabDPT @ a5ca6e01c0fa09ec68c73e958e5199d1932abb3a, `tabdpt1_3.safetensors`, 97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb | Apache-2.0 |
| fm7 | Mitra-v2 (mitra-classifier-2) | `.venv_fm_extra_mitra` | autogluon.tabular 1.6.3, torch 2.10.0, numpy 2.5.4 | autogluon/mitra-classifier-2 @ edada0d20759c58ada8c8605c25f22f6e98ea5f0, `model.safetensors`, 5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09 | Apache-2.0 |

LimiX-16M (fm8) is **not** included.

**B.2 Settings.** These are the settings of the DEV adapters (`paper/p1/exploratory/fm_extra/adapters.py`): CPU, one thread, in-context only.
- **TabDPT:** default 8-member ensemble with the cell seed, `compile=False`, `use_flash=False`, and the pinned weight file passed explicitly.
- **Mitra:** `fine_tune=False`, `n_estimators=1`, the cell seed, the context passed as its own validation set, and **fp32** instead of the package's bf16 autocast.

**One declared addition for Mitra.** AutoGluon 1.6 no longer seeds per fit: its own comment says `set_seed` was removed in v1.4. Mitra's random preprocessing (feature mirroring, class shuffle, feature order) therefore draws from the unseeded global numpy RNG.
- On DEV, two identical Mitra fits in the same process differed by up to 0.11 in probability.
- The worker now calls the package's own `set_seed(cell seed)` before every Mitra fit. With this, Mitra fits are bit-identical across call orders and fresh workers (DEV check, 8 cells).
- So the stored DEV fm7 outputs cannot be reproduced draw by draw, and HELD-OUT fm7 fits will not equal what the unseeded adapter would have produced.
- This is a reproducibility fix, chosen without looking at any metric; it is not tuning.
- fm6 through the worker reproduced the stored DEV fm6 probabilities bit for bit (8 cells).

**B.3 Labels and missing classes.** fm6 and fm7 receive context labels re-encoded to 0..K−1 over the seen classes by `runner/study9.predict_on`, exactly like fm1–fm5. Rule L0 holds: unseen classes get probability 0, and a one-class context gives a constant prediction without calling the FM.
- The worker refuses codes that are not exactly 0..K−1, which guards against the Mitra gapped-label pitfall (P4, I7).
- It also refuses non-finite or wrongly shaped outputs.

**B.4 Mitra class limit (pre-declared).** fm7 is skipped on tables with more than 10 classes, counted after label normalisation. The skip is recorded in `provenance_amendment1.json`. HELD-OUT has C ≤ 10 by eligibility, so the rule is not expected to fire.

**B.5 How the runner calls them.**
- `runner/study13.ExternalFM` starts `runner/fm_extra_worker.py` in the FM's venv (`{venv}/bin/python -I fm_extra_worker.py fm6|fm7`), one worker per (table, FM) job, closed at the job's end.
- The worker uses `HF_HUB_OFFLINE=1` and one thread. It checks the package versions and the weight file's SHA-256 offline before its first answer; any mismatch fails the job.
- Arrays travel as length-prefixed npz frames, loaded without pickle. The worker points its stdout at stderr so that library output cannot corrupt the protocol.
- The main `.venv` and `uv.lock` are untouched, and the worker imports only numpy, the standard library and the model packages.
- The verified pins are written to `provenance_amendment1.json`.

**B.6 DEV adapter checks (no tuning).** These checks used only the completed DEV Part (a) outputs.
- **Part (a) core:** fm6 and fm7 each have 2,000 fits on 20 tables.
- **Selector cells:** 1,600 fits each.
- **Results:**
  - no non-finite probability;
  - largest |row sum − 1| of 2.8 × 10⁻⁷;
  - no probability mass on an unseen class (L0 holds);
  - one-class contexts are constant (223 and 139 fits).

**B.7 Grid.**
- Arm T only, with `stratified` (5 reps) and `random` (20 reps) at m ∈ {1, 3, 5, 10}.
- These are the same stored selections and namespace-34 seeds as the four FMs.
- No redraw, no arm F, and no fm6/fm7 in S-1 to S-9.
- The jobs run after the classical group (`GROUPS` = fm2, fm5, fm1, fm3, classical, fm6, fm7).
- The failure rule of Sec. 1 applies: complete case per contrast; more than 10% of tables lost means "not evaluable".
- S-10 and S-11 are analysed with the rest, after every Study 13 job has finished or definitively failed.

**B.8 Analyses (each its own Holm family; arm T; unit = table; mean over reps, then over m; macro-F1).**
- **S-10, gap per new FM.** Stratified − random per FM ∈ {fm6, fm7}, mean over m ∈ {1, 3, 5, 10}. Wilcoxon signed-rank, two-sided, **Holm over 2**.
  - *Direction:* Δ > 0 for both. This comes from the coverage/balance mechanism and the four existing FMs, not from the fm6/fm7 DEV outputs.
  - *Reading:* "the optimism extends to fm6/fm7" iff both are Holm-significant with Δ > 0.
  - Reported beside it, unadjusted: accuracy and balanced accuracy.
- **S-11, FM gap minus classical gap (E13 style).** D = Δ_FM − the mean classical Δ over {lr, knn1, gnb, lda, hgb} (knn3 excluded, as in S-7), mean over m. Wilcoxon two-sided, **Holm over 2**. No direction is predicted.
- Robustness as in Sec. 3: mean, median, win rate, bootstrap 95% CI, TOST p.

### C. Secondary metric families from E17

**C.1 Which.** Only the E17 families that survived the independent verification are included:
- kappa and MCC;
- all-class against seen-class AUC;
- calibration-in-the-large (CITL) on the more imbalanced half.

Not included: accuracy and Brier in the imbalanced half (not significant after Holm on verification); the 1–3C log-loss/ECE gaps (an artefact of p = 0 and the clip); LR+/LR−; macro precision; F-beta; the other E17 metrics.

**C.2 Metrics.** Every fit now also stores the E17 definitions, `analysis.study13.e17_panel`, on the query rows:
- `kappa` (Cohen) and `mcc`, on the argmax over all C columns: 0 when undefined, as in E17; checked against scikit-learn in the tests;
- `citl` = Σ_k |mean p_k − share_k(query)|;
- the `pc_*` versions of all three.

The existing `auc_all` and `auc` (seen classes, Study 7 convention) are reused.

**C.3 Families (arm T; four confirmatory FMs; unit = table; Wilcoxon signed-rank, two-sided; each family its own Holm).**
- **S-12, kappa and MCC.** Stratified − random per FM × metric, mean over m ∈ {1, 3, 5, 10}. **Holm over 8.** *Direction:* Δ > 0.
- **S-13, all-class against seen-class AUC.**
  - Per table: D = (stratified − random in `auc_all`) − (stratified − random in `auc`) at m = 1, complete cases. Seen-class AUC is undefined for a one-class context, so its table means use the defined draws.
  - **Holm over 4.** *Direction:* D > 0, meaning the AUC gap comes from scoring unseen classes.
  - The mean over m is reported unadjusted.
- **S-14, CITL on the more imbalanced half.**
  - *Imbalance:* TV = ½‖π − u‖₁ of each table's class shares. The shares come from the class counts that the run records at load time (`integrity_screen.csv`, after A). In arm T the pool is the whole table.
  - *Split:* median split over the analysed tables; the more imbalanced half is TV > median, and ties go to the other half.
  - *Test:* per FM, CITL(stratified) − CITL(random), mean over m ∈ {3, 5, 10}. m = 1 is excluded because coverage reverses the sign at 1C (E17). **Holm over 4.**
  - *Direction:* > 0 (stratified contexts are worse calibrated in the large).
  - The other half is reported unadjusted.
  - With n ≈ 16 tables per half, power is modest.

**C.4 Status.** S-10 to S-14 are secondary and exploratory-confirmatory. They do not change any primary hypothesis, decision rule, the C1 or ranking claims (Sec. 4.2) or the cross-study Holm. A reversed or null result is reported as it is.

### D. Compute (re-estimate)
- **Sources.** The four FMs keep Sec. 8 (TabICL 2.2 s per fit, TabPFN v3.5 3.7 s, TabPFN v2 ≈ 4.4 s, Kumo ≈ 8.8 s).
- **New-FM timing ratios.** For fm6 and fm7, the ratios to TabICL come from the DEV Part (a) fits on the same 2,000 cells:
  - *Mean seconds:* fm6 1.25×, fm7 1.80×.
  - *Per-table median:* fm6 1.63×, fm7 1.45×.
  - The same data give fm1 1.66×, fm3 3.62× and fm5 2.23×, consistent with Sec. 8.
  - Taking the larger ratio, fm6 ≈ 3.6 s and fm7 ≈ 4.0 s per fit.
  - Worker start (model load and pin check) took about 10–16 s per job on DEV.

| Item | Fits | Worker-hours |
|---|---|---|
| Sec. 8 total (four FMs, arms T and F, classical) | ≈ 28,000 FM fits | ≈ 32 |
| fm6 + fm7, arm T stratified/random | 32 × 2 × 100 = 6,400 | 32 × 100 × 7.6 s ≈ 6.8, plus ≈ 0.3 worker start |
| E17 metrics, label step | — | negligible |
| **Study 13 total** | ≈ 34,400 | **≈ 39**, ≈ 55–60 with the 1.5× margin |

- **Wall time.** About 10–15 hours at 4 workers.
- **Joint opening.** Study 9 HELD-OUT is unchanged at ≈ 75 worker-hours; the label step adds nothing measurable. Study 10 HO is ≈ 85 if it joins. Total ≈ 200–225 worker-hours (Sec. 8: 190–210), or ≈ 115 for Studies 9 and 13 only.

### E. Guards and tests
- The guard `check_heldout_go` (Sec. 9.2) reads the first `Status:` line (FROZEN), requires no DRAFT marker, and requires this file to be committed and identical to HEAD. Appending this section and committing it keeps all three true.
- The Study 9 driver's checks also still pass: Study 9 FROZEN and committed unchanged, pinned hashes equal.
- **Unit tests** (`tests/unit/test_study13_amendment1.py`, synthetic data only) cover:
  - the rule on UCI-257-like and other labels, the identity without collisions, and unverified tables;
  - the sealed loader with a synthetic raw cache, the logs, and draws on merged classes;
  - the worker protocol and pins;
  - the fm6/fm7 grid and the Mitra skip;
  - the E17 metrics against scikit-learn;
  - the Holm sizes of S-10 to S-14 and SENS-LABEL.

  `tests/unit/test_study9_heldout.py` was updated for the extra label-log loads.

## Amendment 2 (2026-10-28, before any HELD-OUT access): unverified labels and SENS-LABEL+

**Source.** This decision was taken on 2026-10-28. It was committed before the opening note and before any HELD-OUT access. `sealed_heldout/` has not been opened, and nothing about HELD-OUT was read.

**Decision (option (a)).**
- The fallback of Amendment 1 (A.3) stays. If a table's raw label cache is missing or ambiguous, or does not re-encode to the stored codes, the table runs with its stored labels, unnormalised, and is flagged "unverified" in `label_normalisation.csv` and the access log.
- **New rule for SENS-LABEL+ (Amendment 1 A.6).** SENS-LABEL+ (the confirmatory family without merged and unverified tables) is reported only if the unverified tables number **at most half** of the HELD-OUT tables in the label log (≤ 16 of 32).
- Otherwise the report states that SENS-LABEL+ is not reported, with the count of unverified tables and this rule as the reason.
- SENS-LABEL (merged tables dropped) is reported in every case.

**Unchanged.** Everything else in this file and in Amendment 1: the primary hypotheses, the decision rules, both Holm families, the label rule and its fallback, and the secondary analyses.

**Code and test.** `analysis/study13.label_sensitivity` (constant `MAX_UNVERIFIED_SHARE = 0.5`; the note goes in `sens_label_plus_note`). Synthetic test `test_sens_label_plus_only_up_to_half_unverified` checks 6/12 unverified (reported) and 7/12 (not reported).

## Amendment 3 (2026-10-29, before any HELD-OUT access): fm8 LimiX-16M as a third secondary FM

**Source and timing.** The authors want P1 to report seven FMs throughout (fm1, fm2, fm3, fm5 and fm6, fm7, fm8). HELD-OUT is the only set where the extra FMs can still get a pre-registered test. This amendment was written and committed on 2026-10-29, before the opening note (Sec. 11), before the 2026-10-31 deadline and before any Study 13 run. `sealed_heldout/` has not been opened, listed or read for this amendment, and nothing about HELD-OUT was read. The code it needs was committed first, in a separate commit (E).

**Unchanged.** The primary hypotheses (Sec. 3), the decision rules and claims (Sec. 4), the within-study Holm family of 6, the cross-study Holm family (Sec. 4.4), Amendments 1 and 2, and every secondary family other than S-10 and S-11. fm6 and fm7 keep their pins, settings, grid and code path (E.3 shows their outputs are bit-identical).

### A. fm8 LimiX-16M: pins, settings, licence

**A.1 Pins** (exactly `paper/p1/exploratory/fm_extra/out/pins.json`, entry `fm8`; a unit test checks the copy in the worker):

| Field | Value |
|---|---|
| Model | LimiX-16M (LimiX-1_16M) |
| Venv | `.venv_fm_extra_limix` |
| Packages | limix 2.0.0, torch 2.14.1, numpy 2.5.4 |
| Code | github.com/limix-ldm/LimiX @ 516bf396333feb3198cf7aff8a6c10421f218e24 |
| Weights | stable-ai/LimiX-1_16M @ 3a7c18ac7876fa0f4418cc077abc12630c4fefbf, `LimiX-1_16M.ckpt`, SHA-256 0862b8aacc4669ead009ad418c83d33c143396b7aeec069e7d0d667aa70a168a |
| Settings | config `cls_default_noretrieval.json`, `mix_precision=False` |

One pin is new and is not in pins.json: the SHA-256 of the config file the worker loads, `cls_default_noretrieval.json` = 9ad6baac6d4cb8d7f218b76078dbdc801ce9b253f5b7868342df0c2e0b0a1e2f (hashed on 2026-10-29 from the installed package).

**A.2 Settings.** These are the settings of the DEV adapter (`paper/p1/exploratory/fm_extra/adapters.py`, `limix_predict`), which produced every fm8 output listed in B:
- CPU, one torch thread, in-context only (no fine-tuning);
- the package's default no-retrieval classification config (`cls_default_noretrieval.json`), `mix_precision=False` (fp32), the cell seed;
- one `LimiXPredictor` per fit, built through the package's router (`inference.predictor.LimiXPredictor`) from the checkpoint dict loaded once per worker;
- the adapter's two technical shims, unchanged: a placeholder `triton` module (LimiX v1 imports triton for a CUDA kernel that its CPU path never calls) and a temporary-directory default for the package's hard-coded cache folder (nothing is cached; `use_data_cache` is off).

**A.3 Weights: pinned SHA and `weights_only`.**
- The worker first checks the package versions and the weight file's SHA-256 offline, as for fm6 and fm7 (Amendment 1 B.5).
- It then reads the checkpoint bytes once, hashes those same bytes against the pin, and only then loads them with `torch.load(..., map_location="cpu", weights_only=True)`. Any mismatch refuses to run: the worker answers with an error and the job fails.
- The checkpoint is a plain `{state_dict, config}` dict. On 2026-10-29, `weights_only=True` and `weights_only=False` gave identical tensors and config (checked on the cached file). The DEV Part (a) run used `weights_only=False`; the T1′ and Part (c) runs used `weights_only=True`.

**A.4 LimiX-2 is forbidden.** The package (limix 2.0.0) contains both the LimiX v1 code (`inference/v1_0`, `model/v1_0`) and the LimiX-2 code (`v2_0`). The worker loads only the pinned LimiX-1_16M checkpoint and refuses any checkpoint that the package routes to a line other than `v1_0`. The LimiX-16M checkpoint has no `arch_version` field, so the package treats it as 1.0 and routes it to `v1_0` (checked). No LimiX-2 weights are on disk, and none are downloaded.

**A.5 Licence (verified from the files on disk, 2026-10-29).**
- The installed package's licence (`.venv_fm_extra_limix/.../limix-2.0.0.dist-info/licenses/LICENSE.txt`) is byte-identical to `LICENSE.txt` of the source clone at commit 516bf396. It is the "Stable AI Technology Co., Ltd. License, Version 1.0, September 2026": Sections 1–9 are Apache-2.0, and Section 10 adds attribution and naming conditions ("Built with StableAI LimiX"; a distributed model derived from the weights must start its name with "LimiX").
- Section 10 says that use "solely for internal research, evaluation, benchmarking, or testing", without distributing the work or a derived model, does not trigger the attribution or naming requirements. P1 distributes neither the weights nor a model derived from them.
- The README at the same commit says that the LimiX-1-16M weights are under the same Stable AI Technology Co., Ltd. License, and that the LimiX-2 weights are under a separate non-commercial licence. The weights' own licence file is not in the local Hugging Face cache, so the weight licence rests on that README statement.
- This matches the pins.json entry ("Stable AI Technology Co., Ltd. License v1.0 (Apache-2.0 + Sec. 10 attribution/naming)").

**A.6 Labels and missing classes.** As fm6 and fm7 (Amendment 1 B.3): context labels re-encoded to 0..K−1 over the seen classes by `runner/study9.predict_on`; unseen classes get probability 0; a one-class context gives a constant prediction without calling the FM. The worker refuses label codes that are not exactly 0..K−1 and non-finite or wrongly shaped outputs.

**A.7 Class limit (pre-declared).** The LimiX-16M decoder has 10 outputs (`decoder_config.num_classes = 10` in the checkpoint). So fm8 is skipped on tables with more than 10 classes, counted after label normalisation, and the skip is recorded in `provenance_amendment1.json` (`limix_max_classes`, `skipped_jobs`). HELD-OUT has C ≤ 10 by eligibility, so the rule is not expected to fire.

**A.8 Grid and order.** As fm6 and fm7 (Amendment 1 B.7): arm T only, `stratified` (5 reps) and `random` (20 reps) at m ∈ {1, 3, 5, 10}, the same stored selections and namespace-34 seeds as the four FMs. No redraw, no arm F, and no fm8 in S-1 to S-9 or S-12 to S-14. `GROUPS` = fm2, fm5, fm1, fm3, classical, fm6, fm7, fm8. The failure rule of Sec. 1 applies (complete case per contrast; more than 10% of tables lost means "not evaluable").

### B. Provenance: fm8 outputs already seen (disclosed)

fm8 has already been run on DEV and T1′, and its outputs have been seen. No fm8 setting was chosen or changed on any of them. The settings (A.2) are the package defaults plus fp32 on CPU and were fixed in the feasibility commit (R028), before any fm8 metric existed.

Seen, all under `paper/p1/exploratory/fm_extra/out/`:
1. **Feasibility** (R028): `probe_fm8.json` (timing and row-sum checks on DEV 46562) and the smoke run (`smoke/`, 2 fm8 files; `smoke_cells.csv`, `smoke_summary.csv`).
2. **Study 7 Part (a) on DEV:** `s7a/{did}_fm8.parquet` (20 tables, 2,000 stratified/random fits), `s7a_sel/{did}_fm8.parquet` (20 tables, selector cells), and the summaries `s7a_cells.csv` and `s7a_summary.csv`. Seen: the fm8 macro-F1 gap, stratified − random, mean over m = +0.049 [+0.030, +0.065] (20 tables), and the DEV accuracy gap.
3. **Study 12 T1′ design (exploratory; plan R029):** `t1p_fm8/` (52 tables), `t1p_fm8_pilot/` (3 files), `t1p_fm8_cells.csv`, `t1p_fm8_summary.csv`, `t1p_fm8_acc_summary.csv`, `t1p_fm8_numbers.tex`, `T1P_FM8_SUMMARY.md` (commits R030, R031). Seen: macro-F1 gap +0.067 [+0.056, +0.079], mean over m, positive at every m; accuracy gap −0.023 [−0.051, +0.000].
4. **Study 7 Part (c) dose on DEV (exploratory; plan R032):** `dose_c/{did}_fm8.parquet` (20 tables), `dose_c_pilot/` and `dose_c_refcheck/` (6 fm8 files each), `dose_c_opt_check.json`, `dose_c_raw.csv`, `dose_c_chain.csv`, `dose_c_verdicts.csv`, `dose_c_numbers.tex`, `DOSE_C_SUMMARY.md` (commits R033, R034). Seen: negative raw and p/π̂-corrected slopes in both arms; LimiX "inconclusive" under the r14 chain.
5. Also read for this amendment: the `seconds` column of `s7a/*_fm8.parquet` and `dose_c/*_fm8.parquet` and of the stored fm2 runs (D), and the stored probabilities of eight fm8 Part (a) cells of `s7a/` were compared with the worker (E.3).

So the S-10 direction for fm8 (Δ > 0) agrees with fm8's DEV and T1′ macro-F1 gaps that were already seen. The direction is not taken from them: it is the direction of the coverage/balance mechanism and of the four primary FMs, as for fm6 and fm7 (Amendment 1 B.8). Because fm8's DEV and T1′ outputs are known, a HELD-OUT result for fm8 is a replication on new data of an effect already seen on DEV and T1′, not a blind prediction.

### C. Analyses (secondary, exploratory-confirmatory; arm T; unit = table; mean over reps, then over m; macro-F1)

**C.1 S-10, gap per new FM: widened to {fm6, fm7, fm8}, Holm over 3.**
- Stratified − random per FM ∈ {fm6, fm7, fm8}, mean over m ∈ {1, 3, 5, 10}. Wilcoxon signed-rank, two-sided, **Holm over 3** (was 2).
- *Direction:* Δ > 0 for all three. It comes from the coverage/balance mechanism and the four primary FMs, not from the fm6–fm8 DEV or T1′ outputs (B).
- *Reading:* "the optimism extends to fm6–fm8" iff all three are Holm-significant with Δ > 0. Per FM: "extends to fmX" iff fmX is Holm-significant with Δ > 0.
- Reported beside it, unadjusted, as now: accuracy and balanced accuracy.

**C.2 S-11, FM gap minus classical gap: widened to {fm6, fm7, fm8}, Holm over 3.** D = Δ_FM − mean classical Δ over {lr, knn1, gnb, lda, hgb}, mean over m. Wilcoxon two-sided, **Holm over 3** (was 2). No direction is predicted, as now.

**C.3 Family size is fixed.** If an FM has no analysable table (for example every job failed), it stays in the family with n = 0 and p = 1 and is reported as "not evaluable"; the family size stays 3. Robustness as in Sec. 3: mean, median, win rate, bootstrap 95% CI, TOST p.

**C.4 No accuracy family is added.** The primary family (Sec. 3) has no accuracy hypothesis: all six contrasts are macro-F1. Sec. 3 reports the accuracy versions of its contrasts unadjusted and without a direction, and S-3 states that the mean accuracy gap is not predicted to differ from 0. So there is no accuracy hypothesis to mirror, and a directional accuracy test for the new FMs would be a new claim, not a mirror. The accuracy gap of fm6, fm7 and fm8 stays reported, unadjusted, beside S-10 (C.1). (The label S-12 is in any case already used, by the kappa/MCC family of Amendment 1 C.3.)

**C.5 Status.** S-10 and S-11 stay secondary and exploratory-confirmatory. Neither enters the within-study Holm family of 6 or the cross-study Holm family, and no fm8 result may be called "confirmed on HELD-OUT". The C1 and ranking claims (Sec. 4.2) still rest only on fm1, fm2, fm3 and fm5.

### D. Compute

**D.1 Sources.** The `seconds` column of the DEV runs, on the same cells as the stored TabICL (fm2) fits:
- **Study 7 Part (a)** (`s7a/*_fm8.parquet` against `results/study7/a/*_fm2.parquet`, 2,000 cells on 20 tables): LimiX 5.99 s per fit against TabICL 2.89 s, a **mean ratio of 2.07×**; the median over tables of the ratio of table means is 1.50×. (The same method gives fm6 1.25× / 1.63× and fm7 1.80× / 1.45×, the Amendment 1 D values.)
- **Part (c) dose** (`dose_c/*_fm8.parquet` against `results/study7/c/*_fm2.parquet`, 6,180 cells on 20 tables): LimiX 3.95 s against TabICL 4.90 s, ratio 0.81× (per-table median 0.48×). This run had `VECLIB_MAXIMUM_THREADS=1` and the network built once per process, the fm2 run had neither, so this ratio is optimistic and is not used for the budget.
- Planning value: the larger ratio, 2.07× of TabICL's 2.2 s (Sec. 8), so **fm8 ≈ 4.6 s per fit**. Worker start (model load and pin check) took 10–16 s per job for fm6 and fm7 on DEV (Amendment 1 D); for fm8 the first probe call took 6 s longer than the next ones (`probe_fm8.json`). Peak memory ≈ 1.6–1.9 GB per worker (`dose_plan.md` Sec. 7).

**D.2 Budget.**

| Item | Fits | Worker-hours |
|---|---|---|
| Study 13 after Amendment 1 (Sec. 8 + fm6, fm7) | ≈ 34,400 | ≈ 39 |
| fm8, arm T stratified/random | 32 × 100 = 3,200 | 3,200 × 4.6 s ≈ 4.1, plus ≈ 0.2 worker start |
| **Study 13 total** | ≈ 37,600 | **≈ 43**, ≈ 62–67 with the 1.5× margin |

- **Added by this amendment:** ≈ 4.3 worker-hours (≈ 6.5 with the 1.5× margin).
- **Joint opening:** Study 9 HELD-OUT ≈ 75 and Study 10 HO ≈ 85 if it joins; total ≈ 205–230 worker-hours (Amendment 1: 200–225), or ≈ 120 for Studies 9 and 13 only (was ≈ 115).

### E. Code, tests and the DEV identity check

**E.1 Code** (committed before this amendment, in a separate commit, R035):
- `runner/fm_extra_worker.py`: `PINS["fm8"]` (copy of pins.json), `read_pinned` (hash the bytes, then use only those bytes), `limix_checkpoint` (`weights_only=True`), `_limix` (A.2–A.4; config SHA-256 check; `v1_0` only), and `build` dispatches fm6, fm7 and fm8 by name. The fm6 and fm7 functions are unchanged.
- `runner/study13.py`: `EXTRA_FMS` = (fm6, fm7, fm8), `EXTRA_VENVS["fm8"]` = `.venv_fm_extra_limix`, `GROUPS` ends with fm8, `LIMIX_MAX_CLASSES = 10` in `skip_job` and in `provenance_amendment1.json`. `ExternalFM` is unchanged; it starts `.venv_fm_extra_limix/bin/python -I fm_extra_worker.py fm8`.
- `analysis/study13.py`: `EXTRA_FMS` = (fm6, fm7, fm8), so S-10 and S-11 are Holm families of 3, with the fixed-size rule of C.3.
- The guard (`check_heldout_go`) is unchanged; appending this section and committing it keeps the FROZEN status line, no DRAFT marker, and the file identical to HEAD.

**E.2 Tests** (synthetic data only; `tests/unit/test_study13_amendment3.py`, 16 tests): the fm8 pin copy against pins.json and fm6/fm7 pins unchanged; the label-code guard (gapped, shifted, missing and negative codes refused; bad output shapes and non-finite outputs refused); refusal on a SHA mismatch (`read_pinned`; the checkpoint is never unpickled on a mismatch; the worker answers `error` and never `hello`); `weights_only=True` on exactly the pinned bytes, and refusal of a checkpoint routed to `v2_0`; the fm8 grid (arm T stratified/random only) and the class-limit skip; S-10 and S-11 as Holm families of 3, also with a missing FM. `tests/unit/test_study13_amendment1.py` was updated for the three-FM families and `GROUPS`.

**E.3 DEV identity check** (`paper/p1/exploratory/fm_extra/s13a3_identity/check_identity.py`, result `out/identity.json`). The runner's own learners, `runner.study13.external_fm(fm)` called through `runner.study9.predict_on` with query = every row, on 8 DEV cells: tables 1510, 40496, 337, 46562 × (stratified m = 1 rep 0, random m = 3 rep 1), with the stored Study 7 Part (a) labelled sets and seeds. One-thread settings (`OMP`, `MKL`, `OPENBLAS`, `VECLIB_MAXIMUM_THREADS` = 1, `torch.set_num_threads(1)`), `nice -n 15`, at most 2 processes.

| FM | Reference | Cells | max \|p − reference\| | Bit-identical (float32) | Max metric difference | New worker vs pre-Amendment-3 worker |
|---|---|---|---|---|---|---|
| fm8 | stored DEV Part (a) fm8 (`out/s7a`) | 8 | 0.0 | yes | 0.0 | (new FM) |
| fm6 | stored DEV Part (a) fm6 (`out/s7a`) | 8 | 0.0 | yes | 0.0 | bit-identical (max diff 0.0) |
| fm7 | stored seeded DEV fm7 (`out/s7a_v2`) | 8 | 0.0 | yes | 0.0 | bit-identical (max diff 0.0) |

- **fm8 through the worker reproduces the stored DEV fm8 Part (a) probabilities bit for bit** on all 8 cells (pass threshold was 1e-6). The stored fits used the DEV adapter with `weights_only=False`; the worker uses `weights_only=True` on the pinned bytes.
- **fm6 and fm7 are unchanged:** the new worker gives bit-identical probabilities to the worker as committed before this amendment (HEAD copy, run on the same cells) and to the stored DEV outputs.
- The complement metrics (`analysis.study7.complement_metrics`) are identical in every cell. The worker and runner files hashed at the start of the check equal the committed files.
- Fit times in the check: fm8 2.1–13.4 s, fm6 1.1–5.3 s, fm7 1.7–13.7 s per fit (two processes running).

- **2026-10-30, joint opening note:** see `prereg/JOINT_OPENING_2026-10-30.md` (studies, confirmatory cells, the cross-study Holm family of 19 tests, launch settings). Committed before any HELD-OUT access.
- **2026-10-30, label note (written after all confirmatory runners exited, before the joint analysis was run):** FM1 ("TabPFN v2") is the tabpfn 9.2.0 V2 default checkpoint `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` (sha256 cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49), i.e. Real-TabPFN (Garg et al. 2025). No HELD-OUT table is in its 71-table training list (checked by ID/name only). Label correction only; no change to the analysis.
