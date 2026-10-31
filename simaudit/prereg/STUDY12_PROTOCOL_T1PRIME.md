# Study 12: replication of Study 7 Part (a) on the confirmation set T1′

Status: FROZEN 2026-10-27, committed alone and before any Study 12 run. Changes only by dated amendments appended at the end, each committed before the run or analysis it affects.

The only computation done before freezing was a **timing probe** (not stored, not scored, no metric computed). It made one fit per FM on two T1′ tables (pmlb:churn, n = 5000, B = 20; pmlb:yeast, C = 9, B = 90), with arbitrary rows and at most 2000 predicted rows, on one thread. The per-fit seconds were:

| FM | churn | yeast |
|---|---|---|
| TabICL | 7.3 | 3.4 |
| TabPFN v2 | 15.0 | 5.4 |
| TabPFN v3.5 | 17.2 | 8.7 |
| Kumo | 30.5 | 12.3 |

The stored `seconds` column of Study 9's T1′ runs (a timing field, not an outcome) gives 7–22 s per fit at m ∈ {1, 3, 5, 10}.

`sealed_heldout/` is not touched by this study.

Code: `src/simaudit/runner/study12.py` (runner), `src/simaudit/analysis/study12.py` (analysis), driver `reports/study12_report.py`, report `reports/STUDY12_REPORT.md`. Outputs: `results/study12/{key}_{learner}.parquet`, one file per (dataset, learner). Files are written atomically, checkpointed, and skipped when present, so the run is resumable. Study 7's modules are reused read-only; no existing module changes behaviour.

## 0. Why, and what has already been seen

Study 7 (prereg/STUDY7_PROTOCOL_MECHANISM.md, DEV, n = 20) found the following (reports/STUDY7_REPORT.md):
- **(i) The stratified protocol is optimistic.** Stratified − random macro-F1, mean over 1C to 10C:
  - TabPFN v2 +0.059, TabICL +0.040, Kumo +0.048, TabPFN v3.5 +0.082;
  - all with Holm p ≤ 0.001.
- **(ii-a) The protocol reverses a ranking.** TabICL − TabPFN v3.5 under random, mean over 3C to 10C: +0.046 [0.028, 0.065].
- **(ii-c) The tie under stratified is not established.** −0.003 [−0.013, +0.007], TOST p = 0.11.
- **(ii-b) The reversal with TabPFN v3.5's balance option is unresolved.** +0.014, Holm 0.11.
- **Exploratory findings (P1 draft, paper/p1):**
  - a "redraw" protocol (the covered random draws), under which the TabPFN gap remains from 3C on and TabICL stays ahead of TabPFN v3.5;
  - 28.5% of decisive pairwise comparisons flip winner between the protocols;
  - the AUC convention (seen-class columns) drops the draws that miss a class.

Study 12 tests whether the confirmatory Part (a) results replicate outside DEV, on T1′, with new seeds. It also turns those exploratory findings into pre-declared secondary analyses.

**Seen before freezing (disclosed).** The Study 9 T1′ report (reports/STUDY9_T1PRIME_REPORT.md, reports/STUDY9_T1PRIME_OUTPUT.md) has been read. It uses the same 52 tables and the same four FMs, with a nested design, a stop set and different seeds (namespace 12). It reports that stratified sampling at 1C beats random in 45/52 tables for TabICL (≈ 2× label equivalence), together with ρ_S(m) for m up to 10. So the direction of contrast (i), at least at 1C for TabICL and TabPFN v2, is not blind on T1′. The (ii) contrasts (TabICL vs TabPFN v3.5) were not part of Study 9's confirmatory output. The Study 12 runs are new fits with new seeds, a different sampling design (non-nested, no stop set) and the Study 7 strategy definitions. No Study 12 outcome exists at freezing.

## 1. Data
- **T1′.** The 52 tables of `data/pool_t1prime.yaml` (Study 8 Amendment 2, A2.5; integrity screens A2.7 applied; `drop_columns` is empty for every member).
  - Composition: 37 binary and 15 multiclass tables (C = 3 to 10); n from 106 to 5000.
  - The HELD-OUT signature and row-overlap checks passed (reports/STUDY9_T1PRIME_REPORT.md §0.1).
- **All 52 tables are run; no subset.** The timing probe and Study 9's timings give ≈ 20,800 fits (+ redraw refits, Sec. 2.3) at ≈ 12 s mean. That is ≈ 70–80 worker-hours, about 10–12 h at 7 workers, which is feasible. A pre-declared subset is therefore not needed.
  - If a job fails, it is retried once.
  - A table that still fails for some FM is reported, and it is dropped from the contrasts that need that FM (complete-case per contrast). The count is reported.
- **Stratified feasibility.** Every class of every T1′ table has ≥ 14 rows, so stratified sampling is feasible at every m ∈ {1, 3, 5, 10} on every table. Every (dataset, mult) cell exists.
- **Not in scope.** No OpenML multiclass extension ("T2") is part of this freeze. Any extension would be a separate dated addendum, committed before any run on the new tables.

## 2. Design (Study 7 Part (a), with the deviations of Sec. 2.5)

### 2.1 Row permutation and evaluation panel
- **Permutation.** Per dataset, a fixed permutation π_d = `np.random.default_rng([base_seed, seed_did(did), 33]).permutation(n)` (namespace 33; `seed_did` = Study 9's map of negative PMLB/UCI IDs) is applied to X and y before anything else. Stored row indices are mapped back to the original order.
  - Random and stratified sampling are order-invariant in distribution. The permutation only makes the panel below a uniform random subset, and it neutralises the class-sorted storage that the A2.7 S1 screen flags.
- **Labeled set L.** Drawn from **all n rows**, as in Study 7.
- **Evaluation panel.** E = the first e = min(n, 2000) permuted rows. **Metrics are computed on E ∖ L.**
  - For n ≤ 2000 (39 of 52 tables), E ∖ L is exactly Study 7's complement.
  - For larger n, it is a uniform random subsample of the complement. The cap is for compute, as in Study 9.

### 2.2 Strategies, learners, budgets, seeds
- **Strategies.** `stratified`: m per class, drawn with `data.splits.draw_split`, 5 reps. `random`: B = m·C rows uniform without replacement (`selection.strategies.select_random`), 20 reps. Both are called through `runner.study4.select`, unchanged.
- **Learners.**
  - FM1 TabPFN v2, FM2 TabICL, FM3 Kumo-Tabular and FM5 TabPFN v3.5, using the Study 4 wrappers (`runner.study4._predictor`); TabPFN v3.5 uses the locally cached gated weights (licence accepted, research only).
  - Missing classes: the FM is fitted on the seen classes, and unseen classes get probability 0. With one seen class, every row gets probability 1 for it (`study9.predict_on`, read-only).
- **Budgets.** m ∈ {1, 3, 5, 10}, B = m·C.
- **Seeds (namespace 32).** `SeedSequence([base_seed, seed_did(did), m, rep, study4.ORDER.index(strategy), 32])`. The same seed drives the selection and the FM. Selections are identical across learners and cached per dataset.
- **Balance option, post hoc.** `fm1_bal` and `fm5_bal` are not refitted. They are the post-hoc transform p(c|x)/π̂_c, renormalised (`analysis.study7.prior_correct`), applied to the stored fm1 and fm5 probabilities. Study 7 Part (b) showed that this transform equals `balance_probabilities=True` to max |Δp| = 1.5e-7 over 4,000 fits.

### 2.3 Redraw protocol (pre-declared secondary arm)
- **Definition.** Draw B rows uniformly; if some class is missing, draw again, until every class is present. This is the random protocol conditioned on full coverage.
- **Implementation.**
  - Rep r, attempt 0, is the random draw of rep r. If it covers every class, the redraw run of rep r **is** that random run: same rows, same seed, same fit, no refit.
  - Otherwise, attempts a = 1, 2, … use `SeedSequence([base_seed, seed_did(did), m, r, a, study4.ORDER.index("random"), 32, 1])` for the draw and the FM, until a draw covers every class (at most 100,000 attempts; otherwise the cell is reported as infeasible).
  - The redraw arm has 20 reps per (dataset, m). It is paired with the random arm in the shared reps.

### 2.4 Stored per fit, and metrics
- **Stored:** labeled rows (original indices), query rows (E ∖ L, original indices), the float32 probability matrix over all C classes, coverage, seen classes, seconds, and the metrics below.
- **Metrics on E ∖ L** (Study 7 definitions, via `analysis.study7.metric_set`):
  - macro-F1 over all C classes, zero_division = 0 (**primary**);
  - accuracy;
  - balanced accuracy;
  - log-loss (clip 1e-6, renormalised);
  - AUC by Study 7's convention: macro one-vs-rest over the seen-class columns, NaN with one seen class;
  - the seen-classes-only versions (`seen_*`), as in Study 7.
- **New: `auc_all`, a realistic AUC convention.** Macro one-vs-rest over **all classes present in E ∖ L**, using the stored matrix, in which every unseen class's column is the fixed score 0.
  - A class absent from the context therefore gets a constant score, so its AUC is 0.5.
  - With one seen class, every column is constant and auc_all = 0.5.
  - No draw is dropped.

### 2.5 Deviations from Study 7, all declared here
1. The panel cap (2000) for n > 2000 (Sec. 2.1), and the row permutation.
2. Label-free selectors (kmedoids, kmeans_central, typiclust, lucos) are not run. They fed only Study 7's exploratory C2.
3. Parts (b) and (c) are not run. Part (b) is replaced by the exact post-hoc transform (Sec. 2.2), and Part (c) is out of scope.
4. Seeds: namespace 32, with `seed_did`.
5. One-seen-class contexts are handled as in Study 9: a constant prediction, with no fit. In Study 7 the same case was also a constant prediction (`predict_with_budget`), so this is not a change in outcome, only in code path.

## 3. Hypotheses, contrasts and decision rules

**Unit.** The dataset: the mean over reps per (dataset, learner, strategy, m), then the mean over the m values stated. n = 52 (complete cases).

### 3.1 Confirmatory family (6 contrasts, one Holm family, macro-F1)
| ID | Contrast | Per-dataset value | Test | Prediction (direction) |
|---|---|---|---|---|
| (i)-fm1, fm2, fm3, fm5 | stratified − random, per FM | mean over m ∈ {1, 3, 5, 10} | Wilcoxon signed-rank, two-sided | Δ > 0 for every FM |
| (ii-a) | fm2 − fm5 under random | mean over m ∈ {3, 5, 10} | Wilcoxon, two-sided | Δ > 0 (TabICL ahead) |
| (ii-c) | fm2 − fm5 under stratified | mean over m ∈ {3, 5, 10} | TOST, margin 0.01 (paired t); its p enters Holm | equivalent within ±0.01 |

**Decision rules.**
- **(i), per FM.**
  - *Replicated:* Holm p < 0.05 with mean Δ > 0.
  - *Reversed:* Holm p < 0.05 with mean Δ < 0.
  - *Not replicated:* otherwise.
  - **C1 replicates** if (i) is replicated for ≥ 3 of the 4 FMs, as in Study 7's rule.
- **(ii-a).** *Replicated:* Holm p < 0.05 and Δ > 0. *Reversed:* Holm p < 0.05 and Δ < 0. *Not replicated:* otherwise.
- **(ii-c).** Study 7 left this unresolved, so it is a fresh test.
  - *Tie established:* Holm-adjusted TOST p < 0.05.
  - *Difference:* not equivalent, and two-sided Wilcoxon p < 0.05 (unadjusted, reported with its sign).
  - *Unresolved:* otherwise.
- **Ranking claim.**
  - *Supported:* (ii-a) replicated and (ii-c) tie established.
  - *Half supported* (Study 7's DEV status): (ii-a) replicated and (ii-c) not established.
  - *Not supported:* (ii-a) not replicated.
- **Magnitude consistency (descriptive).** For each contrast, whether the DEV estimate lies inside the T1′ 95% bootstrap CI, and the ratio of the T1′ mean to the DEV mean. A replication with a much smaller effect is reported as such.

### 3.2 Robustness, for every confirmatory contrast (as Study 7 Sec. 4.2)
- mean, median, win-rate and bootstrap 95% CI of the mean (`analysis.report.bootstrap_ci`);
- TOST (margin 0.01);
- Bayesian signed-rank (baycomp, ROPE 0.01): P(+), P(ROPE), P(−);
- leave-one-dataset-out: range of the mean, largest p, sign stability;
- per-budget versions (m = 1, 3, 5, 10), unadjusted;
- the same contrasts in accuracy, balanced accuracy, log-loss, AUC (seen convention) and the seen-class metrics. These are secondary and unadjusted.

### 3.3 Pre-declared secondary analyses (no confirmatory weight; Holm within each block where tests are stated)
**S-1. Redraw protocol.**
- (a) Stratified − redraw per FM, macro-F1, mean over m ∈ {3, 5, 10}. At m = 1 a covered draw has exactly one label per class, so redraw and stratified coincide in distribution; m = 1 is reported only as a check.
  - Wilcoxon two-sided; Holm over the 4 FMs.
  - *Prediction:* Δ > 0 for TabPFN v2 and v3.5, the prior-sensitive FMs (the balance mechanism). No direction is predicted for TabICL and Kumo.
- (b) fm2 − fm5 under redraw, mean over m ∈ {3, 5, 10}, Wilcoxon two-sided.
  - *Prediction:* Δ > 0, i.e. the reversal is not explained by coverage alone.
- Also reported: the share of random draws that cover every class, per m, and the share of the (i) gap that redraw closes, (strat − random) − (strat − redraw), per FM and m.

**S-2. AUC with the realistic convention (`auc_all`).**
- Stratified − random per FM:
  - at m = 1 (Wilcoxon two-sided, Holm over 4 FMs);
  - and as the mean over m ∈ {1, 3, 5, 10} (reported).
- *Prediction:* Δ > 0 at m = 1 for every FM, because missed draws now count. The seen-class AUC is reported beside it, to show how much the convention changes the gap.

**S-3. Ranking flip rate.**
- Learners: {fm1, fm2, fm3, fm5, fm1_bal, fm5_bal}, as in Study 7 (and the 4-FM subset).
- Per (dataset, m): a pairwise comparison is decisive if |Δ| > 0.01 under both protocols; it flips if the winner differs. Kendall τ of the mean-macro-F1 rankings is reported per m. Accuracy is reported too.
- **Estimand.** Flip rate = flips / decisive pairs, with a dataset-cluster bootstrap 95% CI (2000 resamples).
- **Noise baseline.** The same flip rate between two independent random-protocol halves (random reps 0–9 vs 10–19; both "protocols" are random).
- *Prediction:* the protocol flip rate exceeds the noise baseline. Decided by the bootstrap CI of the difference (protocol − baseline, paired by dataset) excluding 0.
- DEV value for reference: 28.5%, with no baseline.

**S-4. Binary vs multiclass.**
- Every confirmatory contrast, recomputed within binary (C = 2, n = 37) and within multiclass (C ≥ 3, n = 15) tables: mean, CI and Wilcoxon p, unadjusted.
- The difference between the groups is tested by a Mann–Whitney U on the per-dataset Δ, two-sided.
- *Prediction (coverage theory, reports/THEORY_PROTOCOL_BIAS.md):* the (i) gap at m = 1 is larger for multiclass tables. No direction is predicted for (ii).

### 3.4 Exploratory
Anything else is labelled exploratory in the report.

## 4. Power
At n = 52, the two-sided Wilcoxon has ≈ 0.80 power at α = 0.05/6 for a standardised mean effect d ≈ 0.5. The DEV effects of (i) and (ii-a) had d ≈ 1.0–1.9.

For (ii-c), with a true Δ = 0 and SD ≈ 0.022 (DEV), TOST at margin 0.01 has power ≈ 0.89 at α = 0.05 with n = 52.

## 5. Run plan
- Order by priority: fm2 and fm5 first (they serve every contrast of (ii)), then fm1, then fm3.
- At most 7 workers, one torch/BLAS thread each, `nice -n 10`, in the background, logged to `logs/study12_*.log`.
- No outcome is analysed before every job has finished or definitively failed (Sec. 1).
- Unit tests on synthetic data cover:
  - the permutation and the panel;
  - redraw coverage and its reuse of covered random draws;
  - `auc_all`;
  - the flip rate;
  - the equality of the panel metrics with `complement_metrics` when E covers all rows.

## 6. Threats
- **Partly not blind.** Study 9's T1′ results (Sec. 0) were seen. For (i), this is a replication with a known direction at 1C, on new draws.
- **Panel cap.** For n > 2000, a subsampled complement adds evaluation noise but no bias.
- **T1′ composition.** T1′ is mostly binary (37/52), and the multiclass strata are thin. S-4 makes this explicit.
- **Data sources.** PMLB and UCI tables; some Tier 3 tables have integer-coded categoricals.
- **Learners.** The post-hoc fm*_bal transform is exact for TabPFN by Study 7's check, but it is not a native refit here.
- **Shared machine.** Timing does not affect results.

## 7. Addenda

### Addendum 1 (2026-10-27, during the run, before any outcome is computed): integrity screen and sensitivity analysis
**Trigger.** The authors decided that new confirmatory sets not repeat the DEV problem with table 694 (an identifier column leaking the class), and that an integrity screen be pre-declared.

**What had been seen when this was written.** Only job-completion lines in `logs/study12_run.log` and file names in `results/study12/`. No metric was read or computed.

**T1′ stays unchanged.** It was already screened before Study 9, by Study 8 A2.7 (`src/simaudit/runner/screen_integrity.py` → `data/pool_ext_integrity_audit.csv`):
- 18 candidates were excluded before the draw;
- every T1′ member has decision `keep`, and no leaky column was dropped (no single column reaches 1-NN balanced accuracy 0.90).

The confirmatory analysis (Sec. 3.1) therefore stays on all 52 tables. For the flagged tables, a sensitivity analysis is pre-declared.

**Screen result for the 52 members** (read from the audit CSV; the 1-NN check below was computed on the features only, before any outcome):
- **Identifier / counter columns.**
  - The screen flags "id-like" columns in 7 tables.
  - Of these, the column is **monotone in the stored row index** (a counter) in 4. The single-column leave-one-out 1-NN balanced accuracy on the class, in stored order, is:
    - pmlb:cloud col 0: 0.87. This is a row ID, and the table is class-sorted (κ = 0.80), so it is the 694 pattern;
    - pmlb:haberman col 0: 0.88;
    - uci:519 col 11 (follow-up time): 0.75;
    - pmlb:backache col 0: 0.44.
  - The other 3 (pmlb:tokyo1 col 15, pmlb:churn col 3, uci:850 cols 0 and 4) are near-unique continuous measurements, not monotone. They are reported only.
- **Conflicting duplicate rows** (any group): pmlb:haberman (6 groups), pmlb:breast_cancer (6), pmlb:spambase (3), pmlb:hypothyroid (1), pmlb:splice (1), uci:563 (14). All are below the 5% exclusion share of A2.7.
- **Time series relabelled as classification.** No screen exists for this, so the 52 names were inspected. None is a time series cut into labelled windows.
  - pmlb:Hill_Valley_with_noise and pmlb:Hill_Valley_without_noise classify the shape of a 100-point series. Each row is one whole series, so they are kept and not flagged.

**Pre-declared sensitivity analysis (no confirmatory weight).**
- **SENS-INT.** Every confirmatory contrast and secondary analysis S-1 to S-4, recomputed without the flagged tables, F = {cloud, haberman, uci:519, backache, breast_cancer, spambase, hypothyroid, splice, uci:563} (n = 43), with Holm recomputed within the reduced family.
- **SENS-ID.** The same, excluding only the four counter-column tables (n = 48).
- A confirmatory verdict that changes under SENS-INT is reported as fragile.

**New sets.** Any OpenML extension (e.g. a multiclass "T2") would need this screen as an exclusion rule. It would also need the exclusion of DEV, T1′ and the HELD-OUT IDs (`research_notes/heldout_openml_ids_exclusion.txt`, IDs only), with their name and version families. That extension is not part of Study 12 and would need its own dated addendum before any run.

### Amendment 2 (2026-10-27, during the run, before any outcome is computed): infeasible redraws
- **Problem.** The runner raised an error when a redraw rep found no covering draw within 100,000 attempts. The first case was pmlb:yeast at 1C: C = 9 and the smallest class has 20 of 1479 rows. The error aborted the whole (dataset, learner) job, i.e. the stratified and random runs as well.
- **Fix.** Implementation only; Sec. 2.3 already says such a cell "is reported as infeasible".
  - Such a rep is now recorded in the selection cache with attempt = −1 and is not fitted.
  - The stratified and random arms of the table are unaffected.
  - The redraw contrasts use the feasible reps; a (dataset, m) cell without any feasible rep drops out of S-1 for that m.
  - The report states the number of infeasible reps per table and m.
  - Feasible selections are byte-identical to the old code's.
- **What had been seen.** Job-completion lines and the failure line in the log. No metric had been read.

## Erratum 1 (2026-10-28, before any Study 12 metric was read)
- **T1′ composition.** Sec. 2 (line 40), S-4 (line 155) and the limitations (line 181) state "37 binary and 15 multiclass". The correct composition is **35 binary and 17 multiclass**. Every data source agrees on it: data/pool_t1prime.yaml, the pool_ext C fields, the n_classes stored by Study 9 and Study 12, and the loader labels. The error was hand-typed. The S-4 code (analysis/study12.split_analysis) splits by the stored n_classes, so the analysis already uses 35/17, and nothing in the analysis changes. The hard-coded string in reports/study12_report.py is corrected with the report.
- **What had been seen.** Only n_classes per table, read from the pool files and the stored outputs. No metric was read.
