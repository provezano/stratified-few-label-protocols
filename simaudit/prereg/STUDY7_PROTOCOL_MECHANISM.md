# Study 7: the stratified protocol, its mechanism, and label-free selectors (DEV)

Status: FROZEN 2026-10-21, committed before any Study 7 run. The only computation done before freezing was a timing probe (one fit per FM on dataset 46562 with arbitrary rows, not stored, not scored), used to plan the run order. DEV only, the 20 datasets of pilot I2; `sealed_heldout/` is not touched.

## 0. Why
The planned evaluation paper makes two claims that reviewers will attack:
- **C1.** The stratified "m labels per class" protocol is optimistic, through two mechanisms: class coverage (mechanism 1) and class balance of the context (mechanism 2), and it changes rankings between learners.
- **C2.** Label-free selectors beat realistic random selection.

The evidence so far is from Study 4 (reports/STUDY4_STAGE_B.md, namespace 5): the analytic coverage model (reports/COVERAGE_MODEL.md) explains the stratified gap at B = C but only a third of it from 3C on; mechanism 2 is an untested hypothesis. The TabICL vs TabPFN v3.5 reversal (reports/PROTOCOL_RANK_REVERSAL.txt) was found exploratorily. Study 7 re-runs the core with new seeds and stored probabilities, adds TabPFN's documented balance option, and tests mechanism 2 causally with prescribed context proportions.

Code: `src/simaudit/runner/study7.py` (runner), `src/simaudit/analysis/study7.py` (analysis), CLI `simaudit study7`, report generator `reports/study7_report.py`, report `reports/STUDY7_REPORT.md`. Outputs: `results/study7/{part}/{did}_{learner}.parquet`, one file per (dataset, learner, part), written atomically and skipped when present (resumable).

## 1. Part (a): Study 4 core re-run with stored probabilities
- **Strategies.** `stratified` (m = mult per class; uses y), `random` (**20 reps**), `kmedoids`, `kmeans_central`, `typiclust`, `lucos` (**5 reps** each). Selection functions are those of `selection/strategies.py`, `selection/lucos.py` and `runner/study4.select`, unchanged.
- **Learners.** FM1 (TabPFN v2), FM2 (TabICL), FM3 (Kumo-Tabular), FM5 (TabPFN v3.5), with the Study 4 wrappers. Missing classes: fitted on the seen classes, unseen classes get probability 0 (`predict_with_budget`).
- **Budgets.** B = mult · C, mult ∈ {1, 3, 5, 10}.
- **Seeds (namespace 10).** `SeedSequence([base_seed, did, mult, rep, study4.ORDER.index(strategy), 10])`; the same seed drives the selection and the FM. Selections are therefore identical across learners and are cached once per dataset (`results/study7/selections/{did}.parquet`).
- **Stored per selection.** Labeled indices, the full probability matrix (all rows × C, float32), class coverage, the seen classes, and the metrics below.
- **Metrics, on the complement of the labeled rows.**
  - macro-F1 over all C classes (zero_division = 0) **primary**; accuracy; balanced accuracy (mean recall over the classes present in the complement);
  - log-loss with probabilities clipped to [1e-6, 1] and renormalised (a missed class then costs −log 1e-6 per row);
  - AUC: macro one-vs-rest **over the seen-class columns**, i.e. the mean over seen classes c that have both positives and negatives in the complement of AUC(column c, 1[y = c]); with a single seen class the AUC is undefined (stored as NaN).
  - **Seen-classes-only sensitivity**: the same metrics restricted to complement rows whose true class was seen, with macro-F1 and balanced accuracy over the seen classes only, and log-loss on the seen-class columns.

## 2. Part (b): TabPFN's balance option
- **The option.** `TabPFNClassifier(balance_probabilities=True)` (tabpfn 9.2.0, `classifier.py`; the docstring recommends it for imbalance-insensitive metrics). Source reading (`logits_to_probabilities`, `_apply_balancing`, `utils.balance_probas_by_class_counts`): after the softmax and the ensemble average (and after the down-sampling correction, inactive for contexts this small), the probabilities are divided by the context class frequencies π̂_c = n_c / n (`class_counts_` from `fit`) and renormalised: p_bal(c | x) ∝ p(c | x) / π̂_c. It is a post-hoc prior shift to a uniform class prior; it does not change the forward pass.
- **Runs.** FM1 and FM5 with the option ON, as learners `fm1_bal` and `fm5_bal`, under `random` (20 reps) and `stratified` (5 reps), mult ∈ {1, 3, 5, 10}, the same seeds and therefore the same selections as Part (a).
- **Check (descriptive).** Agreement of `fm1_bal` with the post-hoc transform of Part (a)'s `fm1` probabilities (max absolute difference), which confirms the semantics above empirically.
- **Pre-registered question.** Does the TabICL vs TabPFN v3.5 reversal survive when TabPFN v3.5 balances its probabilities? Contrast (ii-b) below.

## 3. Part (c): controlled dose-response for context balance (causal test of mechanism 2)
- **Cells.** Each DEV dataset × each FM (FM1, FM2, FM3, FM5) × mult ∈ {3, 5, 10}, B = mult · C. Every context has **full coverage** and **prescribed** class counts.
- **Levels.** s ∈ {0.5, 0.6, 0.7, 0.8, 0.9}, the share of the upweighted class u.
  - n_u = round-half-up(s · B), clipped to [1, B − (C − 1)].
  - The remaining R = B − n_u rows are split equally over the other C − 1 classes (⌊R / (C − 1)⌋ each); the R mod (C − 1) extra rows go to classes drawn uniformly at random with the cell seed. Every class gets ≥ 1 row.
  - Binary: u has share s, the other class 1 − s. Multiclass: u has share s, the rest equal.
  - Rounding can map two levels to the same counts at small B (e.g. B = 6); they are still drawn independently. The realised share n_u / B is stored.
- **Arms.** u = the natural majority class (largest count; ties → lowest index) **and** u = the natural minority class (smallest count; ties → highest index).
- **Rows.** Within each class, rows are drawn uniformly at random without replacement; **10 reps** per (dataset, mult, arm, level).
- **Feasibility.** A cell is infeasible, skipped and logged, if a class would need more than n_c − 1 rows (each class keeps ≥ 1 row in the complement).
- **Anchor (multiclass only, descriptive).** s = 1/C, i.e. mult rows per class: the stratified design with independent draws.
- **Seeds (namespace 10).** `SeedSequence([base_seed, did, mult, rep, round(100 · s), *counts, 10])` where `counts` is the per-class count vector. Identical designs share a seed (binary s = 0.5 is the same design in both arms and is fitted once); different levels never do.
- **Outcomes, on the complement.** macro-F1 (primary), accuracy and balanced accuracy, plus the same three after the post-hoc prior correction p / π̂ of Sec. 2 (π̂ = the prescribed context shares), for every FM. For TabPFN this is exactly the balance option; for TabICL and Kumo it is the same Bayes prior correction applied from outside.
- **Pre-registered prediction (mechanism 2).** Macro-F1 declines monotonically in s, in the **majority-upweighted and the minority-upweighted arm alike**. The minority arm is what separates mechanism 2 from "more minority labels help": there, larger s means more labels of the minority class, yet macro-F1 is predicted to fall.
- **Competing alternative.** The effect is only a prior shift: the FM's predictions follow the context's class prior, and a balance option removes it. Under this alternative, the decline vanishes after the p / π̂ correction. Decision rule: mechanism 2 is "beyond a prior shift" for an FM if the corrected trend is still one-sided significant (stratified JT, p < 0.05, both arms) and its mean drop from s = 0.5 to the largest feasible level is at least half of the uncorrected drop; it is "a prior shift" if the corrected drop is TOST-equivalent to 0 (margin 0.01); otherwise "unresolved".

## 4. Analysis plan
**Unit.** The dataset (mean over repetitions), n = 20, unless stated otherwise.

### 4.1 Confirmatory contrasts (one family, Holm across all 15)
| ID | Contrast | Per-dataset value | Test | Prediction |
|---|---|---|---|---|
| (i)-fm1, fm2, fm3, fm5 | Δ(stratified − random) per FM, macro-F1, Part (a) | mean over mult ∈ {1, 3, 5, 10} | Wilcoxon signed-rank, two-sided | Δ > 0 |
| (ii-a) | fm2 − fm5 under random, macro-F1, Part (a) | mean over mult ∈ {3, 5, 10} | Wilcoxon, two-sided | Δ > 0 (reversal replicates) |
| (ii-b) | fm2 − fm5_bal under random, macro-F1, Parts (a, b) | mean over mult ∈ {3, 5, 10} | Wilcoxon, two-sided | open: the question of Part (b) |
| (ii-c) | fm2 − fm5 under stratified, macro-F1, Part (a) | mean over mult ∈ {3, 5, 10} | TOST, margin 0.01 (paired t) | equivalent (the "tie" side of the reversal) |
| (iii)-FM × arm (8) | dose-response in s, macro-F1, Part (c) | per-rep values | stratified Jonckheere–Terpstra, one-sided (decreasing) | decline in both arms |

- mult ∈ {3, 5, 10} for (ii) is where the exploratory reversal appeared; it is fixed here and tested on new seeds.
- **Stratified JT.** Strata are (dataset, mult); within a stratum the groups are the feasible levels in increasing s, with the 10 reps as independent observations. The JT statistics, their null means and their tie-corrected variances (Hollander & Wolfe) are summed over strata with ≥ 2 feasible levels; z = (J − E) / √V, one-sided normal p for a decrease.
- **(ii-b) reading.** Holm-significant Δ > 0: "the reversal survives balancing". TOST-equivalent within 0.01: "balancing removes it". Otherwise "unresolved".
- **Overall reading.** C1 (optimism) is supported if (i) holds for ≥ 3 of 4 FMs; mechanism 2 is supported if (iii) holds in both arms for ≥ 3 of 4 FMs; the ranking claim is supported if (ii-a) is significant and (ii-c) is equivalent.

### 4.2 Robustness reported for every confirmatory contrast
- Mean, **median** and **win-rate** (share of datasets with Δ > 0) of the per-dataset values; bootstrap 95% CI of the mean.
- **TOST** (margin 0.01) wherever a "tie" or "no difference" is claimed, in confirmatory and exploratory text alike.
- **Bayesian signed-rank** (baycomp `SignedRankTest`, ROPE 0.01): P(left), P(ROPE), P(right).
- **Leave-one-dataset-out**: the contrast recomputed 20 times; the range of the mean, the largest p, and whether the sign is stable. For JT, the strata of one dataset are dropped at a time.
- For (iii): the per-dataset OLS slope of macro-F1 on s (median over datasets) and the mean drop from s = 0.5 to the largest feasible level; Page's L test on dataset means over the datasets where all five levels are feasible (exploratory).

### 4.3 Exploratory (no confirmatory weight)
- **C2.** Each label-free selector (kmedoids, kmeans_central, typiclust, lucos) − random, per FM × mult, macro-F1 and accuracy, Wilcoxon with Holm within FM × mult, TOST, Bayesian signed-rank.
- Per-budget versions of (i) and (ii); the same contrasts in accuracy, balanced accuracy, AUC, log-loss and the seen-classes-only metrics.
- **Rank reversal.** Learners {fm1, fm2, fm3, fm5, fm1_bal, fm5_bal}; per mult, Kendall τ between the mean macro-F1 rankings under stratified and under random; and the share of decisive pairwise comparisons (|Δ| > 0.01 under both protocols, per dataset × mult) whose winner flips. Same for accuracy.
- **Critical-difference diagrams** (Friedman + Nemenyi, α = 0.05, per-dataset means over mult): learners under each protocol; strategies within each FM.
- Dose-response in accuracy and balanced accuracy; the multiclass anchor.

## 5. Threats
- The evaluation set (complement) differs between selections; the seen-classes-only metrics and the per-row probabilities allow later rescoring on common rows.
- At high s and large B, multiclass cells are often infeasible (a class has too few rows); the JT uses the feasible levels only, and the number of feasible strata is reported.
- The p / π̂ correction for TabICL and Kumo is ours, not a vendor option.
- Datasets at chance (827, 931) dilute differences; they are kept, as in Study 4.
- The machine is shared (≤ 3 workers, one BLAS/torch thread each); timing does not affect results.

## 6. Addendum 1 (2026-10-21, during the run, before any analysis): data-integrity sensitivity analyses
**Trigger.** Findings of a separate data audit (2026-10-21):
- 11 of the 20 DEV datasets are stored sorted by class, so any selection or index tie-break that depends on row order can exploit the order;
- 694 (diggle_table_a2) has an ID/counter column that encodes row order, hence the class, in the features;
- 40496 (LED-display-domain-7digit), 860 (vinnie), 1063 (kc2) and 46336 (hayes-roth_clean) contain duplicate rows with conflicting labels.

**What had been seen when this was written.** Job-completion counts in the run log, and one smoke test of the report generator on the two finished Part (c) TabICL files (827 and 931, both at chance level), which showed their dose-response means. No Part (a) or Part (b) output and no confirmatory contrast had been computed. The addendum is therefore prospective for everything except those two files.

**The frozen confirmatory analysis (Sec. 4.1) is unchanged.** The following are pre-declared sensitivity analyses, reported next to it and with no confirmatory weight:
- **S1.** Every confirmatory contrast (Holm family recomputed within the sensitivity set) excluding 694 (n = 19).
- **S2.** The same, excluding 40496, 860, 1063 and 46336 (n = 16).
- **S3 (row order).** The four strategies that do not draw uniformly at random (kmedoids, kmeans_central, typiclust, lucos) are re-run on rows permuted by a fixed per-dataset permutation (`np.random.default_rng([base_seed, did, 10, 99]).permutation(n)`), with the same selection seeds as Part (a); the selected rows are mapped back to the original indices and scored exactly as in Part (a). Learner: **fm2 only** (cost), stored as Part `p` (`results/study7/p/{did}_fm2.parquet`). Reported: per strategy × mult, Δ(permuted − original) and Δ(permuted − random) in macro-F1 and accuracy (Wilcoxon, TOST margin 0.01). Random, stratified and the Part (c) designs draw uniformly within the whole data or within classes, so row order cannot change their distribution; they are not re-run.
- **Threats.** The report lists these datasets and these issues under threats.

### 6.1 Amendment to Addendum 1 (2026-10-21, before any analysis): verified dataset list
The verified audit (research_notes/Systematic discovery 2026-10/22_row_order_leakage_fix.md §7). It supersedes the names above:
- **Sorted by class (11):** 46562, 46875, 45615, 841, 42544, 337, 1063, 782, 40496, 694, 41511.
- **Sorted by a feature column (reported only):** 860, 841, 782, 801.
- **ID/counter column:** 694, column index 1 of the encoded X (1-NN on it recovers the class; our check: 5-fold 1-NN accuracy 0.89 on that column alone).
- **Conflicting duplicates:** 40496 (37 groups), 860 (23), 46336 (9), 1063 (6), 1071 (3), 801 (1).
- **Clean:** 827, 931, 46544, 1510, 47155.

Changes:
- **S2** excludes the six datasets with conflicting duplicates: 40496, 860, 46336, 1063, 1071, 801 (n = 14).
- **S4 (new).** 694 is re-run with column 1 dropped, every part and every learner, same seeds (`simaudit study7 --datasets 694 --drop 694:1 --out-name study7_s4`; outputs in `results/study7_s4/`). Every confirmatory contrast is recomputed with 694's rows replaced by the column-dropped run (n = 20). S1 (excluding 694) stays as the fallback if S4 cannot finish.
- Still seen before this note: only job-completion counts and the smoke test described above. No confirmatory or Part (a)/(b) result had been computed.
