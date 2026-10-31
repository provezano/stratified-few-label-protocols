# P1 exploratory B: dose-response arms F, S and N (plan)

**EXPLORATORY: DEV only, not pre-registered.** Written and committed on 2026-10-27, before any fit of these arms. It answers the internal reviews R3-W3, R3-Q2 and R2-W2. Nothing here touches `sealed_heldout/`, `prereg/`, `reports/`, `results/` or `src/`. Code: `paper/p1/exploratory/b_dose/`. Outputs: `paper/p1/exploratory/b_dose/out/`.

## 1. How the original Part (c) chose the rows (answer to R2-Q2)

From `prereg/STUDY7_PROTOCOL_MECHANISM.md` Sec. 3 and `src/simaudit/runner/study7.py` (`dose_counts`, `dose_cells`, `draw_context`, `_dose_cells`):

- **Budget fixed per cell.** B = mult · C, mult ∈ {3, 5, 10}. At share s the upweighted class u gets n_u = round-half-up(s · B), clipped to [1, B − (C − 1)]. The other C − 1 classes split the remaining B − n_u rows equally. The leftover rows go to other classes drawn at random with a cell RNG.
- **Arms.** u is the natural majority class (ties: lowest index) or the natural minority class (ties: highest index).
- **Rows.** Within each class, rows are drawn uniformly at random without replacement: `rng.choice(flatnonzero(y == c), n_c, replace=False)`. The RNG is `default_rng(dose_seed)`, with dose_seed = SeedSequence([base_seed, did, mult, rep, round(100 s), *counts, 10]). The draw uses only class membership, never X or row order.
- **No nesting.** Every level (and every rep) has its own seed, so the contexts at different shares are independent draws. The only shared context is an identical design: binary s = 0.5, which is the same in both arms and is fitted once.
- **The FM seed is the design seed.** The FM's random state therefore also changes from level to level.
- **The evaluation set changes too.** It is the complement of the context, so it differs between levels.
- **Feasibility.** A cell is skipped if a class would need more than n_c − 1 rows.

Consequence (R3-W3): at fixed B, moving s from 0.5 to 0.9 changes three things at once:
- (i) the context class balance;
- (ii) the number of labels of the non-upweighted classes, which falls from about B/2 to about B/10 in binary. In the majority arm these are the natural-minority labels;
- (iii) the identity of the sampled rows (and of the FM seed and the evaluation set).

The original design cannot separate (i) from (ii) and (iii).

## 2. Estimand

For each FM and each side (u = natural majority, u = natural minority), the change in complement macro-F1 per unit of context share s of u under each of the following conditions:

- **F:** the per-class count k of the other classes is held fixed;
- **S:** the share is held fixed and only k varies, along the counts that the original design implies;
- **N:** the counts of F, with sample identity held fixed by nesting.

These are compared with the original fixed-B slope **O**, taken from the stored Part (c) results (read-only).

## 3. Arms

Notation: C classes, B₀ = mult · C.
- k(s) = max(1, ⌊(B₀ − n_u^orig(s)) / (C − 1)⌋) is the per-class count of the non-upweighted classes in the original design at share s.
- k₀ = k(0.5). In binary k₀ = mult.
- n_u(s, k) = max(1, round-half-up(s / (1 − s) · (C − 1) · k)). With every other class at k rows, u then has a share of about s. The realised share is stored.

| Arm | Counts at level s | Rows | What varies along the dose axis |
|---|---|---|---|
| **F** (fixed minority count) | other classes k₀ each, u gets n_u(s, k₀); s ∈ {0.5, …, 0.9} | independent uniform draw within each class per level (s = 0.5 is the base context) | share ↑ and B ↑; the non-upweighted classes keep k₀ labels; u gains labels |
| **S** (fixed share, varying count) | share fixed at 0.5: other classes k(s′) each, u gets n_u(0.5, k(s′)); dose index s′ ∈ {0.5, …, 0.9} | independent uniform draw within each class per distinct count (k = k₀ is the base context) | only the count k, along exactly the path that the original design gives the non-upweighted classes |
| **N** (nested) | the counts of F | one random permutation per class per (dataset, mult, rep); a context takes the first n_c rows of each permutation, so the context at a higher share contains every row of every lower-share context | share ↑ by adding u rows only; no resampling of the existing rows |

- In the majority arm, F holds the natural-minority count fixed and adds majority rows. In the minority arm, F holds the other classes fixed and adds natural-minority rows. A decline there means that more labels of one class lower macro-F1.
- S on the s′ axis is the count-only part of the original path, and F is the share-only part at fixed count. The original slope O is compared with both. Additivity, O ≈ F + S, is descriptive only.
- **Common random numbers.** The FM seed is one per (dataset, mult, rep) and is shared by every cell of every arm, unlike Part (c). Identical contexts are fitted once and shared between cells, for example the base context, binary S cells across sides, and repeated k in S.
- **Feasibility.** As in Part (c): a cell is infeasible if a class needs more than n_c − 1 rows. Such cells are logged, not fitted. F and N need up to 9 · k₀ rows of u in binary, so many minority-arm cells at mult = 10 are infeasible. Feasible counts are reported.

## 4. Data, FMs, budgets, seeds

- **Tables.** The 20 DEV tables of pilot I2 (`data/pool_i2.yaml`), as in Part (c). 694 keeps its ID column, as in the Part (c) primary analysis. A run without 694 is a sensitivity.
- **FMs.** All four: FM2 (TabICL), FM1 (TabPFN v2), FM5 (TabPFN v3.5), FM3 (Kumo-Tabular), with the Study 4/7 wrappers (`study7.predictor`). Missing classes cannot occur, because every context has full coverage.
- **Budgets and reps.** mult ∈ {3, 5, 10}; **5 reps** per (table, mult), compared with Part (c)'s 10, for cost. The dry run counts 1,084 distinct contexts per rep over the 20 tables, so 5,420 fits per FM, 21,680 in all. At the Part (c) per-fit times this is about 55 CPU-hours.
- **Rule for the FM subset, fixed now.**
  - The FMs run in the order FM2 → FM1 → FM5 → FM3, which is cheapest first, as in the Study 7 plan.
  - The cap is 40 wall-clock hours from launch. Any FM that has not finished all 20 tables by then is stopped and **excluded entirely** from the analysis, never partially analysed. The summary reports it as not run.
- **Seeds.** Namespace **41** (unused: 4–6, 8, 10, 12 and 32 are taken); base_seed 20261020.
  - Permutations: `default_rng([base_seed, did, mult, rep, 41, 0])`, one per class in class order.
  - Independent draws: `SeedSequence([base_seed, did, mult, rep, round(100·level), *counts, 41, tag])`, with tag 1 for F and tag 2 for S (level 0.5 for S).
  - FM seed: `SeedSequence([base_seed, did, mult, rep, 41, 7])`.
- **Execution.** At most 3 worker processes, `nice -n 10`, one torch/BLAS thread each. The run is resumable through per-(FM, table) parquet files plus checkpoints, and logs to `b_dose/run.log`. TabPFN v3.5 reads the access token from the tabpfn cache, as in earlier runs. The licence is research-only.

## 5. Metrics

Every fit is scored on two evaluation sets:
- **own complement**, as in Part (c). This is the primary set;
- **common evaluation set**, the rows outside the union of all feasible contexts of that (table, mult, rep). The metric is NaN if a class is missing from it. This set removes the change in evaluation rows along the dose axis.

The metrics are:
- **macro-F1**, the primary metric, over all C classes with zero_division = 0;
- accuracy;
- balanced accuracy;
- each of the three after the p/π̂ prior correction with the context shares (`study7.prior_correct`; prefix pc_).

## 6. Analysis (all exploratory; no confirmatory weight)

- **Table-level slope (primary effect size).** For each FM × side × arm and each table:
  - per mult, average the metric over reps at each feasible level;
  - fit the OLS slope on the dose (s for F and N, s′ for S), requiring at least 2 feasible levels;
  - average the slope over the feasible mults.

  This gives one slope per table. The reference O is the same statistic computed from `results/study7/c` (all 10 reps, the same mults).
- **Summaries.**
  - Mean over tables with a percentile bootstrap 95% CI: 10,000 resamples of tables, `simaudit.analysis.report.bootstrap_ci`.
  - Median, the share of tables with a negative slope, and a two-sided Wilcoxon p.
  - Holm over the macro-F1 slope family (4 FMs × 2 sides × 3 arms), reported descriptively.
- **Paired contrasts per table.** β_F − β_O, β_S − β_O, β_N − β_F and β_O − (β_F + β_S). Each is reported with the same summaries.
- **Secondary.**
  - Drop m(0.5) − m(largest feasible level), with the same summaries.
  - Stratified Jonckheere–Terpstra (`simaudit.analysis.study7.jt_stratified`; strata = table × mult; groups = levels; observations = reps). For N, the reps are nested across levels, so the JT's independence assumption fails. For N we therefore also report the share of (table, mult, rep) with a negative within-rep slope.
  - All of the above for accuracy, BA, the pc_ metrics and the common-set metrics.
  - Sensitivities: without 694, and binary only.
- **Reading rules (fixed now).**
  - **Balance effect beyond the minority count** for an FM × side: θ_F < 0 with a CI that excludes 0.
  - **Count effect only**: θ_F has a CI that includes 0 and θ_S < 0 with a CI that excludes 0.
  - **Sample identity not needed**: θ_N < 0 with a CI that excludes 0.
  - **Prior shift**: the same three rules on the pc_ slopes show whether the decline at fixed count survives the p/π̂ correction. This bears only on the specific prior-shift model p/π̂ that we tested.

## 7. Outputs

- `b_dose/out/raw/{fm}_{did}.parquet`: one row per cell.
- `b_dose/out/slopes.csv` and `b_dose/out/summary.csv`.
- `b_dose/out/SUMMARY.md`.
- `b_dose/out/b_numbers.tex`: macros `\ExB…`, generated by `b_dose/analyze_dose_b.py`.

Paper text that uses these results marks them "(exploratory)".
