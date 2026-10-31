# Preregistration: dose arms F, S and N on a T1′ subset (P1, exploratory)

Written and committed on 2026-10-28, **before any T1′ fit of these arms**. Status: **exploratory**. The result supports only the appendix paragraph on the dose arms (`app:dose`). It has no confirmatory weight and does not touch `sealed_heldout/`, `prereg/`, `results/` or `src/`.

## 1. Question

On DEV (commit R019; `b_dose/out/SUMMARY.md`), with the count of the non-upweighted classes held fixed (arm F), the p/π̂ prior correction leaves no decline of macro-F1 in any of the 8 FM × side cells, while with the share fixed and the count falling (arm S) macro-F1 still declines after the correction in all 8 cells. The paper reads this as: the Part (c) residual after p/π̂ follows the count, not the share. The result is "DEV only". This run asks whether the same holds on tables outside DEV.

## 2. Tables (chosen without looking at any outcome)

- Frame: the 52 tables of `data/pool_t1prime.yaml` (the Study 12 registry). Only metadata are used: the key, n from the registry, and C and d from the cached table. No Study 12 result is read.
- Strata: binary (C = 2, 35 tables) and multiclass (C > 2, 17 tables).
- Rule (`b_dose/t1p_subset.py`): within each stratum, sort the keys, shuffle with `default_rng([20261020, 41, 1201])` (binary first, then multiclass, same generator), and take the first round(K · 35/52) binary and the first K − round(K · 35/52) multiclass tables. K is the largest value ≤ 16 whose estimated cost is ≤ 60 CPU-hours (Sec. 7). This gives **K = 16: 11 binary, 5 multiclass**.
- Subset (`b_dose/t1p_subset.csv`):
  - binary: pmlb:Hill_Valley_with_noise, pmlb:biomed, pmlb:bupa, pmlb:corral, pmlb:glass2, pmlb:ionosphere, pmlb:mofn_3_7_10, pmlb:molecular_biology_promoters, pmlb:saheart, uci:267, uci:519;
  - multiclass: pmlb:cars, pmlb:ecoli, pmlb:led24, pmlb:new_thyroid, uci:212.

## 3. Design (as DEV unless stated)

- **Arms, levels, budgets, reps:** arms F, S and N of `b_dose/design.py`, unchanged; levels s ∈ {0.5, 0.6, 0.7, 0.8, 0.9}; B₀ = mult · C with mult ∈ {3, 5, 10}; 5 reps; upweighted class = natural majority or natural minority. Feasibility as in DEV (a class may give at most n_c − 1 rows).
- **FMs:** TabICL (FM2), TabPFN v2 (FM1), TabPFN v3.5 (FM5), Kumo-Tabular (FM3), through the same wrappers (`runner.study7.predictor`), run in the order FM2 → FM1 → FM5 → FM3. TabPFN v3.5 uses the token already in the local tabpfn cache (research-only licence; weights are not shipped). Kumo-Tabular runs locally, exactly as in DEV.
- **Seeds:** the DEV scheme (namespace 41, base 20261020): per-class permutations, independent draws and one FM seed per (table, mult, rep) shared by all cells (common random numbers). For the negative PMLB/UCI ids, the table enters the seeds as `seed_did(did) = 2^40 + did` (`runner.study12.seed_did`).
- **Evaluation rows (T1′ detail):** DEV tables have n ≤ 1000 and DEV scored each fit on the full complement. T1′ tables reach n = 3200 in this subset, so each fit is scored on a fixed panel: the first min(n, 1000) rows of the Study 12 row permutation (namespace 33). Own set = panel minus the context; common set = panel minus the union of all feasible contexts of that (table, mult, rep). Contexts are drawn from the whole table, as in DEV. For n ≤ 1000 the panel is the whole table, so the code is the DEV code; a check on DEV table 782 (mult 3, rep 0) reproduced the stored DEV metrics exactly for all four FMs.
- **Metrics:** as DEV: macro-F1 (primary), accuracy, balanced accuracy, each also after p/π̂ (`study7.prior_correct`, context shares), on the own and the common set.
- **Device:** CPU only, as DEV. `nice -n 15`, one thread per process (OMP/MKL/OPENBLAS/VECLIB and torch = 1), at most 6 worker processes.

## 4. Stop rule

The run has a wall-clock cap of 30 hours from launch. An FM that has not finished all 16 tables by then is excluded entirely from the analysis (never partially analysed) and reported as not run. A table that fails for an FM excludes that FM in the same way, unless the failure is fixed and the job re-run before the analysis.

## 5. Primary estimand

For each FM × side and each table, the arm-F slope of p/π̂-corrected macro-F1 on the share (DEV analysis: per mult, rep-mean at each feasible level, OLS slope, averaged over feasible mults). The **residual** of a table is −0.4 × slope: the fitted change in corrected macro-F1 from s = 0.5 to s = 0.9, with the sign flipped so that a positive residual is a decline that survives the correction. The estimand is the mean residual over the 16 tables; CIs are percentile bootstraps of tables (10,000 resamples, seed 0).

## 6. Prediction and decision rule

**Primary prediction:** in each of the 8 FM × side cells of arm F, the residual lies within a ROPE of ±0.01 macro-F1.

Each cell gets one verdict:
- **equivalent:** the 90% CI lies inside [−0.01, +0.01] (TOST at α = 0.05 per bound);
- **decline survives (failure):** the 95% CI lies above +0.01;
- **over-correction:** the 95% CI lies below −0.01 (p/π̂ more than removes the decline);
- **inconclusive:** otherwise.

Overall decision:
- **PASS** if all 8 cells are equivalent;
- **FAIL** if at least one cell has a surviving decline. This would contradict the DEV reading that the residual follows the count and not the share;
- **NOT ESTABLISHED** otherwise (no failure, but equivalence not shown in every cell).

**Expected power, stated now.** Applied to the stored DEV slopes, this rule gives 0 equivalent, 0 surviving declines, 2 over-corrections (TabICL and TabPFN v3.5, majority side) and 6 inconclusive cells; the 95% CIs are about ±0.03 wide with 20 tables. With 16 tables a PASS is therefore unlikely even if the DEV picture holds. The DEV statement "residual ≈ 0" means "no decline survives", not equivalence at ±0.01.

**Agreement with DEV** (decided by this rule, fixed now): T1′ agrees with DEV if (a) no arm-F cell has a surviving decline, and (b) after p/π̂, arm S declines (95% CI of the mean slope below 0) in at least 6 of the 8 cells (DEV: 8 of 8). If (a) holds and (b) fails, the count part does not replicate; if (a) fails, the share part does not replicate.

**Secondary (descriptive):** the same rule on arm N; the residual pooled over the two sides per FM; the DEV verdict table (F, S, N; before and after p/π̂; own and common set); binary-only and multiclass-only splits (5 multiclass tables: descriptive only). No multiplicity correction; nothing secondary can change the overall decision.

## 7. Cost

Estimated from the DEV timings (`t1p_subset.py`: per FM, OLS of log seconds per fit on log rows predicted and log d over the 20 DEV tables; fits per table from a dry run of the design on the labels): 4,765 fits per FM, **≈ 51 CPU-hours** in all (TabICL 5, TabPFN v2 13, TabPFN v3.5 12, Kumo 21). The cap of 60 CPU-hours is met at K = 16, so the subset is not shrunk.

## 8. Code and outputs

- `b_dose/t1p_subset.py`, `b_dose/t1p_subset.csv`: subset rule and cost estimate.
- `b_dose/run_dose_t1p.py`: runner; log `b_dose/run_t1p.log`; raw output `b_dose/out_t1p/raw/{fm}_{key}.parquet`.
- `b_dose/analyze_dose_t1p.py`: analysis; `b_dose/out_t1p/SUMMARY.md`, `primary.csv`, `summary.csv`, `slopes.csv`.
- Macros `\ExBTp…` in `exploratory/out/exploratory_numbers.tex` via `make_exploratory_numbers.py`, tagged EXPLORATORY.

Any deviation from this document is listed in `out_t1p/SUMMARY.md`.
