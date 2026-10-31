# Study 9: how many labels does a tabular FM need? (Workstream C, confirmatory)

Status: FROZEN 2026-10-21. Committed before any Study 9 run on Tier 1, Tier 2, Tier 3 or HELD-OUT, and before the DEV dry-run. Changes only by dated amendments at the end. Sec. 10 fixes how the analysis code is frozen.

Code: `src/simaudit/runner/study9.py` (runner), `src/simaudit/analysis/study9.py` (analysis), CLI `simaudit study9`. Outputs go to `results/study9/{set}/{key}_{learner}.parquet`, one file per (set, dataset, learner), written atomically and skipped when present.

## 0. Scope and positioning
- **Paper.** Workstream C feeds the evaluation paper (P1). P1 covers the bias of the stratified protocol, and how many and which labels a tabular FM needs.
- **Exploratory basis.** `reports/WORKSTREAM_C_EXPLORATORY.md` (DEV, Study 4 runs, 20 datasets). Every number there is hypothesis-generating; none is evidence here.
- **Prior art** (`research_notes/Autoresearch idea bank/workstream_c_how_many_labels_prior_art.md`). It shapes the framing:
  - **A universal "magic number" is already refuted.** Viering & Loog 2022; van Smeden 2016; the FM curves are smooth and log-linear in 1–10 per class (TabLLM, TabLens, MachineLearningLM, LUCoS). H1 therefore *estimates* the distribution of the saturation point and its moderators. It does not test for a universal number.
    - The only data-dependent anchors in the literature are a coverage floor (coupon collector) and a ceiling on selection benefit (LUCoS ≈ 8C, TCM ≈ 20C). Both are estimated here per dataset.
  - **Label equivalence (H2) exists as a metric.**
    - "Data utilization rate": Culver et al. 2006, used by Lu et al., TMLR 2025, for warm-start tabular active learning.
    - "Relative sample efficiency": Tabula-8B.
    - Label-efficiency factors: USL, ECCV 2022; Pourahmadi et al. 2021.
    
    The novelty is only its use for label-free cold-start selectors and for the stratified protocol, with tabular FMs, in 1C–30C.
  - **Label-free coverage budgets (H3).** The formula is trivial (multivariate hypergeometric; min{B : Σ(1 − p_k)^B ≤ δ} with replacement). No work estimates p̂ from unlabeled data and validates the recommended B. Known weak points:
    - clusters are not classes;
    - p_min errors are costly: overestimating it by 1.5× drops coverage to about 0.86;
    - coverage does not make a class learnable.
  - **Stopping (H4).** This is the most novel hypothesis and the riskiest. No stopping rule has been validated below about 10 labels per class; Kottke 2019 is negative in 10–50 labels. Nobody has validated a signal at budget n as a predictor of the realized gain from n to n + Δ.
    - Pre-registered safeguards: label-free signals only, computed on a fixed stop set (Kurlandski & Bloodgood 2022); the baseline is the trivial mean gain.
    - A failure is reported as a documented negative.
- **Within-class comparison** (user rule). Stopping signals are compared with stopping baselines, and selectors with random selection under the same FM. FMs are context only.

## 1. Data
| Set | Definition | Role | Learners |
|---|---|---|---|
| DEV | the 20 pilot datasets | dry-run of the pipeline only; never pooled, never confirmatory | FM2 |
| EXP | Tier 1 ∪ Tier 2 (2a + 2b) of Study 8 Amendment 1, after every duplicate check (A1.3) | **confirmatory** | Tier 1: FM1, FM2, FM3, FM5; Tier 2: FM1, FM2 |
| T3 | Tier 3 (PMLB, integer-coded categoricals) | pre-specified sensitivity for H1 and H3 only | FM1, FM2 |
| HELD-OUT | the 32 sealed datasets of pilot I2 | one replication, at the very end (Sec. 9) | FM1, FM2, FM3, FM5 |

- FM1 = TabPFN v2, FM2 = TabICL, FM3 = Kumo-Tabular, FM5 = TabPFN v3.5, with the Study 4 wrappers. HELD-OUT use was authorised; it stays sealed until Sec. 10's freeze is committed.
- **If OpenML does not return.** Without Tier 1 and Tier 2b, EXP would be Tier 2a alone (26 datasets). That is too few for the planned power. The EXP analysis then waits, and the authors decide. Nothing confirmatory is run on Tier 2a alone.
- **Row-order leak and its fix.** Many OpenML files are stored sorted by class: 11 of the 20 DEV datasets are, so row indices leak the label.
  - Before any selection or fit, the runner applies a fixed per-dataset permutation π_d = `default_rng([base_seed, did, 13]).permutation(n)` (namespace 13) to X and y.
  - Every index inside Study 9 refers to the permuted order. Stored row indices are mapped back to the original order (π_d[i]) for logging only.
  - The shared loader and the selection code are not changed.
  - A unit test shows that an index-spacing selector gains nothing on a class-sorted synthetic dataset.

## 2. Design
### 2.1 Stop set, pool and evaluation panel (all in permuted order)
- **Stop set.** S = rows [0, s) with s = min(500, ⌊n / 5⌋). It is never labeled and is the fixed unlabeled stop set of H4.
- **Labeling pool.** P = rows [s, n). Every strategy picks only from P.
- **Evaluation panel.** E = rows [0, e) with e = min(n, 2000), so S ⊂ E.
- **Metrics.** Computed on E ∖ L, where L is the labeled set.
  - The permutation is uniform, so E is a uniform random subset of rows and E ∖ L is an unbiased subsample of the complement used in Studies 4 and 7.
  - E caps the number of predicted rows at 2000, for compute; DEV and HELD-OUT (n ≤ 1000) are scored on the full complement.
- **Clustering.** Label-free selectors cluster all rows (clients = all n rows) but pick from P.

### 2.2 Budgets
- **Grid.** m ∈ M = {1, 2, 3, 5, 7, 10, 15, 20, 30} labels per class; total budget B = m·C.
- **Cap.** A budget is feasible for a dataset iff B ≤ ⌊n / 2⌋.
- **Stratified.** Also needs m ≤ (smallest class count in P) − 1.
- DEV was censored at 10, which is why the grid extends to 30.

### 2.3 Strategies (nested: within a repetition, the budget-m set is a prefix of the budget-m′ set for m < m′)
| ID | Name | Definition | Reps |
|---|---|---|---|
| S-RR | `random` | rep r: one uniform permutation of P (selection seed of r); L_m = its first mC rows | 10 |
| S-ST | `stratified` | rep r: one uniform permutation of each class's rows in P; L_m = the first m of every class. **Uses y**; it is the "free labels" reference | 5 |
| S-KMn | `kmedoids_nested` | greedy facility location (PAM BUILD) in z-scored X, Euclidean; candidates P, clients all rows. j₁ = argmin_j Σ_i d(i, j); then j_t = argmax_j Σ_i max(0, D_i − d(i, j)) with D_i = min over picks so far; ties go to the lowest index. L_m = the first mC picks | 3 (FM seed only; the selection is deterministic) |
| S-LUn | `lucos_nested` | the same greedy rule on the TabClustPFN row embeddings of Study 4 Amendment 2 (`selection.lucos.embed`, read-only reuse) | 3 (FM seed only) |

- **Flagged adaptation.** S-KMn and S-LUn replace the per-budget k-medoids re-runs of Studies 4 and 7 (S-KM, S-LUCoS) with their nested greedy counterparts.
  - Nesting is required by H4 and is what a practitioner who labels in rounds would use.
  - The non-nested S-KM and S-LUCoS are run on Tier 1 by Study 7 (mult ∈ {1, 3, 5, 10}). Their agreement with S-KMn and S-LUn there is reported descriptively, as a bridge to Studies 4 and 7.

### 2.4 Fits, seeds and stored outputs
- **Seeds (namespace 12).**
  - Selection: `SeedSequence([base_seed, did, rep, strategy index, 12])`.
  - FM: `SeedSequence([base_seed, did, m, rep, strategy index, 12])`.
  - The strategy index is the position in `study9.STRATEGIES`, which is append-only.
  - In every seed, did is the OpenML ID. A PMLB dataset uses 2⁴⁰ + its negative ID (`build_pool_pmlb.pmlb_id`), which keeps the seed non-negative and disjoint from OpenML IDs.
- **Fit.** Each fit uses X[L], y[L] and predicts E ∖ L. With missing classes, the FM is fitted on the seen classes and unseen classes get probability 0 (as in Studies 4 and 7). With one seen class, every row is predicted as that class.
- **Stored per fit.**
  - dataset key, learner, strategy, m, B, rep, seeds;
  - the labeled rows (original indices);
  - the query rows (original indices) and the float32 probability matrix over all C classes;
  - class coverage and seen classes;
  - seconds;
  - metrics on E ∖ L: macro-F1 (**primary**), accuracy, balanced accuracy, log-loss and AUC (`analysis.study7.metric_set`, read-only reuse).
- The S rows are inside E, so every stop-set signal is computable from the stored probabilities.

## 3. Hypotheses, estimands and decisions
The unit is the dataset. Curves are the mean over repetitions per (dataset, learner, strategy, m). The primary learner is FM2, run on every EXP dataset; FM1 is the pre-specified replication. Unless stated otherwise, the primary metric is macro-F1 and accuracy is reported alongside it. ε = 0.01.

### H1. Saturation point, its distribution and moderators
**Estimand.** For a curve f on the feasible grid m₁ < … < m_K, the segment slopes are s_k = (f(m_{k+1}) − f(m_k)) / (m_{k+1} − m_k), measured per extra label per class.
- Let k* = min{k : s_j < ε for all j ≥ k}.
- Then m* ∈ (m_{k*−1}, m_{k*}], with m₀ = 0, so m* ≤ m₁ is left-censored.
- If s_{K−1} ≥ ε, then m* ∈ (m_{K−1}, ∞), right-censored.
- A practitioner would continue at m_{k*−1} and stop at m_{k*}.

**H1-dist (estimation, primary for the paper).**
- The distribution F of m* over EXP datasets (FM2, S-RR) is estimated by Turnbull's NPMLE, the interval-censored Kaplan–Meier.
- F is identifiable at the grid points only. We report F̂(m) for m ∈ M; the median and quartiles as grid intervals; and P(m* ≤ 5), P(m* ≤ 10) and P(m* > 20).
- Each comes with a dataset-bootstrap 95% CI (2000 resamples).
- *Prediction from DEV:* median in (5, 20], and an interquartile range spanning at least two grid steps (no universal number).
- *Sensitivity:* each tier separately; equal weight per stratum (post-stratified); EXP ∪ T3; per C bin; S-ST, S-KMn and S-LUn; FM1; accuracy.

**H1a (dataset, not FM).**
- ICC(2,1) (two-way random, absolute agreement, Shrout & Fleiss) of the ordinal category of m*. The category is the index of its interval on M: 1 ≤ … ≤ 9, and "> 20" is the top category.
- Computed across FM1 and FM2 over EXP datasets whose feasible grid is the full M (n ≥ 60C), so that every category means the same interval. No imputation.
- 95% CI by dataset bootstrap.
- **Supported** if the lower bound is ≥ 0.75 (good reliability, Koo & Li 2016); **refuted** if the upper bound is < 0.75; otherwise unresolved.
- *Secondary:* the four-FM ICC on Tier 1; ICC on all EXP datasets using the lower category bound (flagged).

**H1b (grows with C).**
- Interval-censored log-normal accelerated-failure-time regression of m*: log m* = β₀ + β₁ log C + β₂ log IR + β₃ log n + β₄ log d + σ·ε, with ε ~ N(0, 1).
- Fitted by maximum likelihood on the interval probabilities, using EXP, FM2 and S-RR.
- **Test:** one-sided Wald test of β₁ > 0, with the observed-information SE (numerical Hessian). In the Holm family (Sec. 6).
- β₂–β₄ are the pre-registered moderators, two-sided, descriptive. The same model is reported for FM1, S-ST, S-KMn and S-LUn.
- Caveat: C is confounded with the budget unit (B = mC).

**Coverage floor and selection ceiling (descriptive, pre-specified).**
- *Floor:* the share of datasets with m* ≥ the oracle coverage budget m95 (H3).
- *Ceiling:* m_sel* ∈ (m_{k−1}, m_k], where k is the smallest grid index from which |f_sel(m) − f_RR(m)| < ε at every larger feasible m. Computed for S-KMn and S-LUn under FM2, with the Turnbull distribution reported. It is compared with LUCoS (≈ 8C) and TCM (≈ 20C).

### H2. Label equivalence
**Estimand.**
- ρ_S(m) = m_eq / m, where m_eq is the random budget (labels per class) that reaches f_S(m).
- m_eq is found on the monotone envelope (running max) of the dataset's S-RR curve, interpolated linearly in log m over the feasible grid.
- **Censoring, with no imputation.**
  - If f_S(m) ≤ envelope(m₁), then ρ ∈ (0, m₁/m].
  - If f_S(m) > envelope(m_K), then ρ ∈ (m_K/m, ∞).
- This is the "data utilization rate" (Lu et al., TMLR 2025) and the "relative sample efficiency" (Tabula-8B), applied to cold-start selectors and to the stratified protocol.

**H2a (selectors and stratified are worth more than random at 1C).**
- For S-KMn, S-LUn and S-ST (3 tests): a one-sided exact sign test that P(ρ_S(1) > 1) > 1/2 over EXP datasets.
- ρ_S(1) > 1 ⟺ f_S(1) > f_RR(1), which is decidable under censoring. Ties count as "not > 1".
- **Magnitude.** The Turnbull median of ρ_S(1), with a bootstrap CI.
  - The exploratory "about 2×" for the selectors is retained if the CI intersects [1.6, 2.5] and its lower bound is > 1.2.
  - The "about 2.6×" for S-ST is retained if the CI intersects [2.0, 3.2].
  - Otherwise the estimate is reported as the finding.

**H2b (the advantage fades by 5C).**
- For S-KMn and S-LUn (2 tests): a generalized sign test that ρ(1) > ρ(5) within datasets.
- A dataset counts when its two intervals do not overlap; overlapping pairs are dropped and counted.
- One-sided binomial test.

**Reported for all.** ρ at m ∈ {1, 2, 3, 5, 7, 10}, in macro-F1 and accuracy. Also the coverage equivalence: the random budget that reaches the strategy's class coverage (cf. Pourahmadi et al.).

### H3. Label-free coverage budget
**Oracle quantities.** These come from the class counts in P, since random draws come from P.
- p_cov(B) = P(every class has ≥ 1 label) and p_learn(B) = P(every class has ≥ 3 labels), for B uniform draws without replacement. Both are computed exactly, by a dynamic program over the multivariate hypergeometric (checked against inclusion–exclusion in the tests).
- m95 = min{B : p_cov(B) ≥ 0.95} / C, and m95_learn = min{B : p_learn(B) ≥ 0.95} / C.

**Label-free estimate.**
- k-means with K = C (C known, as the budget unit already assumes) on z-scored X of all rows: `n_init = 10`, `random_state = base_seed`.
- The cluster sizes counted over P replace the class counts in the same formulas, giving m̂95 and m̂95_learn.
- The with-replacement closed form min{B : Σ_k (1 − p̂_k)^B ≤ δ} is reported beside it. The ln(C/δ)/p_min shortcut is not used: it overshoots.

**H3a (accuracy).**
- The share of datasets with |log(m̂95 / m95)| ≤ log 1.5 (EXP).
- **Supported** if the share is ≥ 0.75 and its Clopper–Pearson 95% lower bound is ≥ 0.65; **refuted** if the upper bound is < 0.75; otherwise unresolved.

**H3b (pre-specified failure mode: extreme IR).**
- k-means splits large classes and absorbs small ones. Its clusters are then more balanced than the classes, so it *under-predicts* m95 when IR is extreme. This is the unsafe direction.
- **Test:** one-sided Kendall τ_b < 0 between the signed error log(m̂95 / m95) and log IR (Holm family).
- Also reported:
  - the unsafe rate (m̂95 < m95 / 1.5) per IR bin;
  - the over-prediction rate (k-means isolating a tiny outlier cluster, the exploratory 1063/42544 cases);
  - the oracle coverage actually achieved at the recommended budget, p_cov(m̂95 · C).

**H3c (secondary).**
- The H3a share for the learnable variant (m̂95_learn against m95_learn).
- *Sensitivity to p_min:* the achieved p_cov at the budget computed from oracle shares with the rarest share inflated by 1.5× and 2× (the others renormalised).
- *Validity check:* p_cov(B) against the realized full-coverage rate of the 10 nested S-RR draws.

H3 needs no FM. It is computed on EXP (confirmatory) and on T3, DEV and HELD-OUT (descriptive).

### H4. Stopping signal (label-free, fixed stop set)
**Rows.** One row per (dataset, rep, transition k), for S-RR under FM2, over consecutive feasible triples (m_k, m_{k+1}, m_{k+2}).
- *Target:* the realized next slope g = (f(m_{k+2}) − f(m_{k+1})) / (m_{k+2} − m_{k+1}), using that repetition's scores.
- *Decision time:* m_{k+1}.

**Primary signals, both on the stop set S.**
- *A (agreement):* the share of S rows on which the FM2 argmax with context L_{m_k} equals the one with L_{m_{k+1}}.
- *D (FM disagreement):* the share of S rows on which FM1 and FM2 disagree with context L_{m_{k+1}}.

**Models.** Leave-one-dataset-out OLS.
- *Baseline:* g ~ transition fixed effects, i.e. the global mean gain per transition.
- *Signal model:* g ~ transition fixed effects + signal + signal × log m_{k+1}.

**Estimand.** R = Σ|e_signal| / Σ|e_baseline| over all EXP rows. Its 95% CI comes from a cluster bootstrap over datasets of the LODO residuals (2000 resamples); the one-sided p is the bootstrap share with R ≥ 1.
- **Supported** for a signal if the Holm-adjusted p is < 0.05 (2 tests in the Holm family). Otherwise the result is a documented negative: no label-free signal beats the mean gain in this regime.

**Secondary (exploratory).**
- *Other signals:* Cohen's κ between consecutive predictions (Bloodgood's SP; its pe → 1 failure is reported); mean max-probability (MES); predicted-label entropy; and the coverage deficit (classes seen < C).
- *Meta-learned prior:* LODO regression of g on the transition fixed effects plus log C, log n and log d (label-free traits from other datasets, cf. Kielhöfer 2024 and Mahmood 2022). Also its combination with A and D.
- *Decision analysis:* "stop at the first m_{k+1} where the predicted g < ε". Reported: labels used, and the macro-F1 lost against the final feasible budget, compared with the fixed rules 5C and 10C.
- *The same analysis* for S-KMn and S-LUn.
- *Prior-art baselines not used.* Power-law extrapolation and PCF need labeled scores, which a cold-start practitioner does not have. Test-set scores would be an oracle, so they are excluded as deployable baselines.

## 4. Censoring
- m*, m_sel* and ρ are interval-censored with endpoints on the grid. They are never imputed.
- Distributions use Turnbull's NPMLE; the regression uses the interval likelihood; the tests use only the order relations that the intervals decide.
- The cap (B ≤ n / 2) shortens the grid of small or many-class datasets. That right-censors them earlier, which the interval methods handle. The number censored is reported per C bin.

## 5. Moderators
- **Moderators:** C, IR, n and d, all as logs and measured on the loaded data.
- **Where they enter:**
  - the H1b regression;
  - Kendall τ with log ρ_S(1) (H2) and with the signed H3 error (H3b);
  - the H4 meta-prior.
- All are descriptive except the H1b and H3b tests.

## 6. Multiplicity
**Holm family** (9 tests, α = 0.05), EXP, FM2:
- H1b;
- H2a × 3 (S-KMn, S-LUn, S-ST);
- H2b × 2;
- H3b;
- H4 × 2 (A, D).

H1-dist, H1a and H3a are estimation decisions by pre-specified CI thresholds. A Holm-confirmed result is called FM-robust if FM1 shows the same direction with an unadjusted p < 0.05.

## 7. Power (N = number of EXP datasets; the realistic range is 100–200, see Study 8 A1.6)
- **Kendall τ / H1b slope.** The minimum detectable τ at 80% power, one-sided, at the Bonferroni bound α = 0.05/9, is 0.27 (N = 72), 0.23 (100), 0.19 (150) and 0.16 (200). DEV's Spearman 0.55 (≈ τ 0.4) for C is well above all four.
- **Sign tests.** Power at α = 0.05/9:

  | true P(ρ > 1) | N = 100 | N = 150 | N = 200 |
  |---|---|---|---|
  | 0.65 | 0.63 | 0.85 | 0.95 |
  | 0.70 | 0.92 | 0.99 | 1.00 |

- **H3a.** For an observed share of 0.75, the Clopper–Pearson 95% interval is [0.65, 0.83] at N = 100 and [0.67, 0.81] at N = 150. The "supported" rule is attainable from N ≈ 100.
- **H4.** At N = 20 the exploratory CI of the MAE ratio was ±0.2. Scaling by √N gives ±0.07 at N = 150, so a ratio of 0.9 is resolvable.
- **H1-dist.** The CI half-width of P(m* ≤ 10) near 0.5 is ≈ 0.10 at N = 100 and 0.07 at N = 200.

## 8. Order of work
1. Freeze this file (commit).
2. Implement the runner and the analysis with unit tests on synthetic data. Run DEV (FM2 only) as a pipeline dry-run, and run the analysis on DEV. DEV is exploratory, and ICC and D cannot be computed there with one FM.
3. Freeze the analysis code (Sec. 10).
4. Build the pool when OpenML returns (Study 8 Amendment 1).
5. Report counts and compute. Run EXP only after the authors' approval.
6. Run the EXP analysis once and commit the report.
7. Run HELD-OUT once (Sec. 9).

## 9. HELD-OUT replication
- After the EXP report is committed, HELD-OUT is unsealed for Study 9 and run once with all four FMs, using the same code and seeds rule.
- **Reported:** every H estimate with its CI, and the decision under the same thresholds; Holm within the HELD-OUT family. If other Workstreams use HELD-OUT, their p-values are also Holm-adjusted across Workstreams (WORKSTREAMS rule).
- **Replication** means the same decision direction as EXP.
- HELD-OUT (n ≤ 1000, mostly binary, drawn like DEV) is narrower than EXP, so non-replication of the C and IR moderators is expected from range restriction. It is reported, not explained away.

## 10. Freezing the analysis
- The analysis module and its tests are committed after the DEV dry-run. A dated freeze note (Sec. 13) records the commit hash and the SHA-256 of `analysis/study9.py` and `runner/study9.py` before any EXP run.
- After that, a change is allowed only as a dated amendment, for a bug, with the diff and its effect on DEV reported. No threshold, estimand or test may change after any EXP result exists.

## 11. Compute (Study 8 A1.4; refined by the DEV dry-run)
- **Grid size.** Up to 9 budgets × (10 + 5 + 3 + 3) = 189 fits per dataset and learner; the cap removes the large budgets on small tables.
- **Unit costs.** Study 4 timings, ×1.5 for the larger contexts, with at most 2000 predicted rows: about 4.4 worker-hours per Tier 1 dataset (4 FMs) and about 1.5 per Tier 2 or T3 dataset (FM1 + FM2).
- **Totals.** About 72 × 4.4 + 128 × 1.5 + 30 × 1.5 ≈ 550 worker-hours, plus HELD-OUT (≈ 32 × 1.5 × 2).
- **Machine.** The machine is shared: at most 4 workers, one BLAS/torch thread each.

## 12. Threats
- **The curve is measured, not known.** Each point is a mean over 10 (or 3–5) draws on E ∖ L, so slopes near ε are noisy, and m* can move by one grid step. This is why m* is treated as an interval and H1 is a distribution, not a number.
- **Coarse grid.** ε = 0.01 per label per class is a convention. The sensitivity at ε ∈ {0.005, 0.02} is reported.
- **Selector variants.** The nested greedy selectors differ from the per-budget k-medoids of Studies 4 and 7; the Tier 1 bridge quantifies the difference.
- **Representativeness.** EXP is OpenML plus PMLB, numeric-only, and dominated by binary tables. Multiclass high-IR strata are short (Study 8 A1.6). Moderators are analysed on measured C and IR.
- **Order artefacts.** The row-order leak is neutralised by π_d. Selectors that use row order (none of ours do) would otherwise be biased.
- **Signals and FMs.** FM1 and FM2 share some design (ICL transformers); their disagreement may understate model uncertainty.

## 13. Amendments and freeze notes

### Freeze note 1 (2026-10-21): DEV dry-run done, analysis code frozen
- **Run.** `simaudit study9 --set dev --workers 2` (FM2, one thread per worker): 20/20 datasets, 3576 fits, no failure.
- **Fit time.** 5.96 worker-hours in total, on a shared machine with a load average of about 150. That is ≈ 0.71 worker-hours per 1000 evaluated rows for FM2 over the full grid, which supersedes the 0.61 assumed in Sec. 11.
- **Report.** `reports/STUDY9_DEV.md`, generated by `reports/study9_report.py dev`. It is exploratory and not evidence for any H. ICC and signal D are not computable with FM2 alone.
- **Frozen code** (commit R003; SHA-256):
  - `src/simaudit/analysis/study9.py` `aaf0bae1defa10d80b1db235614fe228ebf0aa26e82499d0d29dd51a3f6c50ba`
  - `src/simaudit/runner/study9.py` `6fe40d008265fff82a64c8b98321774dc30736c075e89163acdcf176787dd4c0`
  
  Sec. 10 applies from here.
- **Refined compute for the expansion.** Per 1000 evaluated rows: FM2 0.71 worker-hours; FM1, FM5 and FM3 at 1.8×, 2.1× and 3.0× of that (Study 4 ratios).
  - Tier 2a (17.6k evaluated rows, FM1 + FM2): ≈ 35 worker-hours.
  - Tier 3 (30.7k): ≈ 61.
  - Tier 1 (assumed 65k rows, 4 FMs): ≈ 370.
  - Tier 2b (up to about 92k): ≈ 180.
  - HELD-OUT (≈ 13k, 4 FMs): ≈ 75.
  - In total ≈ 720 worker-hours, about 7.5 days at 4 workers.

### Amendment 1 (2026-10-21): T1′ substitutes for Tier 1 under the Study 8 trigger
Committed before any Study 9 run on expansion data or HELD-OUT, and before the Sec. 10 freeze. Study 8 Amendment 2 (A2.5) defines the OpenML-free confirmation set T1′, drawn from PMLB Tiers 2a and 3 and UCI Tier 4, with Tier 1's quotas. It also fixes a trigger: whether Tier 1 has been frozen by T*, the first of the Study 7 DEV-report freeze and this study's Sec. 10 freeze note.
- **If the trigger fires:**
  - EXP becomes **EXP′ = T1′**, confirmatory, with FM1, FM2, FM3 and FM5 on every member;
  - the "If OpenML does not return" rule of Sec. 1 is superseded, and the T3 sensitivity set is replaced by the T1′ sensitivities of Study 8 A2.5 (numeric-only subset, balanced core, per source);
  - Tier 1 ∪ Tier 2b, when built, is a one-time replication, never pooled.
- **If the trigger does not fire:** Sec. 1 stands, and Tier 4 joins EXP as Tier 2a does (FM1, FM2).
- **Either way:**
  - the integrity screens of Study 8 A2.7 apply: leaky columns are dropped, excluded datasets never enter;
  - no expansion result is analysed before the HELD-OUT signature check of Study 8 A2.6 has run.

## Note (2026-10-28): paper assignment (administrative, no analysis change)
The project's results are split across several papers. The "how many labels" results of this study are reported in paper P3, not in P1. P1 keeps only the stratified-versus-random label equivalence ρ from Study 9, still with the Holm adjustment over the full Study 9 family. No hypothesis, data, analysis or decision rule changes.

### Amendment 2 (2026-10-28, before any HELD-OUT access): label normalisation on HELD-OUT
- **Timing.** This amendment was committed before any Study 9 run on HELD-OUT and before the joint opening (Study 13 Sec. 0.3). No Study 9 HELD-OUT output exists, `sealed_heldout/` has not been opened, and no HELD-OUT label or metadata was read.
- **Why.** In the T1′ table UCI 257, "Very Low" and "very_low" are stored as two classes. Study 9 and Study 13 open HELD-OUT in one event and must use the same labels. The step is defined in Study 13 Amendment 1 (A), committed together with this amendment.
- **What.** The HELD-OUT driver (`runner/study9_heldout.py`) loads every table through `study13.SealedLoader(normalise_labels=True)`. The raw label strings come from the sealed raw OpenML cache and are used only if they reproduce the stored codes row by row.
  - A non-numeric label is lowercased, each run of whitespace, `_`, `-` and `.` becomes one space, and the result is stripped. Numeric labels are left unchanged.
  - Labels that collide are merged. The merged class takes its smallest old code, and the codes are made contiguous.
  - Without a merge, the table is used exactly as stored.
- **Logging.** The step runs once per table at load time, after the manifest and content-hash checks and before any selection. Every table, mapping and merge goes to `results/study9/heldout/access_log.jsonl` and to `label_normalisation.csv`, both committed with the HELD-OUT report.
- **Analysis.**
  - The HELD-OUT estimates and decisions use the normalised labels; C and IR (Sec. 5) are measured on them.
  - A pre-declared sensitivity reports the HELD-OUT family without the tables where a merge happened, and also without unverified tables if any exist. It is descriptive and uses the same thresholds.
  - If no merge happens, the report says so.
- **Not changed.**
  - EXP/T1′: UCI 257 there is handled as a known data issue with its own sensitivity check (`paper/p1/exploratory/uci257/`), not by this amendment.
  - `runner/study9.py` and `analysis/study9.py`: their Freeze note 1 hashes still match (the driver checks them). Sec. 10 is respected, since no analysis code, threshold, estimand or test changes.
  - The new secondary learners and metric families of Study 13 Amendment 1 (B, C) are not added to Study 9.

### Amendment 3 (2026-10-28, before any HELD-OUT access): unverified labels on HELD-OUT
- **Source and timing.** This decision was taken on 2026-10-28. It matches Study 13 Amendment 2 (committed together), because both studies load HELD-OUT through the same `study13.SealedLoader`. It was committed before any Study 9 HELD-OUT run, and `sealed_heldout/` has not been opened.
- **Rule.**
  - A HELD-OUT table whose raw label cache is missing or does not re-encode to the stored codes runs with its stored labels and is flagged "unverified" (Amendment 2).
  - The Amendment 2 sensitivity that also drops unverified tables is reported only if they number at most half of the HELD-OUT tables. Otherwise the report says it is not reported and why.
  - The sensitivity without merged tables is reported in every case.
- **Not changed.** `runner/study9.py` and `analysis/study9.py` (pinned) and every hypothesis, threshold and test. The sensitivity is descriptive and computed outside the frozen analysis module.

### Amendment 4 (2026-10-28, before any HELD-OUT access): integrity screen on HELD-OUT, flag-only
- **Source and timing.** This decision was taken on 2026-10-28. The amendment was committed before any Study 9 HELD-OUT run and before the joint opening. `sealed_heldout/` has not been opened.
- **Ambiguity resolved.** Amendment 1 says "the integrity screens of Study 8 A2.7 apply: leaky columns are dropped, excluded datasets never enter". It does not say whether this covers HELD-OUT. Its bullet sits in an amendment about EXP′/T1′, and Study 8 A2.7 states "DEV and HELD-OUT are not re-screened here". Study 13 screens the same HELD-OUT tables flag-only (decision of 2026-10-27).
- **Decision.** In the Study 9 HELD-OUT run, the A2.7 screen **excludes no table and drops no column**. It runs **flag-only**.
  - It uses the Study 13 code (`study13.screen_all` / `screen_table`).
  - It runs inside the run, after the guards, the manifest check, the content-hash check and the label step (Amendment 2), and before any selection or fit.
  - Output: `results/study9/heldout/integrity_screen.csv`, plus one `integrity screen (flag-only)` entry per table in `access_log.jsonl` (flagged, flag_id, A2.7 decision).
- **Pre-declared sensitivities**, as in Study 13 Sec. 1. They are descriptive, use the same thresholds and decision rules, and Holm is recomputed within the reduced HELD-OUT family:
  - **SENS-INT:** drop every flagged table.
  - **SENS-ID:** drop only the identifier/counter-column tables (`flag_id`).
  - A HELD-OUT decision that changes under SENS-INT is reported as fragile.
- **Rationale.**
  - (i) *Fidelity:* A2.7 says HELD-OUT is not re-screened, and Amendment 1's bullet concerns EXP′/T1′.
  - (ii) *Consistency:* both confirmations in the joint opening use exactly the same tables.
  - (iii) *No forking paths:* the primary analysis and both sensitivities are declared now.
  - (iv) *Power:* no table is lost from the primary HELD-OUT analysis.
- **Not changed.** `runner/study9.py` and `analysis/study9.py` keep their pinned hashes. The sensitivities run the frozen analysis on the reduced table sets. EXP/T1′ handling under Amendment 1 is unchanged.
- **Code and test.** `runner/study9_heldout.screen_flags`; test `test_screen_is_flag_only_and_runs_before_any_fit` (synthetic). A table with 100% conflicting duplicates gets an A2.7 "exclude" decision and is flagged, yet it still runs, and the screen comes before every job load.

- **2026-10-30, joint opening note:** see `prereg/JOINT_OPENING_2026-10-30.md` (studies, confirmatory cells, the cross-study Holm family of 19 tests, launch settings). Committed before any HELD-OUT access.
