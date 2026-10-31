# P1: two corrections for the stratified-protocol bias (exploratory, DEV)

Status: PLAN written and committed 2026-10-27, before any correction was computed or any new FM run was made. EXPLORATORY: not pre-registered, DEV only (the 20 pilot-I2 datasets), no T1′ result is used, `sealed_heldout/` is not touched. No frozen analysis code and no prereg file is modified. The numbers below the "Results" line are added after the runs.

Theory references: reports/THEORY_PROTOCOL_BIAS.md (Lemma 1, Prop. A, Prop. B, the prior non-identifiability cross-reference); reports/STUDY7_REPORT.md.

## 0. What is stored (inventory, read-only)

| Source | Strategies × reps per (dataset, FM, mult) | Stored per run |
|---|---|---|
| `results/study4_stage_a/{did}.parquet` (FM5), `results/study4_stage_b/{did}_fm{1,2,3}.parquet` | random 10, stratified 10 (seed namespace 5) | labeled indices, argmax predictions, macro-F1, accuracy (no probabilities) |
| `results/study7/a/{did}_{fm}.parquet` (FM1, FM2, FM3, FM5) | random 20, stratified 5 (namespace 10) | labeled indices, full probability matrix (all rows × C), metrics on the complement |
| `results/study7/b/{did}_fm{1,5}_bal.parquet` | random 20, stratified 5, TabPFN `balance_probabilities=True` | same as Part (a) |
| `results/study7/c/{did}_{fm}.parquet` | dose-response: prescribed compositions, 2 arms × 5 shares × mult {3, 5, 10} × 10 reps, full coverage | labeled, probabilities, metrics, and the post-hoc p/π̂ (uniform-target) metrics |

Both studies use the same FM wrappers (`study4._predictor`) and the same complement metrics: macro-F1 over all C classes with zero_division = 0, and accuracy. DEV has 14 binary and 6 multiclass datasets (C = 3, 3, 3, 8, 9, 10). For dataset 694, the main Study 7 runs are used, with its ID column, as in Study 7's confirmatory analysis.

## 1. Correction 4: composition-weighted estimator of realistic performance

**Estimand.** For a cell (dataset, FM, mult), the realistic-protocol mean $\theta=\mathbb E_R M=\sum_v P_{\text{MVHyp}}(N=v)\,g(v)$, with $g(v)=\mathbb E[M\mid N=v]$ (Lemma 1). $M$ is macro-F1 or accuracy on the complement.

**Ground truth $T$.** The plain mean of every realistic random run of the cell: Study 4 (10) + Study 7 Part (a) (20) = 30 runs, plus the new FM2 random runs of Sec. 1.3 for multiclass cells. In the subsampling experiments, $T$ is the mean of the runs *not* used by the estimator (held-out truth). The reported MSE is corrected for the truth's own noise, $\widehat{\text{MSE}}=\text{mean}(\hat\theta-T)^2-s^2/n_T$.

**Bias being corrected.** $\Delta^S=\bar M^S-T$, the stratified mean minus the truth, per cell.

### 1.1 (a) Post-stratification of existing random runs
Given $k\in\{5,10,20\}$ random runs drawn without replacement from the cell's pool (2,000 subsamples per cell):

| Estimator | Definition |
|---|---|
| NAIVE | mean of the $k$ runs |
| PS-SEEN | strata = number of seen classes $\lvert S(L)\rvert$; exact weights $P(\lvert S\rvert=j)$ by inclusion–exclusion; empty strata dropped and the weights renormalised |
| PS-A7 (binary) | covered runs averaged and weighted by the exact $P(\text{cov})$; each missed composition gets its exact value under L0 (constant prediction; A7), weighted by its exact probability. No FM run is needed for misses |
| PS-COMP (binary) | strata = exact composition $N_1$ with exact hypergeometric weights; unobserved covered strata are filled by linear interpolation in $N_1$ between observed ones (constant extrapolation at the ends); misses as in PS-A7 |
| GREG | regression (control-variate) estimator with $Z=(\#\text{missed classes}, D)$, $D=\lVert N/B-u\rVert^2$, whose exact means are known (A2, B1): $\bar M-\hat\beta^\top(\bar Z-\mathbb E Z)$. $\hat\beta$ is pooled from the other datasets of the same (FM, mult), leave-one-dataset-out |

### 1.2 (b) Design version: few runs at chosen compositions
- **Binary (14 datasets, all four FMs, no new runs).** The design points are the stratified composition (Part (a), 5 reps) and the Part (c) compositions (minority share 0.1–0.9, 10 reps each, mult 3, 5, 10). By Lemma 1, a run at composition $v$ is a draw from $g(v)$.
  - DES-INTERP: $\hat g$ = piecewise-linear interpolation in $N_1$ over the design compositions (constant extrapolation); misses analytic (A7); integrated against the exact hypergeometric pmf.
  - mult 1: the only covered binary composition is (1, 1), the stratified one, so DES at 1C = stratified runs + exact misses.
  - STRAT+A7 (diagnostic): stratified runs stand in for every covered composition. This removes only the coverage part and shows how much the balance part matters.
  - Equal-run comparison: $r\in\{1,2\}$ reps per design composition (resampled), against NAIVE with the same total number of runs.
- **Multiclass (6 datasets, FM2 only, new runs).** Existing runs do not span the composition simplex: Part (c) varies one class against equal others. New design: 100,000 MVHyp compositions per (dataset, mult) are binned into $K=6$ strata by (#missed classes, $D$-quantile). In each stratum, 4 compositions are drawn from the conditional law (each with its own within-class uniform rows). DES-STRAT = $\sum_j w_j\bar M_j$, with exact-up-to-MC stratum weights $w_j$. This is unbiased by construction; the question is variance at equal runs (1 or 2 runs per stratum vs NAIVE with 6 or 12).

### 1.3 New FM runs (cost estimate; cap 2,000 fits)
FM2 (TabICL), seed namespace 21, `nice -n 15`, 4 workers, output under `results/p1_corrections/`:
- design: 6 datasets × 4 mults × 6 strata × 4 = 576 fits;
- extra realistic random runs to sharpen the multiclass truth: 6 × 4 × 40 = 960 fits.

Total 1,536 fits, ≈ 3 s each, so about 20–30 min on 4 workers. This is under the cap, so the runs go ahead. Binary cells need no new runs.

### 1.4 Criteria for Correction 4 (macro-F1 primary, accuracy secondary)
- **Promising.**
  - |mean bias| ≤ 0.005 with its CI over datasets containing 0;
  - and RMSE at least 20% below NAIVE at the same number of FM runs, for k = 5 or 10, in ≥ 3 of 4 FMs (post-stratification); or, for the design version, at the same number of runs in the binary or the multiclass block.
- **Not promising (kill).**
  - RMSE reduction < 10% at every k in every block;
  - or |bias| > 0.01.
- **Otherwise inconclusive.**

## 2. Correction 5: prior correction from the unlabeled pool

**Setting.** For every stored run (Part (a) random and stratified, all four FMs; Part (c) dose cells), take the probabilities $p(c\mid x)$ on the complement $U$ and the context prior $\pi^s=N/B$ over the seen classes. Unseen classes stay at probability 0.

**Prior estimators (no labels of $U$ are used):**
- **EM** (Saerens et al. 2002): iterate $p'\propto p\,\hat\pi/\pi^s$ and $\hat\pi=\text{mean}_U\,p'$ to convergence (tol 1e-8, ≤ 1,000 iterations).
- **BBSE-soft** (Lipton et al. 2018): solve $\hat C w=\bar p_U$, where $\hat C_{ij}$ is the mean of $p(i\mid x)$ over the context rows of class $j$. The context is the only labelled data, so $\hat C$ is a resubstitution estimate (optimistic, near-identity) and BBSE is reported as a weak variant. $w$ is clipped at 0, and $\hat\pi\propto w\,\pi^s$.
- **MLLS + BCTS calibration** (Alexandari et al. 2020): not run. It needs held-out labelled data for calibration. Cross-fitting on the context would cost ≥ 5 extra fits per run (≈ 50,000 fits for Part (a)), and at 1C there is nothing to cross-fit.

**Corrections compared:**
- **none:** the stored probabilities;
- **EM** and **BBSE:** $p\cdot\hat\pi/\pi^s$, renormalised;
- **oracle:** the true class shares of $U$ in place of $\hat\pi$;
- **uniform:** target $u$. This is TabPFN's `balance_probabilities`, applied post hoc to every FM; the native `fm1_bal` and `fm5_bal` runs of Part (b) are also reported;
- **EM-half** (exploratory, motivated by B4): target $\propto\sqrt{u\cdot\hat\pi_{EM}}$, the geometric midpoint.

**Outcomes.**
- macro-F1 and accuracy, recomputed from the probabilities with `analysis.study7.metric_set`; the recomputed "none" must match the stored metrics;
- per FM × mult × protocol: $\Delta$ = corrected − none, as dataset means over reps, with a paired bootstrap 95% CI over the 20 datasets (10,000 resamples) and the win-rate;
- also the stratified − random gap after each correction;
- the Part (c) dose-response drop (s = 0.5 → largest feasible s) under none / uniform / EM.

### 2.1 Criteria for Correction 5
**Key question.** Does EM help FM2 (TabICL) or FM3 (Kumo) under the realistic random protocol?
- **Promising.**
  - A gain with a CI excluding 0 for FM2 or FM3 in macro-F1 or accuracy, at ≥ 2 of the 4 budgets;
  - without a loss in the other metric larger than 0.01 with a CI excluding 0 at those budgets.
- **Not promising (kill).**
  - No such gain at any budget for FM2 and FM3;
  - or every gain is paid by an equal or larger significant loss in the other metric.
- **Otherwise inconclusive.**
- **Also reported:** the share of the oracle's gain that EM recovers.

## 3. Prior-art check
A quick web search (abstracts) for both corrections. The notes go in `research_notes/Autoresearch idea bank/protocol_corrections_prior_art.md`.

---
## Results (2026-10-27, exploratory, DEV only)

- **Tables.** `reports/P1_PROTOCOL_CORRECTIONS_TABLES.md`, generated by `reports/p1_corrections.py`.
- **New runs.** `reports/p1_corrections_runs.py`: 1,536 FM2 fits, as planned (576 design + 960 random), `nice -n 15`, 4 workers, no failures.
- **Checks.**
  - The recomputed "none" metrics equal the stored ones exactly (max |diff| = 0).
  - The post-hoc uniform correction equals Study 7's `pc_macro_f1` exactly.
  - The A7 constant-prediction values equal the 328 stored missed binary runs (|diff| ≤ 6e-17).
  - EM recovers a known prior on synthetic calibrated posteriors (0.201/0.799 for 0.2/0.8).

### Correction 4: composition-weighted estimator

**The bias being corrected** (stratified mean − random truth, macro-F1, mean over the 20 datasets):

| FM | 1C | 3C | 5C | 10C |
|---|---|---|---|---|
| FM1 | +0.114 | +0.064 | +0.050 | +0.021 |
| FM2 | +0.113 | +0.036 | +0.021 | −0.010 |
| FM3 | +0.129 | +0.043 | +0.026 | +0.001 |
| FM5 | +0.139 | +0.093 | +0.073 | +0.027 |

In accuracy the bias is small: +0.01 to +0.03 at 1C, then −0.03 to +0.004.

**(a) Post-stratification of existing random runs** (truth = held-out runs; RMSE corrected for the truth's noise):
- Every estimator is unbiased within noise. The mean bias is ≤ 0.003 with CIs containing 0, except PS-SEEN, at +0.004 to +0.007 at 3–5C, from dropping empty strata.
- **Best in macro-F1: GREG**, the control variates (#missed classes, D) with exactly known means. Its RMSE ratio to NAIVE is:
  - k = 5: 0.80 / 0.79 / 0.77 / 0.81 (FM1 / FM2 / FM3 / FM5);
  - k = 10: 0.82 / 0.81 / 0.80 / 0.83.

  That is about a 1.5× effective-run gain. Almost all of it comes from 1C (ratio 0.75–0.79). At 10C the ratio is 0.89–0.95.
- **Accuracy.** GREG gives nothing (ratio ≈ 1.00). PS-COMP and PS-A7 give 0.78–0.92.
- **Against the criteria.** The ≥ 20% reduction holds in only 2 of 4 FMs (k = 5). The kill condition is not met either. → **Inconclusive: a modest, real, unbiased variance gain, mostly at 1C.**

**(b) Design version.**
- **Binary, all FMs, no new runs.** DES-INTERP uses the stratified runs + the Part (c) compositions + exact misses.

  | Budget | RMSE ratio to NAIVE at equal runs | Bias (CI) |
  |---|---|---|
  | 1C | 0.29–0.47 (2–3× lower RMSE; e.g. FM2 0.054 vs 0.158 with one run) | −0.009 to −0.014, CIs containing 0 |
  | 3C (r = 2) | 0.61–0.74 | ≤ 0.005, CIs containing 0 |
  | 5C | 0.73–1.02 | |
  | 10C | 1.33–1.66, i.e. **worse** | |

  At 10C the Part (c) grid spreads runs evenly over shares 0.1–0.9, while the hypergeometric mass is concentrated near π. An allocation proportional to mass (Neyman) was not tested.
- **Multiclass, FM2, new design runs.** DES-STRAT uses 6 equal-mass strata over (#missed, D).
  - 1C: ratio 0.29–0.54, but bias −0.012. All 6 datasets are negative (−0.005 to −0.024), on a truth SE of 0.008–0.02 per dataset. With n = 6 this is not resolved. In expectation the design is unbiased; each stratum has only 4 fixed compositions.
  - 3C: 1.02–1.61 (worse).
  - 5C: 0.76–0.79.
  - 10C: 0.62–0.93.
- **Diagnostic STRAT+A7** (stratified runs + exact miss values, i.e. removing coverage only):
  - At 1C it removes almost all of the macro-F1 bias: +0.105…+0.115 → −0.010…−0.012 in binary.
  - From 3C on it leaves most of it, e.g. FM5 +0.102 → +0.082 at 3C and +0.082 → +0.075 at 5C.

  This is a direct check of the coverage/balance split of THEORY_PROTOCOL_BIAS.md: coverage dominates at 1C, balance from 3C on.
- **Against the criteria.**
  - The binary block at 3C meets "promising": |bias| ≤ 0.005 with the CI containing 0, and a ≥ 20% RMSE reduction in 4/4 FMs.
  - At 1C the RMSE gain is large, but |bias| is around 0.01.
  - At 10C the design is worse than NAIVE.

  → **Promising at 1–3C for binary pools (the regime where the protocol bias is largest). Not useful at 10C as designed. Inconclusive for multiclass.**

### Correction 5: prior correction from the unlabeled pool

**EM fails.** On FM posteriors EM drifts towards corner priors.
- **Prior error.** The median L1 error to the true complement prior is 0.29–0.52, against 0.20–0.32 for the plain context shares. Under random sampling, EM is worse than doing nothing about the prior.
- **Effect on macro-F1** (random protocol):
  - −0.04 to −0.15 for every FM × budget, with every CI excluding 0;
  - FM2: −0.065 / −0.117 / −0.084 / −0.039 at 1 / 3 / 5 / 10C;
  - FM3: −0.119 / −0.150 / −0.117 / −0.065.
- **Effect on accuracy:** −0.03 to −0.12, every CI excluding 0. The stratified protocol shows the same pattern.
- **EM-half** (the B4 midpoint) fails the same way.
- **BBSE** with a resubstitution confusion matrix gives small losses: −0.004 to −0.055 in macro-F1, and for FM5 at 3–5C a gain of +0.012/+0.013.
- **MLLS + calibration** was not run. It needs held-out labelled data.

**Oracle ceiling (the true pool prior).** This answers the key question independently of the estimator.
- **TabICL (FM2) and Kumo (FM3), random protocol:**
  - macro-F1 −0.006 to +0.005;
  - accuracy −0.003 to +0.017; only FM2 at 1C has a CI excluding 0 (+0.017).

  Even a perfect pool-prior estimate does not help them. Their posteriors do not respond to the context prior like a Bayes plug-in, which matches Study 7 Part (c).
- **TabPFN (FM1, FM5), random protocol:** oracle +0.013 to +0.025 in both metrics at 3–10C. Here the uniform target (= `balance_probabilities`) does as well or better for macro-F1: +0.015 to +0.042.
- **Uniform target for TabICL/Kumo** (the "native option" applied post hoc): it *hurts* macro-F1 at 3–10C (−0.012 to −0.024) and accuracy (−0.03 to −0.05).
- **Stratified protocol.** The oracle raises accuracy by +0.03 to +0.06 for every FM, with most CIs excluding 0. It lowers macro-F1 for TabPFN (−0.01 to −0.08) and leaves TabICL/Kumo flat. These are the signs B3/B4 predict (accuracy optimum at π, macro-F1 optimum in between).
- **Protocol gap.** No correction closes the stratified − random macro-F1 gap where it is large.
  - At 1C the oracle leaves it at +0.08 to +0.13, because at 1C the gap is coverage.
  - EM "closes" it only by degrading the stratified arm more than the random one.
- **Part (c) dose-response.**
  - EM makes the decline *steeper* for TabICL/Kumo, e.g. FM2 majority arm 0.088 → 0.229.
  - The oracle leaves FM2 unchanged (0.088 → 0.091).
  - The oracle cuts TabPFN's decline, e.g. FM5 majority arm 0.195 → 0.049.

**Against the criteria.** No EM gain for FM2 or FM3 at any budget, in either metric, and the oracle ceiling is about 0. → **Not promising (kill).** The useful by-products for P1:
1. The oracle result strengthens the Study 7 reading that TabICL/Kumo imbalance sensitivity is not a prior effect.
2. Label-free prior estimation from few-shot FM posteriors is unreliable (EM collapses), so "just correct the prior" is not a practical fix for the realistic protocol.

### Prior art (quick check, abstracts only)
Notes: `research_notes/Autoresearch idea bank/protocol_corrections_prior_art.md`.
- **(4) Not found.** Post-stratification or importance weighting over support-set class compositions in few-shot evaluation does not appear. The closest works are:
  - Ochal et al. 2021/2023, imbalanced support sets as conditions;
  - Veilleux et al. 2021, Dirichlet query proportions;
  - Meta-Dataset;
  - Active Testing, which weights points, not compositions.

  Reviewers may call the estimator textbook survey statistics; the new part is its use for the protocol gap.
- **(5) Partially done.**
  - DistPFN (Lee et al., ICML 2026) corrects TabPFN's context-prior bias at test time, apparently without estimating the test prior. Read it in full before any claim.
  - Generative Calibration (Jiang et al. 2023) tried EM/BBSE for LLM in-context learning.
  - Transductive few-shot methods (α-TIM, PADDLE) estimate query proportions inside their own models.

### Limitations
- **DEV only and exploratory.** n = 20 (14 binary, 6 multiclass), and the multiclass design uses one FM and 6 datasets.
- **Noisy truth.** The truth is a mean of 30 random runs (70 for FM2 multiclass). Its noise is removed from the MSE analytically but not from the bias CIs.
- **Untested designs.** Neyman or most-probable allocation for the design, and MLLS + BCTS with cross-fitting.
