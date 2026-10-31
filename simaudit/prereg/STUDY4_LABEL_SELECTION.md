# Study 4: which rows to label first? Cold-start label selection for tabular foundation models

Status: FROZEN 2026-10-21 with Amendment 1 (Sec. 6), committed before any selection run. The strategy list comes from the label-selection scout's reading (research_notes/Autoresearch idea bank/label_selection.md).

## 1. Motivation (exploratory evidence, pilot I2, DEV)
Across 10 random **stratified** labeled sets per dataset, the best FM's macro-F1 varies a lot:

| m (labels/class) | Mean | Std | Range | Best of 10 − mean |
|---|---|---|---|---|
| 1 | 0.584 | 0.104 | 0.317 | +0.132 |
| 3 | 0.695 | 0.062 | 0.198 | +0.068 |
| 5 | 0.717 | 0.056 | 0.172 | +0.063 |
| 10 | 0.764 | 0.032 | 0.100 | +0.041 |

The choice of rows is therefore worth more than any method gain seen so far (about 0.01). "Best of 10" is a test-set oracle, an upper bound, not an achievable result.

Our earlier protocol also has a problem. Stratified "m per class" sampling presupposes the labels; a practitioner labels B rows without knowing their classes.

## 2. Research questions
- **RQ-L1 (protocol).** How optimistic is stratified sampling compared with a realistic total budget B = m·C chosen uniformly at random (class coverage, macro-F1, accuracy)?
- **RQ-L2 (baselines).** At B ∈ {C, 3C, 5C, 10C}, do cold-start selection strategies from other fields beat realistic random selection for tabular FMs, and by how much? Candidates: k-center, k-means central, TypiClust, ProbCover, USL, Σ/V-optimality, and an outlier-avoiding variant; the final list is frozen in Amendment 1.
- **RQ-L3 (interaction).** Does good selection change the benefit or harm of transductive refinement (LLGC on an FM prior)?
- **RQ-L4 (for Study 3).** Is there headroom between the best literature strategy and the test oracle (best of R random draws) that a new strategy could target?

## 3. Design
- **Data.** The 20 DEV datasets of pilot I2. HELD-OUT stays sealed and is used only to confirm a final new strategy against the best baseline found here.
- **Selection.** Each strategy sees only X (z-scored), the budget and a seed; never y. Seeds come from namespace 5, disjoint from the pilot and the search seeds.
- **Learners.** TabPFN v2 (FM1), Kumo-Tabular (FM3), TabPFN v3.5 (FM5) and TabICL (FM2). TabFM is excluded: it is the weakest FM and costs ~60 s per call. Secondary learner: LLGC on the FM5 prior with default (α, k).
- **Missing classes.** If the selected rows cover only some classes, the FM is fitted on the seen classes and unseen classes get probability 0, so their F1 is 0 in the macro average. With one seen class, every row is predicted as that class. This is the real cost of not covering a class.
- **Evaluation rows.** All unselected rows (the complement), the standard practice in active learning. Threat: the evaluation set differs slightly between strategies; Sec. 5 gives a sensitivity analysis.
- **Repetitions.** 5 per (dataset, strategy, budget). Deterministic strategies vary only through their seed (k-means init, k-center start).
- **References.** The stratified sampler (draw_split, the pilot's protocol) and the oracle "best of the realistic random draws" (upper bound).

## 4. Analysis
- **Unit and metrics.** The unit is the dataset (mean over repetitions). Primary metric: macro-F1. Secondary: accuracy and class coverage.
- **Tests.**
  - Paired Δ against realistic random for each strategy × budget × learner.
  - Wilcoxon signed-rank with Holm correction within each budget.
  - Bootstrap 95% CIs over datasets; ROPE 0.01.
  - Distributions reported per dataset.
- **RQ-L1.** Δ(stratified − realistic random), and the class-coverage rate of the realistic random baseline.
- **RQ-L3.** Δ(LLGC − FM) by strategy; is the lower tail (10% quantile) better under the cluster-based strategies?

## 5. Threats
- The evaluation set depends on the selection (sensitivity analysis: score only on rows never selected by any strategy in that repetition).
- Datasets where every method sits at chance (e.g., 931, 827) dilute differences; they are reported separately, not excluded.
- Strategy hyperparameters (e.g., ProbCover's δ, TypiClust's neighbourhood size) follow the papers' defaults or label-free rules. They are never tuned on y.

## 6. Amendment 1 (2026-10-21, before any selection run): frozen strategies and Stage A

### 6.1 Strategies
Code: `src/simaudit/selection/strategies.py`; runner: `src/simaudit/runner/study4.py`. All strategies use z-scored X only, except the two references.

| ID | Name in code | Specification | Source |
|---|---|---|---|
| S-ST | `stratified` | m rows per class (`draw_split`, the pilot sampler); **uses y** | Reference (RQ-L1) |
| S-RR | `random` | B rows uniformly at random; **10 reps** (it is also the oracle pool) | Main baseline |
| S-KM | `kmedoids` | Alternating k-medoids, K = B, k-medoids++ seeding, best of 10 inits | LUCoS raw-space baseline (Ipas et al. 2026); Bae et al. 2024 |
| S-TPC | `typiclust` | k-means K = B; repeatedly take the cluster with the fewest picks, then the largest (clusters of ≤ 5 rows are used only once the larger ones run out), and pick its most typical row; typicality = 1/mean distance to min(20, size − 1) NNs | Hacohen et al. 2022 |
| S-PC | `probcover` | δ* = largest δ on a 100-point quantile grid (0.5–50% of pairwise distances) with ball purity ≥ 0.95 against k-means(C) pseudo-labels; greedy max coverage; random once all is covered. **Uses C** | Yehuda et al. 2022 |
| S-KC | `kcenter` | Greedy k-center from one random row | Sener & Savarese 2018 |
| S-OUT | `outlier_filtered` | IsolationForest (default) drops the 10% most anomalous rows, then uniform random | Chen et al. 2022 (outlier queries) |
| S-SIG | `sigma_opt` | Greedy Σ-optimality: Σ = (L + 0.01 I)⁻¹ on the unweighted union kNN graph (k = 10); pick arg max (1ᵀΣ·v)²/Σvv, then condition | Ma et al. 2013 |
| S-KMC | `kmeans_central` | k-means K = B; the row nearest each centroid | Common diversity baseline |
| D-SM | `stratified_medoids` | m k-medoids per class; **uses y; diagnostic only** | Decomposition (coverage vs within-class position) |

**Flagged adaptations.**
- Σ-opt: the regularizer 0.01 and the unweighted graph are ours; Ma et al. leave both open.
- TypiClust: the initial labeled set is empty, so K = B.
- ProbCover: the δ grid is ours.

These hyperparameters are fixed now and never tuned on y.

### 6.2 Stage A design
- **Budgets.** B = mult · C with mult ∈ {1, 3, 5, 10}. S-ST and D-SM use m = mult per class.
- **Repetitions.** 10 for S-RR, 5 for every other strategy.
- **Seeds.** `SeedSequence([base_seed, did, mult, rep, strategy index, 5])`. The strategy index is the position in `study4.ORDER`, which is append-only.
- **Learners.**
  - FM5 (TabPFN v3.5), with unseen classes at probability 0.
  - LLGC on the FM5 prior (α = 0.9, mutual kNN k = 10, labeled rows clamped), for RQ-L3.
  - FM1, FM2 and FM3 enter only as Stage B, and only if some strategy's paired Δ against S-RR leaves the ROPE (0.01) for FM5 at some budget.
- **Stored per selection.** Labeled indices and the full prediction vector, so that (i) the sensitivity analysis (scoring only on rows no strategy selected in that rep) and (ii) the split-half oracle are computed after the run, with no extra FM calls.
- **Split-half oracle (RQ-L4).** For each (dataset, mult), the complement rows are split in two halves by a fixed seed. The best of the 10 S-RR draws is chosen on half A and scored on half B, and vice versa. The two scores are averaged.

### 6.3 Confirmatory contrasts (DEV, Holm within each budget)
S-KM vs S-RR and S-TPC vs S-RR on FM5 macro-F1. Everything else is exploratory. Δ(S-ST − S-RR) answers RQ-L1 and is descriptive.

**Power.** The minimum detectable effect is ≈ 0.03 at n = 20 (label_selection.md, Sec. d). Effects of about 0.01 are reported as unresolved, not as null.

## 7. Amendment 2 (2026-10-21, before any LUCoS run): LUCoS baseline
The authors authorised downloading the LUCoS code and the TabClustPFN checkpoint.

**Sources.**
- S-LUCoS: MIT licence.
- TabClustPFN code, pinned commit d5d25a9; the repository declares no licence, so use is research-only and nothing is vendored or shipped.
- Checkpoint `step-10000.ckpt`: SHA-256 feb7ab7ad21cd4702b57ce92338ace65f0d852ef1bc00653f2d7049a235095a2.
- All three are kept in `../third_party_src/`, outside git.

**Strategy S-LUCoS (`lucos`), appended to `ORDER`, so earlier seeds are unchanged.**
- Embedding: TabClustPFN row representations (`col_embedder` → `row_interactor`), computed after the repository's own preprocessing (NaN → 0, z-norm, zero-variance columns removed). The checkpoint is loaded with `weights_only=True`.
- Selection: k-medoids with K = B on the embeddings, Euclidean metric, k-medoids++ initialisation, one initialisation per repetition, as in LUCoS's `KMedoidsUnderSampler` (sklearn-extra, method "alternate").
- Label-free. The embedding depends only on X and is cached by a hash of X.

**Runs.** The same datasets, budgets, repetition count (5) and seed rule as the other strategies, with all four FMs and the LLGC secondary learner.

**Analysis.** Exploratory: Δ against S-RR and against S-KM, Holm within FM × budget. No confirmatory status changes.

**Flagged adaptation.** k-medoids reuses our `_kmedoids` (`strategies.py`) with `n_init=1` instead of sklearn-extra's "alternate": k-medoids++ by plain D² sampling (numpy RNG, no greedy local trials), at most 100 iterations (sklearn-extra: 300), float64 distances (LUCoS: float32); if the single initialisation ends with fewer than B distinct medoids, the 10-init S-KM routine is run on the same embeddings.

## 8. Amendment 3 (2026-10-21, before any USL, RDSS or ZCore run): three more selection baselines
The authors authorised downloading and executing the official code of three more label-free selectors. RDSS and ZCore are the two other selectors in the LUCoS configuration (`config_exp2_LUCoS.yaml`, `instance_selection.py`); USL is the cold-start selector named in RQ-L2 and not yet run.

**Sources.** All three are kept in `../third_party_src/`, outside git. Nothing is vendored: the strategies are re-implemented in `src/simaudit/selection/`, and the official code is executed only by the integration equivalence tests.

| Baseline | Repository | Pinned commit | Licence |
|---|---|---|---|
| USL (Wang, Lian & Yu, ECCV 2022) | TonyLianLong/UnsupervisedSelectiveLabeling | ceaab3c (main, 2023-07-14) | MIT |
| RDSS (Shao et al., NeurIPS 2024) | YanhuiAILab/RDSS | 3d7bbd6 (the LUCoS submodule pin) | Apache-2.0 |
| ZCore (Griffin, Marks & Corso, WACV 2026) | voxel51/zcore | f9ed952 (the LUCoS submodule pin) | none declared; research-only |

**Representation.** LUCoS runs RDSS and ZCore in two spaces: the original features (imputed, one-hot, unscaled) and the TabClustPFN embedding. Here all three baselines use z-scored X, like every non-LUCoS strategy, so they pair with S-KM in the same space. The TabClustPFN-space variants are not run (flagged; they could be appended later under a new amendment).

**Strategies, appended to `ORDER` in this order, so every earlier seed is unchanged.**

| ID | Name in code | Specification | Hyperparameters (source) |
|---|---|---|---|
| S-USL | `usl` | Features Z = z-scored X / √d′ (d′ = non-constant columns). Density = 1 / mean squared distance to the K nearest rows, the row itself included (as keops `Kmin`). Lloyd k-means with B centroids, initialised at `torch.randperm(n)[:B]` under the selection seed. Then the regularised selection of `get_selection_with_reg` (CIFAR code): for `NITERS` rounds, each cluster in index order contributes its row of maximum density − W·reg. After each round, reg ← momentum·reg + (1 − momentum)·Σ_s 1/d²(i, s)^α over the selected rows s outside the row's own cluster, with squared distances ≥ horizon counted as 1e10. | K = 20; k-means 200 iterations; NITERS = 10, W = 0.5, momentum = 0.9, horizon = 4, α = 0.5 (USL `config.py` defaults, which `cifar10_usl.yaml` uses unchanged; K = 20 from the ImageNet config) |
| S-RDSS | `rdss` | `khsample` with the Gaussian kernel: μ_j = Σ_{i≠j} k(i, j) / N. Each step picks arg min over unselected j of F_j − α·μ_j, where F_j is the running mean kernel similarity of j to the selected rows, updated with RDSS's own recursion. Deterministic: the seed is unused, so the 5 repetitions share one selection and differ only in the FM seed. | γ = 0.5 (`gaussiankernel` default); α = 1 − 1/√B (RDSS paper and LUCoS default) |
| S-ZC | `zcore` | Constant columns are dropped (as LUCoS does). Each of `n_sample` iterations: draw `sample_dim` distinct columns and a point from the per-column triangular(min, median, max). The L1-nearest row (on those columns) gains +1. Its neighbours lose 1/d^exp, normalised to sum 1, over its `redund_nn` L1-nearest rows; if the nearest of these is at distance 0, that one row loses 1 instead. Scores start at U(0, 1) (`rand_init`), are min-max normalised and cast to float32; the selection is `argsort(−score)[:B]`. All draws come from one generator seeded by the selection seed. | n_sample = 10⁶, sample_dim = 2 (capped at d′), redund_nn = 1000, redund_exp = 4, rand_init = True (ZCore CLI defaults, identical to LUCoS's `ZCoreUnderSampler`) |

**Runs.** The same 20 DEV datasets, budgets (mult ∈ {1, 3, 5, 10}), repetition count (5) and seed rule (`SeedSequence([base_seed, did, mult, rep, ORDER index, 5])`) as the other strategies. Learners: FM2, FM1, FM5 and FM3, each with the LLGC secondary learner. Outputs go to separate files (`{did}_{tags}.parquet` for FM5, `{did}_{learner}_{tags}.parquet` otherwise); no existing file is touched.

**Analysis.** Exploratory: Δ against S-RR and against S-LUCoS for each strategy, by FM × budget, with Wilcoxon p-values (Holm within FM × budget over the strategies compared with the same reference), macro-F1 and accuracy. No confirmatory status changes.

**Flagged adaptations.**
- *All three: z-scored X instead of the official representations.* USL used self-supervised CLD/MoCo/CLIP image features; RDSS used CLIP features; ZCore used CLIP and ResNet features. For RDSS and ZCore, LUCoS's original space is unscaled.
- *USL, scaling.* The CLD features are unit-norm (`l2norm`). Dividing the z-scored X by √d′ gives a mean squared row norm of 1 and a mean squared pairwise distance of about 2, as on the unit sphere. This keeps the meaning of horizon = 4 and of the relative weight W without projecting low-dimensional tables onto a circle. Rankings in k-means and kNN are unchanged by the scaling.
- *USL, K.* The CIFAR value K = 400 is at least n for most DEV tables, so K = min(20, n) is used (USL's ImageNet setting, also TypiClust's).
- *USL, empty clusters.* In the official code an empty cluster gets a NaN centroid. CIFAR-2500 runs one extra centroid to absorb it, and the selection asserts exactly B picks. Here an empty cluster is dropped from later assignments.
- *USL, missing picks.* If fewer than B clusters are non-empty (unavoidable when X has fewer than B distinct rows), the missing picks are the unselected rows with the highest final score (density − W·reg).
- *USL, same-cluster mask.* The official code masks the selected row of the same cluster through `one_hot(cluster)`, assuming that the k-th pick comes from cluster k. Here the mask uses the cluster of each pick; the two agree whenever no cluster is empty.
- *USL, zero distances.* A zero squared distance to a pick in another cluster (official code: assertion failure) is treated as 1e10, as the ImageNet code does for self-distances.
- *RDSS.* A vectorised re-implementation of `khsample`: the full kernel matrix is computed once, using the same μ and the same F recursion.
- *ZCore, worker streams.* The official `zcore_score` splits n_sample across a `multiprocessing.Pool` whose workers draw from the global `np.random` state. Under the `fork` start method, which LUCoS's file-path module loading requires, all workers inherit the same state and return identical scores (checked on 2026-10-21 on this machine). LUCoS's 16 workers therefore ran 62,500 distinct iterations, each counted 16 times. Here the 10⁶ iterations are independent draws from one seeded generator, as the paper specifies.
- *ZCore, ties.* When the nearest neighbour is at distance 0, the official code takes `np.argsort(...)[1:]`, whose order among exact ties depends on numpy's sort. The same `np.argsort` is applied per row here, so the tie order matches numpy's. Batching the iterations changes only the floating-point summation order of the scores.
- *ZCore, repetitions.* LUCoS computes one score vector per dataset (n_runs = 1). Here the score depends on the selection seed, so each repetition and budget gets its own score vector.
- *Selection cache.* Selections are cached on disk by (hash of X, strategy, budget, seed), so all four FMs see identical rows without recomputing ZCore.

These hyperparameters are fixed now and never tuned on y.

## 9. Amendment 4 (2026-10-21): k-medoids bug fix and the raw-space k-medoids baseline (S-KMR)
Source: the LUCoS official-code check (`reports/LUCOS_OFFICIAL_REPRO.md`).

### 9.1 Bug fix in `strategies._kmedoids` (dated 2026-10-21)
**The bug.** When no initialisation ended with K distinct medoid indices, `_kmedoids` returned `zeros(K)`, i.e. row 0 repeated K times. This can happen only when rows repeat: two medoids that sit on identical rows can collapse onto one index during the medoid update. LUCoS's fallback to 10 initialisations could not repair it.

**The fix.** If, and only if, no initialisation succeeds:
- k-medoids (same seed and n_init) is run on the distinct rows, with K′ = min(K, n_distinct);
- the remaining K − K′ seats are filled by farthest point among the unselected rows, Euclidean, with ties broken by a draw from `default_rng([seed, 4])`, never by row order.

Every selection where some initialisation succeeded is bit-identical to before (regression test against the frozen pre-fix routine). `select_lucos` keeps its order: one initialisation, then 10 on failure, now with the fix.

**Impact on stored Study 4 selections.** The fixed code was re-run on every stored kmedoids, stratified_medoids and LUCoS selection (FM5 files). The only selections that changed are LED-display (40496), LUCoS, mult = 10, reps 2 and 3, which held 1 distinct row instead of 100. kmedoids and stratified_medoids are unchanged everywhere. Those two repetitions are recomputed for FM1, FM2, FM3 and FM5 (and their LLGC rows). The pre-fix files are kept in `superseded/`.

### 9.2 Strategy S-KMR (`kmedoids_raw`), appended to `ORDER`
- **What it is.** LUCoS's raw-space K-Medoids baseline (`OriginalSpaceEncoder` + `KMedoidsUnderSampler`). The official code applies no scaling and no encoding to the raw features; our DEV tables are all-numeric without missing values, so the raw X is that space.
- **Selection.** k-medoids with K = B on **unscaled** X, Euclidean, k-medoids++, **one** initialisation per repetition. If that initialisation ends with repeated medoids, the 10-init routine with the 9.1 fallback is used, exactly as for S-LUCoS.
- **Flagged adaptations.** The same as S-LUCoS in Amendment 2: our `_kmedoids` instead of sklearn-extra's "alternate" (plain D² seeding, at most 100 iterations, float64).
- **Motivation (post hoc, exploratory).** In the official pipeline, LUCoS ties this baseline at 1C. Our S-KM (z-scored, best of 10 initialisations) loses to it by +0.052 AUC at 1C.
- **Runs.** The same datasets, budgets, repetitions (5) and seed rule; FM2, FM1, FM5 and FM3 with LLGC; separate files `{did}[_{learner}]_kmedoids_raw.parquet`.
- **Analysis.** Exploratory. It is added to the selector report against random and against S-LUCoS. The LUCoS − S-KMR contrast replaces S-KM as the paper-faithful check of the representation's contribution.
