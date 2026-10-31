# P1 case study: re-evaluating TabEBM's few-label claim under a realistic labelling protocol (pre-registration)

**Status: FROZEN 2026-10-28.** Written 2026-10-28. The freeze is the local commit of this file. At the freeze no code exists for this study yet. No case-study table has been downloaded, loaded or fitted. Nothing here touches `sealed_heldout/`, `prereg/`, `reports/`, `results/`, `src/` or `paper/p1/main.tex`.

**Status in P1: EXPLORATORY.** The case study is not part of any confirmatory Holm family of Studies 7, 9, 12 or 13. In the paper it carries the marker "(exploratory)". It illustrates the consequence of the protocol on a published conclusion. It is not a further test of the P1 hypotheses.

## 0. What we already know (disclosure)

- We have read the paper's published numbers under its own protocol (Table 1 of the paper; Sec. 2 below). We have **not** seen any number for the case-study tables under a random, non-stratified draw, from any source.
- Table overlap with our own sets, checked on IDs and names before writing this file:
  - **HELD-OUT:** no overlap. None of the paper's eight OpenML IDs (40966, 14, <withheld>, 1504, 841, 1472, 40971, 40499) is in `research_notes/heldout_openml_ids_exclusion.txt`, and no name matches the HELD-OUT names in `data/pool_i2.yaml`.
  - **Study 14 T2 (frozen, not yet run):** two of the paper's tables are in, or share their rows with a table in, the T2 metadata draw (`data/t2_metadata_candidates_DRAFT.csv`; ids and names withheld in this copy, because T2 is the held-out set of a later study). **Both tables are excluded here,** so that the case study does not expose any T2 table.
  - **DEV:** stock (OpenML 841) is a DEV table. DEV is development data and may be reused; this is declared.
  - **T1′:** mfeat-fourier (OpenML 14) holds the same data as T1′ table `pmlb:mfeat_fourier`. T1′ has already been used by Studies 9 and 12 (and by the dose arms prereg of R001). Using it here does not affect a pending confirmation; this is declared.
- No author was contacted, and none will be.

## 1. The published comparison

- **Paper.** A. Margeloiu, X. Jiang, N. Simidjievski, M. Jamnik. "TabEBM: A Tabular Data Augmentation Method with Distinct Class-Specific Energy-Based Models." NeurIPS 2024. arXiv:2409.16118 (v2). Code: github.com/andreimargeloiu/TabEBM (Apache-2.0).
- **Claim (Sec. 3.1, citing Table 1 and Fig. 4 left), quoted:** "TabEBM is the only generator that consistently improves performance across sample sizes" (Margeloiu et al., 2024).
- **What the claim rests on.** Table 1 reports the downstream balanced accuracy, averaged over six predictors (logistic regression, KNN, MLP, random forest, XGBoost, TabPFN), for eight OpenML tables and N_real ∈ {20, 50, 100, 200, 500} real training rows. TabEBM has the best average rank (1.21). The Table 1 caption says "accuracy", but the text says balanced accuracy; we take balanced accuracy as the paper's metric and report accuracy as well (threat T6).
- **Their labelled-set protocol.** The paper writes "we first split it into stratified train and test sets" (N_test = min(N/2, 500)), subsamples N_real rows from the train set, and "split[s] each subset into stratified training and validation sets with a ratio of 4:1". Classes with fewer than 10 rows are removed first. Ten repetitions. The paper's code release has no experiment scripts, so we read the subsample as a proportional stratified draw (scikit-learn `train_test_split(..., stratify=y)`). It forces every class into the subset whenever N_real ≥ C. This is the coverage part of the P1 stratified protocol; it is not the balanced "m per class" variant (threat T1).
- **Why this paper.** NeurIPS paper; Apache-2.0 code on PyPI; public OpenML tables; a single, quotable headline; a macro-averaged metric (balanced accuracy = mean per-class recall); and a method whose design is per class (one energy-based model per class). A missed or one-row class is the case where the design should be most sensitive to the protocol. The cost fits on CPU (Sec. 7).

## 2. Scope: the few-label cells

We test the claim only where labels per class are few. We pre-declare the cells (table, N_real) with N_real / C ≤ 10 that Table 1 reports, after the exclusions of Sec. 0:

| Table (OpenML ID) | C | N_real | N_real / C | Paper Baseline | Paper SMOTE | Paper TabEBM |
|---|---|---|---|---|---|---|
| protein (40966) | 8 | 20 | 2.5 | 28.14 | N/A | 33.84 |
| protein (40966) | 8 | 50 | 6.3 | 50.72 | 54.52 | 55.91 |
| fourier (14) | 10 | 20 | 2.0 | 28.30 | N/A | 37.13 |
| fourier (14) | 10 | 50 | 5.0 | 53.69 | 55.51 | 56.57 |
| stock (841) | 2 | 20 | 10.0 | 78.75 | 82.18 | 83.49 |
| energy (1472) | 23 | 50 | 2.2 | 17.77 | N/A | 23.98 |
| energy (1472) | 23 | 100 | 4.3 | 25.94 | N/A | 31.24 |
| collins (40971) | 26 | 100 | 3.8 | 11.44 | N/A | 13.07 |
| texture (40499) | 11 | 50 | 4.5 | 72.40 | 76.40 | 78.90 |

Nine cells on five tables. In the paper, TabEBM beats the baseline in all nine.

**Reconstruction check (before any fit).** After loading and the paper's rule (drop classes with fewer than 10 rows), each table's (N, D, C) must equal the paper's Table 3: protein (1080, 77, 8), fourier (2000, 76, 10), stock (950, 9, 2), energy (698, 9, 23), collins (970, 19, 26), texture (5500, 40, 11). For energy and collins, the target and its discretisation are not documented. We try the OpenML default target, treated as categorical. If a table does not match, it is dropped from the primary analysis. The decision is logged in `out/reconstruction.csv` and committed before the first fit.

## 3. Methods, learners and devices

- **Augmentation arms (3):** Baseline (real rows only); SMOTE; TabEBM. The paper's other generators (TVAE, CTGAN, NFLOW, TabDDPM, ARF, GOGGLE, TabPFGen) are out of scope: most need a GPU to run at the paper's scale. So we do not test the "only generator" part of the claim. We test whether TabEBM "consistently improves" over the baseline, and whether it ranks first among the three arms.
- **TabEBM.** The released `TabEBM.py`, pinned to one commit of the repository (recorded at download), with the defaults of its `generate` function (200 SGLD steps, step size 0.1, SGLD noise 0.01, start noise 0.01, 4 surrogate negatives at distance 5). The current code uses a TabPFN v2 surrogate; the paper used a TabPFN v1 checkpoint (threat T2). The surrogate weights are the TabPFN v2 classifier checkpoint already cached for P1 (`tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`), passed explicitly. The code needs a `tabpfn` release that still exports `ModelInterfaceConfig`; P1's `tabpfn` 9.2.0 does not. So TabEBM runs in a separate virtual environment (Sec. 8), and the downstream TabPFN runs in the P1 environment.
- **Synthetic size.** N_syn = 500 per run, split over the classes seen in the real training rows in proportion to their counts (largest-remainder rounding). Classes not seen get no synthetic rows; TabEBM cannot model a class it has no row of.
- **SMOTE.** `imblearn.over_sampling.SMOTE` with `sampling_strategy` = {class: n_k + s_k}, where s_k are the same per-class synthetic counts as for TabEBM, and k_neighbors = min(5, n_k − 1). A class with n_k < 2 gets no synthetic rows. The paper reports SMOTE as N/A in four of our cells. We apply this fallback under both protocols and mark those cells "SMOTE fallback" (threat T3).
- **Downstream predictors (6), defaults as in the paper:** logistic regression, KNN, MLP and random forest (scikit-learn), XGBoost, and TabPFN v2 (P1's `tabpfn` 9.2.0, default classifier; the paper used TabPFN v1, threat T2). If a predictor cannot be fitted because the training rows contain only one class, it predicts that class (same rule for every arm and protocol).
- **Unseen classes.** A predictor can only predict the classes in its training rows (L0, as in P1 §4.2). Balanced accuracy is computed over the classes present in the test set, so a class that appears in the test set but in no training row has recall 0. Macro-F1 is computed over all C classes with `zero_division=0`.
- **Preprocessing.** Numeric features only (all six tables). Missing values (protein) are mean-imputed, and features are z-scored, both with statistics from the full training pool without labels. The same preprocessing is used for every arm and protocol.
- **Device.** CPU only. `OMP_NUM_THREADS=1`, `nice -n 15`, at most 6 worker processes (another run uses 6), one thread each.

## 4. Protocols (paired by repetition)

For each table and repetition r ∈ {0, …, 14}:

1. **Test split (shared).** A stratified train/test split with N_test = min(N/2, 500), seeded by `SeedSequence([20261028, did, r, 0])`. The same test rows score every arm, protocol and predictor of repetition r. This is the common evaluation set of R3-W4.
2. **S, the paper's protocol (replication).** From the train pool, `train_test_split(train_size=N_real, stratify=y)`; then a stratified 4:1 split into real training and validation rows. Seed `SeedSequence([20261028, did, r, N_real, 1])`. If scikit-learn raises, because some class has too few rows for a stratified split, the (cell, r) is logged as "S infeasible" and is not redrawn.
3. **R, the realistic protocol.** A uniform random N_real-subset of the train pool, without labels; then a random, non-stratified 4:1 split. Seed `SeedSequence([20261028, did, r, N_real, 2])`. **No draw is discarded or redrawn**, whatever classes it misses.
4. **Validation rows.** As in the paper, the validation rows are not used to train the generators or the predictors. So the training rows are 0.8 · N_real under both protocols (for example, 16 at N_real = 20).
5. **Model seeds.** TabEBM `seed` and predictor `random_state` = `SeedSequence([20261028, did, r, N_real, p, 3])`, with p the protocol index. Within a (cell, r, protocol), the three arms receive the same real rows.

15 repetitions per protocol (the paper used 10). We add 5 to narrow the per-cell intervals under R, whose variance is higher. The count drops to 10 if the timing pilot (Sec. 7) projects more than 70 CPU-hours.

## 5. Outcomes, statistics and decision rules

**Unit and estimand.** Per cell c, protocol P and repetition r: D_{c,P,r} = mean over the 6 predictors of [BA(TabEBM) − BA(Baseline)], in balanced-accuracy points (0–1 scale). Δ_P(c) = mean over r of D_{c,P,r}; its 95% interval is a percentile bootstrap over repetitions (10,000 resamples, seed 20261028). The protocol effect on the gain is G(c) = Δ_S(c) − Δ_R(c), with a paired bootstrap over r. Across cells, we report the mean over cells. Its interval comes from resampling repetitions within each cell; it is conditional on these nine cells and is not a population interval.

**Gate G0: replication under S.** The paper's claim is reproduced under S if Δ_S(c) > 0 in at least 8 of the evaluable cells of Sec. 2 (out of 9, or the share 8/9 rounded down if the reconstruction check drops cells), **and** the cell-mean Δ_S has an interval above 0. If G0 fails, the R results are reported descriptively only, and the case study concludes "not reproduced under the paper's own protocol in our implementation". No protocol attribution is made.

**Primary outcome (if G0 passes): does the conclusion "TabEBM consistently improves over real data alone" survive under R?** Let k_R be the number of evaluable cells with Δ_R(c) > 0.

- **Holds:** k_R ≥ 8 of 9, and the cell-mean Δ_R has an interval above 0.
- **Changes:** k_R ≤ 6 of 9, **or** the cell-mean Δ_R has an interval that includes or lies below 0, **or** at least one cell has an interval for Δ_R entirely below 0. In addition, for a "Changes" verdict the paper must attribute the change to the protocol: the cell-mean G must have an interval above 0. If this attribution condition fails, the verdict is "Weakened".
- **Weakened:** every other outcome. Example: k_R = 7, or the "Changes" condition holds but the attribution does not.

For the proportional rule, k_R ≥ 8 becomes ≥ ⌈8/9 · n_cells⌉ and k_R ≤ 6 becomes ≤ ⌊6/9 · n_cells⌋ if cells are dropped.

**Secondary outcome S1: the winner.** As in the paper's Table 1, for each cell, predictor and protocol, we rank the three arms by mean balanced accuracy over repetitions (ties share ranks) and average the ranks over cells × predictors. The winner changes if TabEBM has the lowest (best) average rank under S but not under R. We also report a Friedman test over the cell × predictor blocks for each protocol.

**Other secondary outcomes (descriptive):**
- macro-F1 over all classes (P1's metric) and accuracy, with the same Δ, G and verdict rules;
- the share of R draws whose training rows miss at least one class, per cell, next to the hypergeometric prediction (P1 Prop. 1);
- Δ per predictor (6) and per cell;
- the share of SMOTE fallbacks and of single-class training sets;
- an equivalence check for "Holds": TOST on the cell-mean G with the P1 margin of 0.01. If G is equivalent to 0, the text may say that the protocol does not change the gain.

**Wording rule.** The case study can say "the conclusion changes under the realistic protocol" only on a "Changes" verdict, "is weakened" only on "Weakened", and "holds" only on "Holds". Every statement names the metric (balanced accuracy), the nine cells and "our re-implementation of the paper's protocol".

## 6. Threats to validity

- **T1: what "stratified" means.** The paper does not publish its splitting code. We read the subsample as proportional stratification, not "m per class". On balanced tables (fourier, texture), the two are close; on imbalanced tables, the proportional version forces coverage but not balance. So the S-versus-R contrast here measures mainly the coverage term of P1. We say this in the text.
- **T2: model versions.** The TabEBM surrogate and the downstream TabPFN are TabPFN v2 here and v1 in the paper. So the S numbers need not match Table 1. G0 checks the direction of the paper's result, not its values. We report our S numbers next to the paper's.
- **T3: SMOTE where the paper reports N/A.** Our fallback makes SMOTE defined in every cell. The paper's SMOTE comparison is therefore not exactly reproduced in four cells. The primary outcome does not involve SMOTE.
- **T4: reconstruction of energy and collins.** Their targets and class counts may not be reproducible from OpenML. The pre-fit check (Sec. 2) drops a table that does not match, before any result exists.
- **T5: subset of generators.** We do not test the "only generator" part of the claim, and we do not claim to.
- **T6: metric ambiguity.** The Table 1 caption says "accuracy"; the text says balanced accuracy. We use balanced accuracy and also report accuracy.
- **T7: few cells.** Nine cells on five tables; the intervals are conditional on these cells. The case study shows what can happen to one published conclusion. It does not estimate how often conclusions change.
- **T8: test split per repetition.** The paper may reuse one test split across its ten repetitions. We draw one per repetition, shared by both protocols. This changes the variance, not the paired contrast.
- **T9: data exposure.** fourier (T1′) and stock (DEV) were already used by P1 studies; the T2 tables of Study 14 are excluded (Sec. 0); HELD-OUT is not involved.
- **T10: researcher degrees of freedom.** Everything above is fixed before any download. Changes after the freeze go into dated amendments at the end of this file, each committed before the run or analysis it affects.

## 7. Cost (CPU)

- **Fits.** 9 cells × 2 protocols × 15 repetitions = 270 runs. Each run makes one TabEBM generation, one SMOTE generation and 3 arms × 6 predictors = 18 predictor fits, so 4,860 predictor fits in total.
- **TabEBM.** It runs 200 SGLD steps per seen class, each a forward and backward pass of TabPFN on a context of n_k + 4 rows. Over the cells, C is 13.4 on average. At an assumed 0.15–0.25 s per step on one CPU thread, a run takes 6.7–11.2 minutes, so the 270 runs take **30–50 CPU-hours**. Under R, fewer classes are seen, which lowers the cost.
- **Predictors.** About 20 s per run (TabPFN v2 on ≤ 540 rows dominates), so about **1.5 CPU-hours**.
- **Total estimate: 32–52 CPU-hours.** Ceiling: 80 CPU-hours. On 6 workers, that is 6–9 wall-hours.
- **Timing pilot (before the run, after the freeze).** 20 TabEBM SGLD steps on synthetic data (`sklearn.datasets.make_classification`, no case-study table) measure the time per step. If the projection exceeds 70 CPU-hours, the repetitions drop from 15 to 10 (Sec. 4), and the change is recorded in an amendment before the run. If even 10 repetitions project above 80 CPU-hours, the run stops and the authors decide.

## 8. Software, data and approvals (nothing downloaded yet)

Downloads and installs need the authors' explicit approval. The list:

| Item | Source | Size (approx.) | Licence |
|---|---|---|---|
| TabEBM code (`TabEBM.py`; repo at a pinned commit, or the PyPI sdist `tabebm==2025.08.19`) | github.com/andreimargeloiu/TabEBM / pypi.org/project/tabebm | < 1 MB | Apache-2.0 |
| `tabpfn` release that exports `ModelInterfaceConfig` (separate venv, for TabEBM only; exact version chosen from PyPI release notes before install) | pypi.org/project/tabpfn | ~ 1 MB wheel; reuses the cached v2 weights | Prior Labs License (Apache-2.0 with an attribution clause) |
| `xgboost` (downstream predictor) | pypi.org/project/xgboost | ~ 2–5 MB (macOS arm64 wheel) | Apache-2.0 |
| `imbalanced-learn` (SMOTE) | pypi.org/project/imbalanced-learn | ~ 0.3 MB | MIT |
| OpenML 40966 MiceProtein v4 (ARFF) | openml.org/data/v1/download/17928620 | ~ 0.6 MB | OpenML "Public"; source UCI (CC BY 4.0) |
| OpenML 14 mfeat-fourier v1 | openml.org/data/v1/download/14 | ~ 1 MB | OpenML "Public"; source UCI (CC BY 4.0) |
| OpenML 1472 energy-efficiency v1 | openml.org/data/v1/download/1587929 | < 0.1 MB | OpenML "Public"; source UCI (CC BY 4.0) |
| OpenML 40971 collins v4 | openml.org/data/v1/download/17953251 | ~ 0.2 MB | OpenML "Public"; source StatLib |
| OpenML 40499 texture v1 | openml.org/data/v1/download/4535764 | ~ 2 MB | OpenML "Public"; source KEEL |
| OpenML 841 stock v2 | already cached (`data/cache/openml/.../841`) | none | OpenML "Public" |

No GPU, no LimiX, no HELD-OUT, no T2 table. TabPFN v3.5 is not used. The TabPFN v2 weights are already cached, so no weight download is needed.

## 9. Files

- Code (to be written after the freeze): `paper/p1/exploratory/case_study/` (`run_case_study.py`, `analyze_case_study.py`).
- Outputs: `paper/p1/exploratory/case_study/out/` (per-fit CSV with predictions, the reconstruction log, the timing pilot and the report).
- Paper: one short paragraph and one table in the appendix, marked "(exploratory)", added by the authors. `main.tex` is not changed by this study's runner.

## 10. Amendments

### Amendment 1 (2026-10-28, after the freeze R002; before any fit on a case-study table)

Written after the downloads, the code review, the reconstruction check (loading only, no fit) and the timing pilot (synthetic data only). No case-study outcome exists.

(a) **Validation split.** A literal stratified 4:1 split is infeasible in 6 of the 9 cells: scikit-learn needs at least C validation rows, but 0.2 · N_real < C for protein 20, fourier 20, energy 50 and 100, collins 100 and texture 50 (arithmetic from Table 3). Since the paper reports these cells, it cannot have used that split. Replacement, in both protocols: floor(0.2 · n_k) rows of each labelled class go to validation, chosen at random; the rest are training rows. This rule uses labels only after labelling, so it is available to a practitioner. The protocols now differ only in how the N_real rows are drawn.

(b) **Versions and code.** TabEBM commit 72eb78dab896c7a8f39c4dcc288c834fd72eff2b (`TabEBM.py` sha256 e5ed68e8af8e4b4362485471931718356b6a9d2760472719d3d8245688840f4a); tabpfn 2.1.2 (wheel sha256 3fcd04fdc44b037cab3098ae4f1fa2da48a0e9eaa01aab8ffe8eba120e9bf48a; the last release before the TabEBM commit, and it exports `ModelInterfaceConfig`); imbalanced-learn 0.12.4 (wheel sha256 d47fc599160d3ea882e712a3a6b02bdd353c1a6436d8d68d41b1922e6ee4a703; 0.13 needs `sklearn-compat`, which was not approved); xgboost 3.2.0 and the other dependencies from the local uv cache (no download); scikit-learn 1.5.2 (tabpfn 2.1.2 needs < 1.7); torch 2.14.1. TabEBM is called once per class, each time with a fresh instance: its fit cache is keyed by class index only, so reusing an instance would sample from a model fitted on another class. The class-k EBM uses only class-k rows, so per-class calls are equivalent to one call and allow the per-class counts of Sec. 3. `torch.load` is forced to `weights_only=True`, and outbound sockets are blocked in the TabEBM process.

(c) **TabPFN v2 downstream supports at most 10 classes.** So it is used only on protein, fourier and stock. On energy, collins and texture, the predictor mean of Sec. 5 is over the five other predictors, under both protocols.

(d) **Reconstruction check: all six tables match Table 3** (`out/reconstruction.csv`). Feature sets: protein drops MouseID, Genotype, Treatment and Behavior; collins drops Text, Genre, Counter and Corpus; energy keeps y2 (cooling load) as its ninth feature, which matches the paper's D = 9. stock is read from P1's cached npz of DEV table 841.

(e) **Timing pilot** (`out/pilot_timing.csv`; it ran alongside a 6-worker dose-arm run). One SGLD step took 0.16–2.19 s, not the 0.15–0.25 s assumed in Sec. 7. Model fitted to the pilot: t_step ≈ 0.16 + 2.75e-4 · D · s + 8.1e-4 · D (s = synthetic rows of the class). Projection: 4.25 CPU-hours per repetition and protocol, so **≈ 128 CPU-hours at 15 repetitions and ≈ 85 at 10**, plus about 2 for the predictors. By Sec. 7, 10 repetitions project above 80 CPU-hours, so **the run stops here and the authors decide.** No gate or main run has started.

### Amendment 2 (2026-10-28, before the gate or main run; no case-study fit exists)

The authors chose option (a) after Amendment 1 (e): **10 repetitions per protocol** (r = 0, …, 9), projected at ≈ 87 CPU-hours. **The authors approved this overrun of the 80 CPU-hour ceiling of Sec. 7.** Every other rule is unchanged: at most 6 workers, `nice -n 15`, one thread each; sockets blocked; `torch.load` with `weights_only=True`; one watcher on a done-file. Order: S first (the gate), then R. The gate decides only how R is read (Sec. 5), so R runs whatever the gate gives. **Machine deadline:** HELD-OUT opens after 2026-10-31 23:59 UTC. If the projected finish moves past 2026-10-31 18:00 UTC, the run stops cleanly and keeps its partial outputs, and no partial analysis is made.
