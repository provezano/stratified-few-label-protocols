# Study 9 on EXP′ = T1′: how many labels does a tabular FM need? (confirmatory)

Prereg: `prereg/STUDY9_HOW_MANY_LABELS.md` (frozen 2026-10-21; Amendment 1; freeze note 1 at R043), with `prereg/STUDY8_EXPANSION.md` A2.5–A2.7 and its trigger note (T1′ replaces Tier 1).
Data: T1′, 52 datasets (`data/pool_t1prime.yaml`). Runs: 4 FMs (FM2 TabICL, FM1 TabPFN v2, FM5 TabPFN v3.5, FM3 Kumo), 208/208 jobs, 0 failures, in `results/study9/t1prime/`.
Verbatim generated output: `reports/STUDY9_T1PRIME_OUTPUT.md`. Part A of that file is the frozen `analyze` + `report` output. Part B holds the pre-specified extras.

**Caveat on size.** N = 52, against the 100–200 the power analysis (Sec. 7) assumed. At N = 52, the minimum detectable Kendall τ is about 0.32, not 0.16–0.23. A non-significant moderator test here is weak evidence of absence.

## 0. Order of work and integrity

| Step | What | Commit / time (UTC) |
|---|---|---|
| 1 | HELD-OUT check script written and committed **before it was run** | R004 |
| 2 | HELD-OUT checks run (A2.6 hash + signature; A2.3a/b row overlap). Access logged (Appendix B) | 2026-10-27 21:30:31 |
| 3 | Outcome recorded as a dated note in `STUDY8_EXPANSION.md` **before any T1′ result was read**: no exclusion | R005 |
| 4 | Report driver committed **before the analysis was run** | R006 |
| 5 | Driver checks the SHA-256 of the frozen modules, then runs `analyze` unchanged | 2026-10-27 21:33 |

**Frozen code checksums** (they match freeze note 1):
- `src/simaudit/analysis/study9.py`: `aaf0bae1…3f6c50ba`;
- `src/simaudit/runner/study9.py`: `6fe40d00…87dd4c0`.

Neither file has changed since commit R003. The old wrapper `reports/study9_report.py` is not in the freeze, and it cannot read T1′ (it looks up `pool_ext.yaml`). So a new wrapper was written, `reports/study9_t1prime_report.py`, and it calls the frozen functions only.

### 0.1 HELD-OUT checks (A2.6, A2.3a/b)
- **Manifest.** `sealed_heldout/MANIFEST.json` lists 32 IDs, the same as the HELD-OUT IDs in `data/pool_i2.yaml`. All 32 npz files match their manifest SHA-256. The files were hashed as bytes, not parsed.
- **What was read.** Only the `X` array of each HELD-OUT npz. `y`, `name`, `version` and `sha` were **never read**. No HELD-OUT label or outcome was loaded or printed, and no model was run.
- **Hash.** No T1′ member's (X, y) content hash equals one of the 32 HELD-OUT hashes.
- **Row overlap** (the member's share of unique rows found in a same-d HELD-OUT table, after 6-significant-digit rounding or dense ranks; threshold 0.5). The maximum over all 52 × 32 pairs is **0.004** (pmlb:analcatdata_lawsuit). No exclusion.
- **Signature (n, C, majority, minority).** Equal signatures need equal n. **No T1′ member has the n of any HELD-OUT dataset**, so the check passes exactly and needs no HELD-OUT class counts. Had any n matched, the script was written to stop without reading labels and to refer the case to the authors.
- **Result:** `heldout_signature_check: passed`, and the row-overlap check passed (`data/t1prime_heldout_checks.yaml`).

### 0.2 Exclusions applied before any outcome
- **HELD-OUT dedup:** none.
- **A2.7 integrity screens:**
  - 18 candidates were excluded before the draw (S3 conflicting duplicates, or A2.3a/b row overlap). They never entered T1′.
  - S2 found no leaky column, so `drop_columns` is empty for all 52 members.
  - S1 (sorted rows) only raises a flag; the runner's π_d permutation (namespace 13) neutralises it.
- **Analysis set:** the 52 datasets exactly. The result keys equal the pool keys, as the driver checks.

## 1. Confirmatory results (EXP′, FM2 primary, macro-F1, ε = 0.01)

### Holm family (9 tests, α = 0.05)
| Test | Statistic | p | p_Holm | Verdict | FM1 (FM-robust?) |
|---|---|---|---|---|---|
| H1b: β₁ (log C) > 0 | β₁ = 0.137 (SE 0.220) | 0.267 | 1 | **not supported** | β₁ = 0.115, p = 0.33 |
| H2a S-KMn: P(ρ(1) > 1) > ½ | 27/52 = 0.52 | 0.445 | 1 | **not supported** | 0.48, p = 0.66 |
| H2a S-LUn | 25/52 = 0.48 | 0.661 | 1 | **not supported** | 0.52, p = 0.44 |
| H2a S-ST (stratified) | 45/52 = 0.87 | 3.5e-8 | 3.1e-7 | **supported** | 0.90, p = 6.4e-10 → **FM-robust** |
| H2b S-KMn: ρ(1) > ρ(5) | 18/37 decided | 0.629 | 1 | **not supported** | 21/38, p = 0.31 |
| H2b S-LUn | 19/37 decided | 0.500 | 1 | **not supported** | 22/38, p = 0.21 |
| H3b: τ(signed error, log IR) < 0 | τ_b = −0.481 | 4.4e-7 | 3.5e-6 | **supported** | (no FM involved) |
| H4 A (agreement): R < 1 | R = 1.006 [0.998, 1.015] | 0.934 | 1 | **not supported: documented negative** | R = 1.008, p = 0.98 |
| H4 D (FM1–FM2 disagreement) | R = 1.007 [0.998, 1.019] | 0.929 | 1 | **not supported: documented negative** | see 2.4 |

### Estimation decisions (CI thresholds)
**H1-dist** (Turnbull NPMLE; FM2, random; N = 52):
- **Median m\* in (7, 10]** labels per class; the IQR runs from the cell (5, 7] to the cell (10, 15].
- P(m\* ≤ 5) = 0.21 [0.12, 0.33]; P(m\* ≤ 10) = 0.62 [0.49, 0.75]; P(m\* > 20) = 0.08 [0.02, 0.16]. These CIs use 2000 resamples (Part B).
- F̂ at the grid points 1, 2, 3, 5, 7, 10, 15, 20, 30: 0.02, 0.02, 0.10, 0.21, 0.38, 0.62, 0.82, 0.92, 0.96.
- **DEV prediction confirmed:** the median lies in (5, 20], and the IQR spans more than two grid steps. There is no universal number.
- *Discrepancy, reported rather than corrected.* The frozen `h1()` calls `distribution_summary` with its default of 500 resamples, but Sec. 3 says 2000. The Part A table therefore uses 500. Rerunning the same frozen function with `n_boot = 2000` (Part B, B1) changes only the second decimal of one CI bound.

**H1a** (ICC(2,1) of the m\* category, FM1 vs FM2, the 49 full-grid datasets): ICC = 0.770 [0.609, 0.879] → **unresolved**. The lower bound is below 0.75 and the upper bound is above it. The saturation category agrees well across FMs, but "good reliability" is not established.

**H3a** (|log(m̂95 / m95)| ≤ log 1.5): share = 0.50, Clopper–Pearson [0.36, 0.64] → **refuted**, since the upper bound is below 0.75. The label-free k-means budget is within 1.5× of the oracle for only half of the datasets.

### Magnitudes (H2a, pre-specified retention rules; Turnbull median of ρ_S(1), 2000-resample CI)
- **S-ST: 1.95 [1.57, 2.17].** By the letter of the rule, the "≈ 2.6×" claim is *retained*, because the CI intersects [2.0, 3.2]. But the point estimate is 1.95, below 2.0. The finding to report is "≈ 2×".
- **S-KMn: 1.04 [1.00, 1.45]** → "≈ 2×" **not retained**.
- **S-LUn: 1.00 [1.00, 1.32]** → **not retained**.
- For both selectors, about half of the ρ(1) intervals are left-censored: the selector does no better at 1C than random at 1C.

## 2. Pre-specified secondary and sensitivity results (no confirmatory weight)

### 2.1 H1
**H1-dist sensitivities** (B1). The median cell is (7, 10] in every subset below:
- tiers 2a, 3 and 4 separately; the numeric-only subset; the balanced core;
- post-stratified (equal weight per stratum; the weighted EM is flagged new code);
- FM1, FM3 and FM5;
- S-KMn and S-LUn;
- accuracy;
- C2 and C5–10.

The exceptions:

| Subset | Median cell |
|---|---|
| S-ST | (5, 7] |
| C3–4 (n = 9) | (15, 20] |
| ε = 0.005 | (10, 15] |
| ε = 0.02 | (3, 5] |

ε is the dominant lever: P(m\* ≤ 10) is 0.31 at ε = 0.005 and 0.98 at ε = 0.02.

**Censoring of m\*.** 1 dataset is left-censored and 5 are right-censored (C2: 3; C3–4: 2).

**H1b replications** (B3). β₁ ranges from 0.06 to 0.21 over FM1, FM3, FM5, S-ST, S-KMn and S-LUn, and every p is ≥ 0.20. The C moderator is not detected anywhere. The IR, n and d coefficients are all near 0.

**H1a secondaries.**
- Four-FM ICC: 0.747 [0.653, 0.818] → unresolved.
- Pairwise ICCs: 0.71–0.81, all unresolved.
- Flagged lower-bound variant on all 52 datasets: 0.758 [0.601, 0.872].

**Coverage floor.** m\* ≥ m95 holds (decided) for 33 of 52 datasets, fails for 13, and is undecided for 6. Saturation mostly comes after full class coverage.

**Selection ceiling** (S-KMn and S-LUn under FM2). The median cell of m_sel\* is (30, ∞]. In 94% and 84% of datasets respectively, |f_sel − f_RR| ≥ ε persists past 20C.
- Note: this measures a *difference* of either sign, not an advantage. Taken with H2, the selector and random curves stay apart by ≥ 0.01 somewhere at high m, in both directions, rather than converging.
- The LUCoS (≈ 8C) and TCM (≈ 20C) anchors are not reproduced as convergence points.

### 2.2 H2
- **ρ medians at m = 1, 2, 3, 5, 7, 10, macro-F1 (FM2):**
  - S-ST: 1.95, 1.38, 1.31, 1.21, 1.29, 1.09; it fades with m;
  - S-KMn: 1.04, 1.19, 0.91, 1.17, 0.87, 1.10;
  - S-LUn: 1.00, 1.24, 1.40, 1.20, 1.27, 1.21.
- **Accuracy reverses the selector picture (B6, exploratory).**

  | Strategy | ρ(1) > 1 | Sign-test p | Median ρ(1) | Fade test, ρ(1) > ρ(5) |
  |---|---|---|---|---|
  | S-KMn | 0.79 | 1.8e-5 | 2.49 | p = 0.008 |
  | S-LUn | 0.73 | 6.0e-4 | 2.31 | p = 0.004 |
  | S-ST | 0.52 | 0.44 | 1.03 | — |

  - Under accuracy, the label-free selectors are worth about 2.3–2.5× random at 1C, and the advantage fades by 5C.
  - The stratified advantage is a macro-F1 effect: balanced labels help the minority classes.
  - This is the protocol-bias mechanism of Study 7. It is not a confirmatory result, because the primary metric is macro-F1.
- **Not computed (pre-specified, still pending):**
  - the Kendall τ of log ρ_S(1) with the moderators;
  - the coverage-equivalence ratio.

### 2.3 H3
- **H3c** (the learnable variant, m̂95_learn): share = 0.48.
- **Unsafe rate** (m̂95 < m95 / 1.5) by IR bin:

  | IR bin | n | Unsafe rate | Median p_cov at the recommended budget |
  |---|---|---|---|
  | IR < 2 | 29 | 0.00 | 0.98 |
  | IR 2–5 | 11 | 0.36 | 0.91 |
  | IR > 5 | 12 | 0.67 | 0.74 |

  The extremes are allbp, with p_cov = 0.07 (IR 258), and car_evaluation, with 0.19. This is the pre-specified failure mode, confirmed (H3b).
- **Over-prediction rate:** 0.27. It mostly comes from k-means isolating small clusters on balanced data, e.g. spambase (m̂95 = 201 against 3.5), Hill_Valley and segmentation.
- **p_min sensitivity.** Inflating the rarest share by 1.5× gives a median achieved p_cov of 0.94 (min 0.85); 2× gives a median of 0.98 (min 0.76).
- **Validity of the exact p_cov.** Against the realized full-coverage share of the 10 nested random draws (463 cells), the mean predicted value is 0.821 against 0.835 realized; mean |diff| 0.044; r = 0.96.

### 2.4 H4
- **FM1 as the primary learner.** Signal A: R = 1.008, p = 0.98. Signal D: R = 0.981 [0.970, 0.991], bootstrap p < 0.001. With FM1 as primary, the disagreement signal reduces the error of the mean-gain baseline by about 2%.
  - This is the replication direction for D. Because FM2-primary D is not Holm-significant, D is **not** FM-robust or confirmed.
  - Even if real, a 2% reduction in MAE has no practical value.
- **Not computed (exploratory, pending):**
  - κ / SP, MES and entropy signals;
  - the coverage-deficit signal;
  - the meta-learned prior;
  - the decision analysis against fixed 5C/10C rules;
  - H4 for S-KMn and S-LUn.

### 2.5 Not run
- **Tier 1 bridge** (S-KMn/S-LUn against the per-budget S-KM/S-LUCoS of Study 7): needs the Study 7 extension on T1′, which has not been run.
- **HELD-OUT replication (Sec. 9):** not run; that is the separate, later step.

## 3. Reading
1. **No magic number, as predicted.** A typical tabular FM saturates (macro-F1 slope below 0.01 per extra label per class) at **7–10 labels per class**. The median is the same across four FMs, three sources and three label strategies. The spread is wide: 21% of datasets saturate by 5 and 8% are still gaining past 20. The threshold ε moves the answer more than any dataset trait.
2. **The C moderator (H1b) is not detected.** Saturation measured in labels *per class* does not grow detectably with C. This is weak evidence at N = 52, with only 17 multiclass datasets.
3. **The stratified protocol is worth about 2× random labels at 1C in macro-F1.** This is confirmatory and FM-robust. It quantifies the free-label advantage of the stratified protocol.
4. **Label-free selectors (nested k-medoids, LUCoS embeddings) give no macro-F1 advantage over random at 1C** (H2a, not supported), so there is no fade to test (H2b). In accuracy they look like ≈ 2.4× random (exploratory). The selector benefit is metric-dependent.
5. **Label-free coverage budgets are unreliable** (H3a refuted). Their failure is in the unsafe direction under imbalance, as pre-specified (H3b supported, τ = −0.48).
6. **Stopping (H4): a documented negative.** Neither stop-set signal beats the mean gain per transition in this regime (R ≈ 1.006).

## Appendix A. Dataset key mapping (result file name → T1′ member)
Result files are named `{dataset_key(did)}_{learner}.parquet`, and `dataset_key` writes `pmlb_{-did}` for every negative ID. UCI members have did = −(10⁹ + UCI ID). So **`pmlb_1000000XXX` = UCI dataset XXX**, not PMLB. The other `pmlb_<hash>` keys are PMLB datasets. Source tiers: 2a = PMLB numeric, 3 = PMLB integer-coded categoricals, 4 = UCI.

| file key | pool key | name | tier | stratum | n | balanced core |
|---|---|---|---|---|---|---|
| pmlb_402875677 | pmlb:Hill_Valley_without_noise | Hill_Valley_without_noise | 2a | C2 × IR<2 | 1212 | yes |
| pmlb_1000000267 | uci:267 | Banknote Authentication | 4 | C2 × IR<2 | 1372 | yes |
| pmlb_776526808 | pmlb:glass2 | glass2 | 2a | C2 × IR<2 | 163 | yes |
| pmlb_1000000451 | uci:451 | Breast Cancer Coimbra | 4 | C2 × IR<2 | 116 | yes |
| pmlb_1000000545 | uci:545 | Rice (Cammeo and Osmancik) | 4 | C2 × IR<2 | 3810 | yes |
| pmlb_899643728 | pmlb:spambase | spambase | 2a | C2 × IR<2 | 4601 | yes |
| pmlb_612078220 | pmlb:sonar | sonar | 2a | C2 × IR<2 | 208 | yes |
| pmlb_70999782 | pmlb:Hill_Valley_with_noise | Hill_Valley_with_noise | 2a | C2 × IR<2 | 1212 | yes |
| pmlb_771782289 | pmlb:prnn_crabs | prnn_crabs | 2a | C2 × IR<2 | 200 | no |
| pmlb_190478228 | pmlb:threeOf9 | threeOf9 | 2a | C2 × IR<2 | 512 | no |
| pmlb_196466410 | pmlb:analcatdata_boxing2 | analcatdata_boxing2 | 2a | C2 × IR<2 | 132 | no |
| pmlb_158349848 | pmlb:bupa | bupa | 2a | C2 × IR<2 | 345 | no |
| pmlb_321653624 | pmlb:corral | corral | 2a | C2 × IR<2 | 160 | no |
| pmlb_1000000850 | uci:850 | Raisin | 4 | C2 × IR<2 | 900 | no |
| pmlb_629865933 | pmlb:saheart | saheart | 2a | C2 × IR<2 | 462 | no |
| pmlb_891171204 | pmlb:congressional_voting_records | congressional_voting_records | 3 | C2 × IR<2 | 435 | no |
| pmlb_696605717 | pmlb:chess | chess | 3 | C2 × IR<2 | 3196 | no |
| pmlb_140070728 | pmlb:molecular_biology_promoters | molecular_biology_promoters | 3 | C2 × IR<2 | 106 | no |
| pmlb_216388530 | pmlb:tic_tac_toe | tic_tac_toe | 3 | C2 × IR<2 | 958 | no |
| pmlb_471213511 | pmlb:irish | irish | 3 | C2 × IR<2 | 500 | no |
| pmlb_51679336 | pmlb:tokyo1 | tokyo1 | 3 | C2 × IR<2 | 959 | no |
| pmlb_452957568 | pmlb:GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1 | GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1 | 3 | C2 × IR<2 | 1600 | no |
| pmlb_298032972 | pmlb:biomed | biomed | 3 | C2 × IR<2 | 209 | no |
| pmlb_285622297 | pmlb:ionosphere | ionosphere | 3 | C2 × IR<2 | 351 | no |
| pmlb_33590086 | pmlb:mofn_3_7_10 | mofn_3_7_10 | 2a | C2 × IR2-5 | 1324 | yes |
| pmlb_1000000519 | uci:519 | Heart Failure Clinical Records | 4 | C2 × IR2-5 | 299 | yes |
| pmlb_148263816 | pmlb:haberman | haberman | 2a | C2 × IR2-5 | 306 | yes |
| pmlb_675884895 | pmlb:breast_cancer | breast_cancer | 3 | C2 × IR2-5 | 286 | yes |
| pmlb_536739196 | pmlb:hepatitis | hepatitis | 3 | C2 × IR2-5 | 155 | yes |
| pmlb_760627086 | pmlb:profb | profb | 3 | C2 × IR2-5 | 672 | yes |
| pmlb_147524437 | pmlb:analcatdata_lawsuit | analcatdata_lawsuit | 2a | C2 × IR>5 | 264 | yes |
| pmlb_1000000563 | uci:563 | Iranian Churn | 4 | C2 × IR>5 | 3150 | yes |
| pmlb_112947990 | pmlb:hypothyroid | hypothyroid | 3 | C2 × IR>5 | 3163 | yes |
| pmlb_52156669 | pmlb:churn | churn | 3 | C2 × IR>5 | 5000 | yes |
| pmlb_814233394 | pmlb:backache | backache | 3 | C2 × IR>5 | 180 | yes |
| pmlb_844895957 | pmlb:cloud | cloud | 2a | C3-4 × IR<2 | 108 | yes |
| pmlb_1000000212 | uci:212 | Vertebral Column | 4 | C3-4 × IR2-5 | 310 | yes |
| pmlb_842065541 | pmlb:new_thyroid | new_thyroid | 2a | C3-4 × IR2-5 | 215 | yes |
| pmlb_403982248 | pmlb:schizo | schizo | 2a | C3-4 × IR2-5 | 340 | yes |
| pmlb_1000000697 | uci:697 | Predict Students' Dropout and Academic Success | 4 | C3-4 × IR2-5 | 4424 | yes |
| pmlb_512854293 | pmlb:cars | cars | 3 | C3-4 × IR2-5 | 392 | yes |
| pmlb_153261989 | pmlb:splice | splice | 3 | C3-4 × IR2-5 | 3188 | yes |
| pmlb_570174309 | pmlb:car_evaluation | car_evaluation | 3 | C3-4 × IR>5 | 1728 | yes |
| pmlb_434692359 | pmlb:allbp | allbp | 3 | C3-4 × IR>5 | 3772 | yes |
| pmlb_352495961 | pmlb:led24 | led24 | 2a | C5-10 × IR<2 | 3200 | yes |
| pmlb_211571049 | pmlb:mfeat_fourier | mfeat_fourier | 2a | C5-10 × IR<2 | 2000 | yes |
| pmlb_1000000763 | uci:763 | Land Mines | 4 | C5-10 × IR<2 | 338 | yes |
| pmlb_197305078 | pmlb:segmentation | segmentation | 3 | C5-10 × IR<2 | 2310 | yes |
| pmlb_1000000257 | uci:257 | User Knowledge Modeling | 4 | C5-10 × IR>5 | 403 | yes |
| pmlb_611798166 | pmlb:yeast | yeast | 2a | C5-10 × IR>5 | 1479 | yes |
| pmlb_13510280 | pmlb:ecoli | ecoli | 2a | C5-10 × IR>5 | 327 | yes |
| pmlb_203521824 | pmlb:dermatology | dermatology | 3 | C5-10 × IR>5 | 366 | yes |

## Appendix B. HELD-OUT access log (`sealed_heldout/`)

Interactive, before the script ran (orientation, by the operator; directory listings and the manifest only):
```
2026-10-27T21:28:28Z  sealed_heldout/              | directory listing (ls -la)
2026-10-27T21:28:28Z  sealed_heldout/MANIFEST.json | printed (IDs, sealed_at, reason, npz SHA-256; no data)
2026-10-27T21:28:28Z  sealed_heldout/npz/          | file-name listing
2026-10-27T21:28:28Z  sealed_heldout/openml/       | first 5 directory names listed
```
Script `tools/heldout_dedup_t1prime.py` (commit R004), copied from `data/t1prime_heldout_access_log.jsonl`:
```
2026-10-27T21:30:31.253+00:00 sealed_heldout/MANIFEST.json | read manifest (JSON: IDs and npz SHA-256)
2026-10-27T21:30:31.253+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.254+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.254+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.254+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.255+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.255+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.256+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.256+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.257+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.257+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.257+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.257+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.258+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.258+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.258+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.258+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.259+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.259+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.260+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.260+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.260+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.260+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.261+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.261+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.261+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.262+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.262+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.262+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.262+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.263+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.263+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.263+00:00 sealed_heldout/npz/<withheld>.npz | SHA-256 of file bytes (not parsed)
2026-10-27T21:30:31.263+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.264+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.265+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.265+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.265+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.265+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.265+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.265+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.265+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.266+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.266+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.266+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.266+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.266+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.266+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.267+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.267+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.267+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.267+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.267+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.267+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.267+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.268+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.268+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.268+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.268+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.268+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.268+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.268+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.269+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.269+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
2026-10-27T21:30:31.269+00:00 sealed_heldout/npz/<withheld>.npz | read array 'X' only (y, name, version, sha not read)
```

No other file under `sealed_heldout/` was opened. `sealed_heldout/openml/` was not read beyond its listing. No HELD-OUT `y` was read at any point.
