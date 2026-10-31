# E17: full metric panel for the protocol gap (EXPLORATORY, not pre-registered)

Stratified minus random labelled sets, scored with every relevant classification
metric from the stored per-query probabilities of Study 7 (DEV, 20 tables) and
Study 9 (T1′, 52 tables). No new fits. Nothing under results/, reports/, prereg/
or sealed_heldout/ is read for writing or touched. This is an exploratory check;
no result here is confirmatory.

## 1. Predictions (written BEFORE any E17 metric was computed)

Time stamp of this section: **2026-10-28T02:27:16Z** (2026-10-27 23:27 -03).
It is committed on its own, before the script is run (see `git log` of this file).

Honesty note. Before writing this section I had read the paper draft, the two theory
reports and earlier exploratory summaries, so I already knew the stored results
for macro-F1, accuracy, balanced accuracy and the two AUC conventions (E2). The
predictions for those metrics are therefore not blind. The predictions for
kappa/MCC, F-beta, macro precision, Jaccard, weighted F1, top-2 accuracy,
average precision, log loss, Brier, D² scores, calibration (ECE, CITL, KL) and
the binary likelihood ratios are made without having seen any of those numbers.

Orientation everywhere: gap > 0 means stratified looks better (for losses and
calibration errors the gap is random − stratified).

Missing-class rule assumed (L0, as in the paper and in
`selection.strategies.expand_proba` / `runner.study9.predict_on`): the FM is fitted
on the seen classes only; an unseen class gets probability 0; a context with a
single class predicts that class with probability 1.

Mechanisms (TPB = THEORY_PROTOCOL_BIAS.md, TLB = THEORY_LABEL_BUDGET.md):
- coverage (TPB A, TLB T3–T5, T7): random draws miss classes; at 1C the whole gap
  is coverage (T7). A missed class costs ~1/C in a class-averaged metric and ~π_k in
  a prevalence-weighted one (T5); absorption (A7) makes accuracy nearly immune.
- balance (TPB B): a stratified context carries a uniform prior; a random one an
  unbiased estimate of the pool prior. For a prior-sensitive (Bayes plug-in) learner
  the uniform prior is BA-optimal (B2), the true prior accuracy-optimal (B3), and
  macro-F1 lies in between (B4, B5).

| Family | Metrics | Prediction at 1C | Prediction at ≥3C | Imbalance / C |
|---|---|---|---|---|
| F1 prevalence-weighted decision | accuracy (= micro-F1 = 1 − hamming = 1 − zero-one), top-2 accuracy (C ≥ 3), weighted F1 | small positive (absorption, T5: ≤ 1/(e m)); weighted F1 somewhat larger than accuracy (a missed class enters with F1 = 0 times its share) | near zero, mixed sign; negative on imbalanced pools for prior-sensitive FMs (B3) | gap decreases (more negative) with imbalance; top-2 gap ≈ 0 except via coverage |
| F2 chance-corrected | Cohen's kappa, MCC | clearly positive and close to macro-F1 in size: a constant prediction (single-class context) scores exactly 0 | positive, between accuracy and macro-F1 | positive gap grows with imbalance less than macro-F1 |
| F3 class-averaged decision | BA, macro-F1, F0.5, F2, macro precision, macro recall, macro Jaccard | strongly inflated (coverage, ≈ f·(1/C)Σq_k) | inflated (balance), BA ≥ F2 ≥ macro-F1 ≥ F0.5 ≥ macro precision; macro precision may be ≈ 0 or negative on imbalanced pools (uniform prior over-predicts the minority and lowers its precision) | grows with imbalance; at 1C grows with C (toward (1−1/C)^C) |
| F4 ranking, all classes (unseen class = constant score → AUC 0.5, single-class context → 0.5) | AUC OvR macro, OvR weighted, OvO macro, AP macro | inflated; OvR weighted less (missed classes are rare); AP macro inflated (missed class AP = its prevalence) | ≈ 0: AUC is invariant to a prior shift for C = 2 and nearly so for C > 2 | only the coverage part, so it shrinks fast with m |
| F5 ranking, seen classes only (prereg convention; NaN for a single-class context) | same four | ≈ 0 (binary: only covered draws remain, Lemma 1), small for multiclass | ≈ 0, mixed sign | no clear dependence |
| F6 proper scoring rules / skill | log loss, multiclass Brier, D² log loss, D² Brier | log loss strongly positive (a missed class has p = 0, clipped at 1e-6 → 13.8 nats per row: a convention artefact); Brier small positive (bounded, prevalence-weighted like accuracy) | **negative** on imbalanced pools (uniform context prior is miscalibrated for the test prior), ≈ 0 or slightly positive on balanced pools | gap decreases with imbalance; D² scores follow their losses |
| F7 calibration | top-label ECE (15 bins), classwise ECE, CITL (L1 of mean predicted minus true share), KL(test prior ‖ mean predicted) | positive (random has mean share 0 on missed classes) | **negative** on imbalanced pools: the mean predicted distribution is pulled toward uniform under S; random's prior is unbiased | negative gap grows with imbalance; ECE weakest |
| F8 binary likelihood ratios (positive = pool minority class) | log LR+, −log LR− | positive (constant predictions give LR ≈ 1) | uncertain; LR− likely better under S (higher minority recall), LR+ mixed | — |

Further predictions:
- FM rank flips (stratified vs random ranking of the 4 FMs per table and budget):
  most frequent where the gap depends on prior sensitivity (calibration, log loss,
  Brier, macro-F1, BA), least for all-class AUC and accuracy-type metrics.
- Correlation of gaps across cells: three clusters — (i) class-averaged + chance-corrected
  + all-class ranking (coverage-dominated), (ii) prevalence-weighted + proper scores
  (accuracy-like), (iii) calibration (prior-match) — with seen-class AUC loosely attached
  to (ii). The eight theoretical families will not be separated cleanly.

---

## 2. Results (computed after Section 1 was committed; time stamp 2026-10-28T03:03:34Z)

Script: `e17_metrics.py` (run as `OMP_NUM_THREADS=1 nice -n 15 .venv/bin/python paper/p1/exploratory/e17_metrics/e17_metrics.py`; seed 20261027; 10 000 bootstrap resamples).
Data: DEV = Study 7 Part (a), 20 tables × 4 FMs, m ∈ {1,3,5,10}, 5 stratified / 20 random draws.
T1′ = Study 9, 52 tables × 4 FMs, m ∈ {1,2,3,5,7,10,15,20,30}, 5 stratified / 10 random draws. Stratified is
only feasible while m < smallest pool class, so n drops to 44 tables at 20C and 40 at 30C. Only stratified and random draws are scored: 35 420 fits in total.

### 2.1 Checks
- **Reproduction.** For every stratified/random row, the stored macro_f1, accuracy, balanced_accuracy, log_loss and
  (seen-class) auc were recomputed twice, with the frozen `analysis.study7.complement_metrics` / `metric_set` and with
  E17's own code. Max abs error: 0 with the frozen code; 1.8e-15 with E17's code (`e17_reproduction.csv`).
- **sklearn 1.9.1** (230 seeded draws, `e17_sklearn_check.csv`): every decision metric, kappa, MCC, log loss, Brier,
  D² Brier, D² log loss, top-2 and AP agree to ≤ 2e-15. Micro-F1 and 1 − Hamming equal accuracy exactly.
  The binary AUC differs from `roc_auc_score(y, p1)` by up to 1e-3. This is not an error. The paper's convention averages the AUC of both columns, and with
  float32 probabilities p0 ≠ 1 − p1 exactly, so near-ties rank differently. E17 keeps the paper's convention.
- **Missing-class rule** (the paper, Sec. 3, and L0; `selection.strategies.expand_proba`, `runner.study9.predict_on`): fit on the seen classes,
  unseen class probability 0, and a single-class context predicts that class with probability 1. Macro metrics run over all C classes with an
  unseen class = 0. The pre-registered AUC uses the seen classes and is NaN for a single-class context. Its gap uses table means of the defined draws,
  as in the paper.
- Share of random draws that miss a class: DEV 0.66/0.18/0.05/0.01 at 1/3/5/10C; T1′ 0.63/0.21/0.11/0.05 at 1/3/5/10C.

### 2.2 The metric map (pooled over FMs; `e17_gap_tests.csv`, figure `e17_metric_map.{pdf,png}`)
Gap > 0 means stratified looks better. Values are the mean over tables, T1′ first and DEV in brackets; * means Holm p < 0.05 within the family.

| Family | 1C | 3C | 10C | Verdict |
|---|---|---|---|---|
| F3 class-averaged (BA, macro-F1, F0.5, F2, recall, Jaccard, precision) | +0.06 to +0.14* (+0.08 to +0.18*) | +0.04 to +0.07* | BA/recall +0.058*, F2 +0.041*, F1 +0.034*, F0.5 +0.027*, precision +0.003 | **inflated** at every budget (BA/recall significant through 30C) |
| F2 chance-corrected (kappa, MCC) | +0.095* (+0.11–0.12*) | +0.06–0.07* | +0.04* | **inflated**, same size as macro-F1 |
| F4 ranking, all classes (AUC OvR/OvO/weighted, AP) | +0.07 to +0.08* (+0.09–0.10*) | +0.03–0.04* | +0.01–0.02 (n.s.) | **inflated**, mostly coverage; fades by 10C |
| F1 prevalence-weighted (accuracy, top-2, weighted F1) | acc −0.014 (+0.015); wF1 +0.055*; top-2 +0.05 | acc −0.017; wF1 +0.028* | acc −0.018; wF1 +0.008 | **accuracy: unchanged on average with mixed signs**; weighted F1 inflated at small budgets |
| F5 ranking, seen classes | −0.002 to −0.02 | ≈ +0.005 | ≈ +0.01 | **unchanged** (no test survives Holm) |
| F6 proper scores (log loss, Brier, D²) | log loss +2.33* nats, Brier +0.096* | log loss +0.29*, Brier −0.009 | log loss −0.003, Brier −0.019 | **inflated at 1–3C** (coverage via p = 0), then **reverses** |
| F7 calibration (CITL, KL, ECE, classwise ECE) | CITL +0.18*, KL +2.33*, ECE +0.08* | ≈ 0 | CITL −0.077, classwise ECE −0.029 | **inflated at 1C, then stratified worse** (on imbalanced pools, see 2.3) |
| F8 binary LR (35 / 14 binary tables) | LR− +0.28*, LR+ −0.04 | LR− +0.25*, LR+ −0.23* | LR− +0.41*, LR+ −0.34* | **split**: LR− better, **LR+ worse** under stratified |

Mean values of the D² Brier score are dominated by a few very imbalanced tables, where the null Brier 1 − Σπ² is tiny (one table has IR = 261).
Use the median or the rank-biserial (the figure shows both).

### 2.3 Imbalance and number of classes (`e17_moderators.csv`, `e17_imbalance_halves.csv`)
Imbalance is the TV distance of the pool prior from uniform, with a median split per set (T1′: 26 + 26 tables). The values below are medians for the more imbalanced half of T1′ at 3/10/20C:
- **Calibration: stratified is worse.** CITL −0.09*/−0.21*/−0.19* (rank-biserial −0.69/−0.97/−0.98). Classwise ECE −0.03*/−0.05*/−0.07*. KL −0.04* at 20C.
  In the balanced half the same metrics favour stratified (CITL +0.12*/+0.05*/+0.03*).
  Spearman of the gap with imbalance: CITL ρ = −0.79 to −0.82 and classwise ECE −0.62 to −0.74 (T1′, all p < 0.05).
- **Proper scores: stratified is worse from 3C on.** Brier −0.03/−0.035/−0.036 (raw p 0.06/0.004/0.005; Holm 0.62/0.07/0.10). D² Brier −0.11* at 10C and 20C.
  Log loss −0.04 at 20C (n.s.).
- **Accuracy: stratified is worse.** −0.047/−0.036/−0.036 (rank-biserial −0.62 to −0.72, raw p 0.002–0.005, Holm 0.05–0.10). In the balanced half
  stratified is better at 1–3C (+0.046*, +0.025*). Spearman of the gap with imbalance: −0.38 to −0.62 at every budget, on both sets.
- **Class-averaged metrics: the opposite.** BA +0.056*/+0.048*/+0.047* in the imbalanced half against +0.033*/+0.008/+0.003 in the balanced half.
  BA ρ with imbalance is +0.72 at 10C on T1′ and +0.87 on DEV.
- **All-class AUC:** in the imbalanced half the gap lasts longer (+0.05* at 3C, +0.01 at 10C), because the rare class keeps being missed. ρ is +0.35 to +0.48 from 3C on.
- **LR+** is −0.33* to −0.67* in the imbalanced half at every budget.
- **Number of classes C:** effects are weak and mostly not significant. At 1C, multiclass tables have larger class-averaged gaps (T1′ macro-F1: +0.120 multiclass against +0.089 binary) and larger log-loss/KL gaps
  (ρ_C ≈ +0.3, p < 0.05), as more classes can be missed.

### 2.4 Which predictions held
| Prediction (Sec. 1) | Outcome |
|---|---|
| F3 inflated, strongly at 1C | **held** |
| F3 order BA ≥ F2 ≥ F1 ≥ F0.5 ≥ precision at ≥ 3C | **held from 5–10C on** (T1′ 10C: .058 > .041 > .034 > .027 > .003). At 1–3C the order is **reversed**: macro precision is the most inflated metric at 1C (+0.14 / +0.18), because an unseen class has precision 0 and the absorbed rows lower the precision of the seen classes |
| F2 ≈ macro-F1 at 1C, between accuracy and macro-F1 later | held at 1C; later kappa/MCC are as large as macro-F1, not in between |
| F1: accuracy small with mixed sign; weighted F1 > accuracy at 1C; top-2 ≈ coverage only | **held**. Accuracy is negative on imbalanced pools, as B3 predicts |
| F4 inflated at 1C, ≈ 0 from 3C | partly: inflated at 1C, but still +0.03–0.04* at 3–5C on T1′, because coverage of rare classes is slow on imbalanced pools |
| F5 ≈ 0 | **held** |
| F6: log loss + at 1C (clip artefact), Brier small +, negative later on imbalanced pools | **held** (Brier's 1C gap, +0.10, was larger than "small") |
| F7 + at 1C, − later on imbalanced pools, ECE weakest | **held** |
| F8: positive at 1C; LR− better later, LR+ mixed | LR− held. **LR+ is clearly worse** under stratified from 2C on, and at every budget on imbalanced pools |
| Flips most frequent for calibration and proper scores, least for AUC/accuracy | **partly**: see 2.5. Proper scores do not flip more than noise |
| Three gap clusters, families not cleanly separated | **mostly held**: see 2.6. Calibration did not form its own cluster |

### 2.5 FM ranking flips (`e17_flip_rates.csv`, `e17_flip_excess.csv`)
For each table × budget, the 4 FMs are ranked under stratified and under random. The flip rate is the share of the 6 FM pairs whose order differs.
Many of these flips are noise, because the FMs are close. The noise level is measured by the same statistic between two disjoint 5-draw halves of the random draws.
T1′ flip rates, protocol against noise, with Holm p over 30 metrics for the excess:
- macro-F1 0.39 vs 0.24 (p < 1e-3); F2 0.39 vs 0.24; BA 0.37 vs 0.25; F0.5 0.38 vs 0.25 (all p < 1e-3);
- KL 0.48 vs 0.34 and CITL 0.47 vs 0.34 (p ≈ 0.001); kappa 0.37 vs 0.27 (p = 0.001); MCC 0.36 vs 0.29 (p = 0.009); weighted F1 0.36 vs 0.27 (p < 1e-3);
- **no excess** over noise for accuracy (0.33 vs 0.29, Holm 0.42), all AUC/AP (both conventions, ≈ 0.27 vs 0.29), log loss/Brier (0.22 vs 0.21) and macro precision.

On DEV (20 tables), no excess survives Holm. The largest is F2 (+0.13, raw p 0.008). Conclusion: the protocol changes which FM looks best under class-averaged, chance-corrected and
calibration-in-the-large metrics. It does not change the ranking under accuracy, AUC or proper scores beyond resampling noise.

### 2.6 Correlation structure (`e17_corr_gaps.csv`, `e17_corr_draws.csv`, `e17_clusters.csv`)
These clusters are computed over the gaps per (set, table, FM, m) cell, with Spearman correlation, average linkage and 1 − ρ as the distance.
- With k = 3:
  - {kappa, MCC, BA, macro-F1, F0.5, F2, macro precision, macro recall, all-class AUC OvR/OvO/weighted, AP} is the coverage/class-balance cluster;
  - {accuracy, weighted F1, macro Jaccard, log loss, Brier, D², ECE, classwise ECE, CITL, KL} is the prevalence/calibration cluster;
  - {the four seen-class ranking metrics} form a cluster of their own.
- The ARI against the 7 theoretical families is 0.34 at k = 3 and 0.40 at k = 7. At k = 7, proper scores split from calibration only partly ({log loss, D² log loss, KL}, {Brier, D² Brier, classwise ECE, CITL}).
- The draw-level structure (random draws, centred within cells) is different. All decision and ranking metrics form one block, and the probability-based metrics (log loss, Brier, ECE, CITL, KL) form another. Within a protocol, draws that are good for one metric are good for all of them. **The families separate only in the protocol contrast**, as the coverage/balance mechanisms predict.

### 2.7 Side analysis: when do label-free selectors work? (`e17_selector_clusters.csv`, `e17_selector_cluster_rho.csv`, T1′)
The pool is partitioned by nearest selected medoid, in each selector's own space: z-scored X for k-medoids and cached TabClustPFN embeddings for LUCoS. This partition is compared with the classes.
- The agreement is low: median AMI is 0.04 (k-medoids, 1C) to 0.09 (3C).
- It still predicts the selector's macro-F1 gain over random (4-FM mean). At 1C, Spearman ρ = 0.63 [0.41, 0.80] for k-medoids and 0.63 [0.39, 0.79] for LUCoS; at 3C, ρ = 0.60 and 0.46.
- At 1C, class coverage of the selected set predicts the gain even better (ρ = 0.83 / 0.76).
- Reading: selectors help where the cluster structure of X follows the classes, and mostly because they then cover the classes.

### 2.8 (Section with suggested wording for the paper removed.)

### 2.9 Limitations
Exploratory: the predictions for macro-F1, accuracy, BA and AUC were not blind (Sec. 1). Gaps are differences of table means of 5 stratified against 10–20 random draws. The tests run over tables with FMs pooled, and Holm is applied within families. The per-FM tests are in the CSV. With only 14 binary tables on DEV, the DEV LR results are weak, and top-2 accuracy has only 6 (DEV) or 17 (T1′) multiclass tables. Imbalance halves are a post-hoc median split. In the selector analysis, AMI is confounded with table difficulty and coverage. E17 contrasts protocols at a fixed budget and does not estimate budgets, so it does not overlap with E15.
