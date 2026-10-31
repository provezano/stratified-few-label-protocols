# Human check of the systematic review: adjudication of disagreements (EXPLORATORY)

Status: **proposals**, confirmed by an author on 2026-10-29 (last section).

- Sample: the 32-paper random sample (`human_check_sample.csv`, 20% of the 161 included papers).
- Human codes: `data/human_check_form.csv` (an author, blind, median 4.5 min per paper).
- Final codes: `data/final_coding.csv` (copy of `research_notes/systematic_review_2026-10/coding/final_coding.csv`;
  the code columns equal `supplementary/audit/coding/final_coding.csv`, only one `notes` cell differs).
- Proposed codes: `data/adjudication.csv`. Script: `hc_analysis.py` (writes `agreement.csv`, `out/correction.csv`, `out/results.json`, `out/hc_numbers.tex`) (bootstrap B = 10,000, seed 20261029).
- sha256: final_coding `d23cfcb1…6b06`, human_check_form `eaec32b4…c465`, human_check_sample `1779c21e…8349d`
  (full hashes: `shasum -a 256 data/*.csv`).

Sources read: the full text of each paper (the arXiv/proceedings PDF text used by the review,
`downloads/text/<id>.txt`, with appendices) and, where the paper links one and the text leaves the
draw open, the official repository (read online, not run). Quotes are verbatim; only PDF
word-joins were re-spaced. The codebook is `01_CODING_MANUAL.md` §7, applied literally.

## Disagreements

There are 11 papers with a disagreement on `category` or `coverage_forced` (10 on category, 9 on
coverage; 8 on both). An earlier estimate of about 13 included the few-shot and metric
disagreements; those two variables are not adjudicated here.

| # | Paper | Human | Final | Proposal | Verdict |
|---|---|---|---|---|---|
| 1 | AX2306.08954 (AL benchmark, TMLR) | B / no | A′ / yes | A′ / yes | final right |
| 2 | AX2311.09962 (multi-omics SSL) | A′ / yes | A′ / unclear | A′ / unclear | final right (codebook example) |
| 3 | AX2312.12112 (CLLM) | E / unclear | A / yes | A / yes | final right |
| 4 | AX2403.15464 (EHR-CoAgent) | E / unclear | A / yes | A / yes | final right |
| 5 | AX2404.09491 (FeatLLM) | B / no | A / yes | A / yes | final right |
| 6 | AX2406.17322 (ALPBench) | C / no | A′ / yes | A′ / yes | final right |
| 7 | AX2508.17391 (LLMs as universal predictors) | E / unclear | D / unclear | D / unclear | both defensible |
| 8 | AX2601.11311 (FORESTLLM) | E / unclear | A / yes | A / yes | final right |
| 9 | AX2604.16412 (CC-SSL, GECCO) | D / unclear | A′ / yes | A′ / yes | final right (from code) |
| 10 | AX2610.23765 (USE-AD, anomaly detection) | E / no | A / yes | A / yes (provisional) | codebook ambiguous |
| 11 | H24fbd01a (AL with LLMs, NeurIPS 2024) | C / yes | A / yes | A / yes | final right (codebook AL rule) |

Totals: final right 9, human right 0, both defensible 1, codebook ambiguous 1. In none of the 11
cases does the paper support the human code against the final code. Where the evidence sits:
appendix (3, 5, 6, 8), official code (1, 9; also confirms 5), main methods text that the human
code does not reflect (1, 4, 11), and a codebook rule (2: rounding rule; 6, 11: "code the initial
pool" rule for active learning).

### Evidence per paper

**1. AX2306.08954** — *An Expanded Benchmark … Uncertainty Sampling for Active Learning in Tabular Datasets.*
- Human: "uniformly sampling few examples as the initial labeled pool" (§3) → B / no.
- Decisive text, p.9, §3 "Handle errors and exceptions of experiments": "we run the experiments
  repeatedly and skip any seed that lacks labels in the training or test set at the initial setup."
  Footnote 3 defines the cold-start problem as "some classes are under-represented in the initial labeled pool".
- Official code (`github.com/ariapoy/active-learning-benchmark`, `src/main.py`): if
  `len(y_lbl_class) != len(y_class)` the run returns `"Not enough label in label pool"`, and such
  seeds are dropped from the aggregated results (`src/utils.py`, default `'RS'` draw: shuffle, take the first `init_lbl_size`).
- Codebook: A′ lists "discarding seeds that miss a class"; coverage `yes` covers "by rejection".
  The draw itself is uniform (the human reading), but the reported runs are conditioned on full
  coverage. **Final right.** Note: the headline metric (AUBC) therefore excludes cold-start runs.

**2. AX2311.09962** — *Self-supervised learning of multi-omics embeddings in the low-label, high-data regime.*
- Both code A′; only coverage differs (human yes, final unclear).
- p.5, §3.1: "either 1%, 5% or 10% of the training set was used for finetuning. All splits were
  stratified to keep the proportion of classes preserved in the splits."
- p.3, §2: "only keeping classes with > 100 samples. This allowed us to investigate the performance
  of our models in the low-label regime while keeping all data splits stratified".
- No repository is linked in the text. The class filter shows intent, but no guarantee of ≥ 1 row
  per class at 1% is stated (about 0.7 expected rows for a class with 101 samples after the
  20% test and 10% validation holds).
- Codebook: coverage `unclear` "if labels are used but the guarantee is not stated (e.g.
  proportional stratification with tiny k and unstated rounding)" — this case. **Final right**
  (the human `yes` is a reasonable reading of intent, not of a stated guarantee).

**3. AX2312.12112** — *Curated LLM (CLLM).*
- Human: "none found" in §3.1 / Tables 1–2 → E.
- p.3, §2: "Assume Dtrain is drawn i.i.d." (problem setting, not the experimental draw).
- p.17, App. B: "we sample a training set Dtrain such that |Dtrain| = n, and each target class has
  the same number of samples." (n ∈ {20, 40, 100, 200}, 10 seeds).
- Codebook: k total balanced across classes → A; per-class draw → coverage yes. **Final right.**

**4. AX2403.15464** — *LLMs-based Few-Shot Disease Predictions using EHR (EHR-CoAgent).*
- Human: "none found"; notes that random sampling is described for the test set only.
- p.3, "Few-Shot: Enhancing Performance with Limited Examples": "We randomly select a small number
  of positive and negative samples (e.g., 3 positive and 3 negative) from the training data".
  Table 1 (p.5): "Few-Shot (N=6)" for both the ML baselines and the LLMs.
- Codebook: a per-class draw (3 per class, binary) → A / yes. The "e.g." leaves a small doubt,
  but N = 6 matches 3 + 3. **Final right.**

**5. AX2404.09491** — *FeatLLM.*
- Human: "only k (<N) labeled samples are randomly sampled to train the model" (p.3, §3) → B / no.
- p.18, App. C: "From the remaining data, k-shot samples were selected, constructing a balanced training set."
- Official code (`github.com/Sungwon-Han/FeatLLM`, `utils.py`, `get_dataset`):
  `sample_num = shot // len(np.unique(y_train))` inside `groupby(default_target_attribute)`, with the
  remainder `shot % n_classes` spread one per class. Datasets have ≤ 4 classes, so k = 4 gives ≥ 1 per class.
- The main text alone supports B; the appendix and the code show a class-balanced draw → A / yes.
  The codebook's "randomly sample k" → B aid applies "unless the code (if checked) shows otherwise". **Final right.**

**6. AX2406.17322** — *ALPBench.*
- Human: C / no ("none found"; notes that a class may be absent initially).
- p.22, App. A.4 "Active Learning Setting": "|D0L| = R = 5·|Y| … For some (imbalanced) datasets, it may
  happen that not every class is at least once represented in D0L. In these cases, we additionally
  randomly sample one instance from DU per missing class and add them with their corresponding label to D0L."
- Codebook: "For active learning, code how the initial labelled pool is drawn"; a "constraint of
  ≥ 1 per class" → A′ / yes. **Final right.**

**7. AX2508.17391** — *LLMs as Universal Predictors? … Small Tabular Datasets.*
- Coverage agrees (unclear). Category: human E, final D.
- The main experiments use the whole training split of 50–500-row datasets as ICL context (no
  reduced-label draw). The only reduced-label experiment is the ablation, p.11, §4.4: "Fractions
  of 25%, 50%, 75%, and 100% of the training split are evaluated." The draw of the subsample is not described.
- Codebook: "Learning curves over training sizes with the draw unstated → D if sizes are fractions";
  but E also covers "no experiment", which fits the main protocol. **Both defensible**; proposal D
  (the decision aid is explicit). No effect on the headline: D and E are both "draw not described".

**8. AX2601.11311** — *FORESTLLM.*
- Human: "none found" in §Experiments, Table 2, Appendix B → E.
- p.12, App. B "Data Preparation": "From the training set, we construct a k-shot subset for supervision,
  selecting examples in a class-balanced manner whenever possible. In scenarios with limited data or
  many classes, we permit approximate balancing". It follows FeatLLM (Han et al. 2024; paper 5).
- Codebook: k total balanced across classes → A; ≤ 4 classes at k = 4, so ≥ 1 per class → yes.
  **Final right.** (The human form cites Appendix B as inspected, but the sentence is there.)

**9. AX2604.16412** — *Cooperative Coevolution … Semi-Supervised Tabular Classification.*
- Human: p.4, §4: "The labeled set size is determined by labeled fractions L={0.01, 0.05, 0.10}" → D / unclear.
  This is the correct code from the text alone.
- Official code, linked in the paper (ref. [18], `github.com/jamaltoutouh/cc-ssl-gecco2026`,
  `cc_ssl_openml.py`, `make_ssl_splits`): `min_labeled = 2 * n_classes`;
  `n_L = max(min_labeled, int(round(n_pool * labeled_frac)))`; `min_per_class = 2`; per class
  `rng.choice(idx_c, size=take_c, replace=False)`, then a uniform fill of the rest.
- Codebook: code may be read "only when the text leaves the draw ambiguous" (it does: fraction
  only), and a per-class seed → A′ / yes. **Final right** (on the code; human right on the text).

**10. AX2610.23765** — *Universal Semantic Enhancement for Weakly Supervised Anomaly Detection.*
- Human: E / no ("selection procedure for those anomalies is not specified").
- p.4, §3.1: "only a small subset of anomalous samples M={(xj, yj)|yj = 1} with |M| ≪ n is labeled."
  p.10, §4.1: "In the training set, we fed the model only |M| = 8 labeled anomalies".
- The labelled set is, by definition, 8 rows of the anomaly class; the normal class gets no label
  (PU-style setting). The draw uses the label (it picks anomalies), so it is not B or E in the
  sense of "draw not described", but it is not "m labels per class" either, and "≥ 1 labelled row
  per class" fails for the normal class. The codebook does not cover one-class labelling.
- **Codebook ambiguous.** Provisional proposal: keep A / yes (the closest category: a fixed number
  of rows drawn from a class chosen by its label). Alternative: E (human) or exclusion from the
  few-shot frame as a different task. The alternative is used in the sensitivity analysis.

**11. H24fbd01a** — *Active Learning with LLMs for Partially Observed and Cost-Aware Scenarios.*
- Both code coverage yes. Category: human C, final A.
- p.8, §4: "We start training with two fully observed samples per class, conduct 150 acquisition cycles".
- Codebook: "For active learning, code how the initial labelled pool is drawn. If the whole
  labelled set is chosen by an algorithm with no random initial set, use C." Here the initial pool
  is 2 per class → A. **Final right** (the final file also keeps C as the secondary category).

## Agreement (32 papers; 95% bootstrap CI over papers)

| Pair | Variable | Agree | Raw [CI] | κ [CI] |
|---|---|---|---|---|
| human vs final | category | 22/32 | 0.69 [0.53, 0.84] | 0.63 [0.43, 0.81] |
| human vs final | coverage_forced | 23/32 | 0.72 [0.56, 0.88] | 0.58 [0.35, 0.81] |
| human vs final | few_shot | 30/32 | 0.94 [0.84, 1.00] | 0.76 [0.33, 1.00] |
| human vs final | missed_class_reported | 32/32 | 1.00 | undefined (all "no") |
| human vs final | main_metric | 26/32 | 0.81 [0.66, 0.94] | 0.75 [0.56, 0.91] |
| human vs proposed | category | 22/32 | 0.69 [0.53, 0.84] | 0.63 [0.43, 0.81] |
| human vs proposed | coverage_forced | 23/32 | 0.72 [0.56, 0.88] | 0.58 [0.35, 0.80] |
| final vs proposed | category | 32/32 | 1.00 | 1.00 |
| final vs proposed | coverage_forced | 32/32 | 1.00 | 1.00 |
| human vs proposed (alt.) | category | 24/32 | 0.75 [0.59, 0.88] | 0.70 [0.51, 0.86] |
| human vs proposed (alt.) | coverage_forced | 25/32 | 0.78 [0.63, 0.91] | 0.68 [0.46, 0.86] |
| final vs proposed (alt.) | category | 30/32 | 0.94 [0.84, 1.00] | 0.92 [0.80, 1.00] |
| final vs proposed (alt.) | coverage_forced | 30/32 | 0.94 [0.84, 1.00] | 0.90 [0.74, 1.00] |

"Alt." resolves the two non-clear-cut cases toward the human (paper 7 → E; paper 10 → E / no;
paper 2 coverage → yes). `main_metric` treats the human's "AUROC" as AUC (one paper); without this
mapping the agreement is 25/32, as found earlier (κ = 0.71). `few_shot` and `main_metric`
were not adjudicated.

## Effect on the headline (few-shot frame: 73 of 100 with a described draw)

Method: direct confusion-matrix correction. For each final group g ∈ {A, A′, B/C, D/E}, the
sample gives P(reference group | final group); the corrected count is Σ_g N_g · P(· | g), applied
to the 132 few-shot papers (40 A, 33 A′, 27 B/C, 32 D/E). Bootstrap over the 32 sample papers.

| Reference | A + A′ (K) | described draw (N) | K / N |
|---|---|---|---|
| proposed adjudication | 73 [73, 73] | 100 [100, 100] | 73% [73, 73] |
| alternative adjudication | 70 [62, 73] | 97 [89, 100] | 72% [70, 73] |
| human codes taken as true | 36.5 [17.5, 55] | 81 [65, 95] | 45% [24, 63] |

- Over-assignment of A/A′ by the final coding: 0 of 18 sample papers coded A/A′ (Clopper–Pearson
  95% upper bound 19%); against the human codes, 9 of 18 (50%, CI 26–74%), and adjudication
  overturns all 9.
- Under-assignment: 0 of 14 (upper bound 23%) against both references.
- Because no error is observed, the bootstrap interval for the proposed adjudication has zero width;
  this reflects the sample, not certainty. The worst case the sample allows (19% of the 73 papers
  not truly A/A′) gives 59 of 100 (59%) if those papers truly use B/C, or 59 of 86 (69%) if their
  draw is truly undescribed. Both remain a majority.
- Rogan–Gladen on all 161 (A/A′ vs other): 48% with the proposed adjudication (sensitivity =
  specificity = 1), 45% with the alternative, 15% if the human codes were taken as true.

**Conclusion (proposal):** the headline "73 of 100" (40 A, 33 A′) does not need to change. The
human–final disagreement comes from where the evidence sits (appendix, code, codebook rules), not
from wrong final codes. If paper 10 were resolved as E, the counts become 72 of 99 (39 A);
the paper's statement ("most … use the labels to draw it") holds under every scenario except
"human codes taken as true", which the adjudication does not support.

(Section with proposed wording for the paper removed.)

Headline counts: no change proposed.

## Confirmation

2026-10-29: an author confirmed every proposal above, kept AX2610.23765 coded A, and confirmed that the 32 papers were coded blind to all codes.
