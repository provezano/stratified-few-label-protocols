# Pilot I2 — Tabular foundation models vs graph SSL, and their combination

Status: DRAFT for approval (2026-10-20). Nothing below may change after the
first pilot result is computed, except through a dated amendment at the end.

## 1. Questions and hypotheses

- **RQ-a (relevance).** On small tabular datasets with very few labels, does
  classic graph-based SSL still beat a tabular foundation model (FM) that sees
  only the labeled points?
  - H2a: best FM ≥ best classic graph SSL (macro-F1, paired by dataset).
- **RQ-b (combination).** Does propagating over a graph, built on FM
  embeddings or seeded with FM predictions, add value over the FM alone?
  - H2b: best hybrid > best FM by more than ROPE = 0.01 on a non-trivial
    share of datasets.
- **RQ-c (when).** Can a label-free diagnostic predict, per dataset, whether
  the hybrid beats the FM?
  - H2c: a leave-one-dataset-out rule built on the diagnostic has lower
    regret than both "always FM" and "always hybrid".

Evidence motivating each hypothesis: `research_notes/.../_idea_I2_foundation_vs_graphs.md`.
The fact behind H2b: TabPFN test points do not attend to each other (Nature
2025, Methods), so transduction must come from outside the model.

## 2. Dataset pool (fresh and unselected)

**Eligibility.** OpenML datasets, queried once and frozen to
`data/pool_i2.yaml` with IDs and versions. A dataset is eligible if it meets
all of the following:
1. status `active`, format not sparse, a default target attribute that is
   nominal;
2. 100 ≤ NumberOfInstances ≤ 1000 (see the note on pool size below);
3. 2 ≤ NumberOfClasses ≤ 10 (TabPFN v2 class limit);
4. all features numeric (NumberOfSymbolicFeatures = 1, i.e. only the target);
5. no missing values;
6. 2 ≤ NumberOfFeatures − 1 ≤ 100;
7. the smallest class has ≥ 11 instances, so that m = 10 per class still
   leaves unlabeled points.

**Exclusions.**
- The 124 IDs of the PRIORSEL study (S50 ∪ X74; `finding_dataset_selection_PRIORSEL.md`).
- Duplicates. Two datasets are duplicates if their (X, y) SHA-256 is equal
  after sorting rows. The lowest dataset ID is kept.
- Multiple versions of one name: the lowest active version is kept.
- Generator families: at most one dataset per family. The family is the name
  truncated at its first digit (e.g. `fri_c3_500_5` → `fri_c`,
  `arcene_seed_0_…` → `arcene_seed_`), and the lowest ID is kept.

**Note on pool size (metadata probe, 2026-10-20, before any outcome).** With
n ≤ 500, numeric-only and no missing values, only 37 datasets remain after
excluding the 124 PRIORSEL IDs, about 22 once synthetic and iris copies are
removed. The PRIORSEL study consumed most of that regime. Widening to n ≤ 1000
(still within TabPFN's CPU limit) gives 102, about 67 without copies, and keeps
preprocessing identical across arms. Allowing categoricals would force a
one-hot choice that handicaps graph arms relative to FMs (a confound), so they
stay excluded. The PRIORSEL ID list is frozen in `data/priorsel_excluded_ids.txt`.

**Sampling.**
- Sample 60 eligible datasets with `numpy.random.default_rng(20261020)`;
  use all of them if fewer are eligible.
- Assign the first 20 sampled to **DEV** and the rest to **HELD-OUT**.
- **The pilot uses DEV only.** HELD-OUT stays untouched for the confirmatory
  study.
- Analyses are reported overall and stratified by n ≤ 500 vs n > 500.

## 3. Protocol

- **Labels.** m ∈ {1, 3, 5, 10} labeled points per class, sampled
  stratified-exactly. Seeds 0–9 per (dataset, m), via `SeedSequence`.
- **Transductive evaluation.** All non-labeled points are the unlabeled pool
  and also the evaluation set. Their labels are never visible to any arm.
- **No validation labels anywhere.** Every arm uses all m·C labels for
  fitting, and every hyperparameter is fixed a priori (below). This removes
  the selection noise studied in the A+B line.
- **Metrics.**
  - Primary: macro-F1 (unweighted mean of per-class F1) on the unlabeled
    points.
  - Secondary: accuracy.
- **Preprocessing.**
  - Graph arms get a z-score fitted on all X, which uses no labels.
  - FM arms get raw X; they standardize internally.

## 4. Arms (all hyperparameters fixed a priori)

| ID | Arm | Specification |
|----|-----|---------------|
| FM1 | TabPFN v2 | `TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=seed)`, fit on labeled points |
| FM2 | TabICL v2 | `TabICLClassifier(device="cpu", random_state=seed)` (checkpoint `tabicl-classifier-v2-20260212`), fit on labeled points |
| FM3 | Kumo-Tabular (NVIDIA, 2026) | `sdm.models.KumoTabular(task="classification", size="large", device="cpu")`, `num_estimators=8`, `torch.manual_seed(seed)`. Pretrained only on synthetic tables (HF blog); weights OpenMDW-1.1 |
| G1 | LLGC | Mutual kNN, k = 10, Euclidean on z-scored X. RBF weights with σ = mean k-th-NN distance / 3 (de Sousa et al. 2013). Isolated vertices linked to their 1-NN. α = 0.99 (Zhou et al. 2004). Closed form. Argmax. |
| G2 | GFHF | Same graph as G1; harmonic solution per connected component; argmax. A component with no labeled point is predicted by FM1's argmax and the count is reported. |
| ST | Loop-FM | LoopTabFM (Li et al. NeurIPS 2025, Alg. 1) adapted to multiclass: K = 5 loops of TabPFN v2, hard argmax pseudo-labels, Val_Risk = mean(1 − max p). Returns the minimum-risk loop. The adaptation is our own and is declared as such |
| H1 | FM-embedding graph | TabPFN v2 fit on the labeled points; `get_embeddings(X_all, "test")` averaged over estimators; z-score; graph and LLGC as in G1 |
| H2 | FM-prior smoothing on the feature graph | F = (I − αS)⁻¹ Y₀. Y₀ is one-hot on labeled points and FM1 probabilities on unlabeled points; α = 0.9; G1's graph. Analogous to the Smooth step of Correct & Smooth |
| H3 | FM-prior smoothing on the embedding graph | H2 with H1's graph |
| H4 | Kumo-prior smoothing on the feature graph | H2 with FM3 probabilities as the prior |

Sensitivity, reported but not used for decisions: k ∈ {5, 20} for G1/H1/H3.

**Not included, and why:**
- TabDPT-semi: trained on real OpenML data, which risks contamination with
  our pool.
- GE/GP-evolved similarities: they need validation labels.

## 5. Label-free diagnostics (for H2c)

Each diagnostic is computed per (dataset, m, seed) from X, the m·C labels and
FM1's outputs only:
- D1, prediction homophily: the share of G1-graph edges whose endpoints have
  the same FM1 argmax.
- D2: mean FM1 confidence (max probability) on unlabeled points.
- D3: agreement between the FM1 and H1 argmax on unlabeled points.
- D4: the number of connected components of the G1 graph divided by C.

**Rule.** A logistic regression on (D1–D4, log m), fit with leave-one-dataset-
out over the DEV datasets. It predicts sign(macro-F1(best hybrid) −
macro-F1(FM1)). The decision is "use hybrid iff p > 0.5".

## 6. Decision criteria (fixed before running)

Units:
- a **dataset-level score** is the mean over seeds;
- **Δ** is hybrid − FM, paired by dataset and m;
- **best FM** is the best of FM1, FM2 and FM3, chosen per dataset on DEV and
  reported as a selected quantity;
- **best hybrid** is the single hybrid with the highest mean Δ over the DEV
  cells, which then becomes the hybrid for HELD-OUT.

| Outcome | Condition on DEV | Action |
|---------|------------------|--------|
| **KILL** | Mean Δ ≤ 0.005 **and** the share of datasets with Δ > 0.02 at any m is < 20% | Shelve the graph-SSL-on-tabular line. Report H2a as a short note |
| **GO** | Share of datasets with Δ > 0.02 at some m is ≥ 25% **and** the LODO rule's mean regret is < min(regret always-FM, regret always-hybrid), with AUC ≥ 0.70 | Run the confirmatory study on HELD-OUT with the frozen hybrid and rule |
| **GREY** | Anything else | Report to the authors; no automatic continuation |

**H2a** is reported descriptively on DEV. It is not a stop criterion.

Statistics:
- Wilcoxon signed-rank with Holm over the 4 values of m;
- bootstrap 95% CI over datasets;
- Bayesian signed-rank with ROPE 0.01 (`baycomp`), descriptive.

With 20 DEV datasets these are exploratory. Confirmatory claims come only
from HELD-OUT.

## 7. Compute and reproducibility

- **Budget.** 20 datasets × 4 values of m × 10 seeds = 800 cells. Each cell
  takes about 8 TabPFN fits at about 1.4 s on CPU plus 1 TabICL fit at about
  0.9 s; the graph arms are negligible. That is about 3 h on one process, or
  under 1 h with 6 workers.
- **Determinism.** CPU only, since MPS outputs differ slightly from CPU
  (probe 2026-10-20). Torch and BLAS run with fixed thread counts, and fixed
  seeds give bit-identical repeats (verified in the probe).
- **Provenance.** Git SHA, `uv.lock` hash, dataset SHA-256 and arm config,
  recorded per result file.

## 8. Amendments

- **2026-10-20, before any pilot result.**
  - FM3 (Kumo-Tabular) and H4 were added at the authors' request to cover the newest competitors evaluated in [internal project]. Kumo-Tabular claims rank 1 on TabArena (NVIDIA-reported, not yet independently rerun).
  - TabPFN v3.5 and TabFM were not added: their weights are non-commercial and need license acceptance by the authors. They may be added as an amendment before HELD-OUT.
  - Hybrid embeddings come only from TabPFN v2, the one FM whose embedding API we verified.
  - LLGC rows with no propagated mass (components without labels) take FM1's argmax, the same rule as GFHF, and the count is reported.
  - The TabPFN v2 checkpoint is `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`. Its HF model card says it is identical to `v2_default`, a production base checkpoint (Nature 2025).

### Pool freeze note (2026-10-20, before any pilot result)
`build-pool` produced 72 metadata-eligible candidates, 52 valid unique (DEV 20, HELD-OUT 32), 8 content duplicates and 12 rejections (11 non-numeric features; did 895 rejected by a transient OpenML HTTP 503). The frozen pool is kept as is: the rejection does not depend on any result, and re-querying would change the sampling frame after freezing.

## Amendment 2 (2026-10-20, written while the pilot was running and before any result was inspected)

The authors accepted, for research use only, the TabFM Non-Commercial License v1.0 and the TabPFN v3.5 non-commercial licence. Added arms:

| Arm | Definition |
|---|---|
| FM4 | TabFM (Google weights `google/tabfm-1.0.0-pytorch`, loaded via the `sdm.models.TabFM` wrapper, CPU, 8 estimators, torch seed = cell seed) on the m·C labeled points |
| H5  | LLGC prior smoothing on the feature graph with the FM4 prior (as H4, alpha_prior = 0.9) |
| FM5 | TabPFN v3.5 (`ModelVersion.V3_5`, CPU, defaults); run only once the authors set `TABPFN_TOKEN` (weights are gated) |
| H6  | As H5 with the FM5 prior |

Rules:
- The extension runs in a separate pass on the **same splits and seeds** (`cell_seed`, `draw_split`); its results go to `results/pilot_i2_ext/`.
- **The primary analysis and the KILL/GO/GREY decision stay exactly as registered** (FM1–FM3 against H1–H4).
- The extended FM and hybrid sets (FM1–FM5, H1–H6) form a **secondary** analysis. It is reported next to the primary one and cannot change the primary outcome. Rationale: a stronger FM baseline can only make the hybrid comparison harder, so the secondary analysis is a robustness check of any positive primary result.
- H5 and H6 rows with no propagated mass fall back to the argmax of their own FM. Such rows cannot occur when every unlabeled row holds a prior.
- Weights and licences are research-only, so no result built on FM4, FM5, H5 or H6 is used outside research.
- FM5 checkpoint: `tabpfn-v3.5-20260909.safetensors` (876,027,932 bytes), local CPU inference; bit-identical on repeat with a fixed seed. Started 2026-10-20 ~18:50, still before any result was inspected for the decision. Note: interim descriptive numbers for the primary arms were shown to the authors at their request at ~18:45; Amendment 2 was written before that, and no registered rule was changed afterwards.

## Amendment 3 (2026-10-20 ~21:45, before the affected cell had any result)

Issue: the cell (did 782 *rabe_266*, m = 1) crashed in every arm before producing any result. With 2 labeled points as context, TabPFN v2 test embeddings collapse onto duplicates, so every 10th-neighbour distance on the embedding graph is 0 and σ = 0 (the code raised by design).

Fix: if the mean k-th-neighbour distance is 0, σ = (mean positive pairwise distance) / 3; if no positive distance exists, σ = 1.
- The fix only changes inputs that previously raised, so none of the 79 completed cells is affected; their results are kept unchanged.
- The affected cell is re-run with the fix.
- The embedding collapse is reported as a finding: at m = 1, the embedding-graph arms (H1, H3) can be uninformative.
- Provenance note: re-running `run-pilot` rewrote `results/pilot_i2/provenance.json` with the source hash of the fixed code. The original hash, from before Amendment 3, was not preserved. The only source changes between the two runs are Amendment 2's extension modules (not used by the pilot arms) and the σ fallback above.
