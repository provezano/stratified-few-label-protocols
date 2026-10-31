# TabDPT (fm6) pretraining-contamination check (EXPLORATORY, 2026-10-30)

Post hoc and documentary. No model was run, no fits, no HELD-OUT data or outcome
was read. HELD-OUT enters only as the ID list in
`research_notes/heldout_openml_ids_exclusion.txt` plus public OpenML metadata
(feature count, suite membership) for those IDs. The approach mirrors
`../realtabpfn_check/`: match by OpenML ID and by name, then judge re-uploads and
derived tables by hand.

## 1. What fm6 is

| Item | Value | Source |
|---|---|---|
| Model | TabDPT **v1.3** | `src/simaudit/runner/fm_extra_worker.py` PINS, `../fm_extra/out/pins.json` |
| Package | `tabdpt` 1.3.1 in `.venv_fm_extra` (torch 2.14.1, sklearn 1.9.1, numpy 2.5.4) | pins + `importlib.metadata` in the venv |
| Default weights | `tabdpt/estimator.py`: `_VERSION = "1_3"`, `tabdpt1_3.safetensors` from `Layer6/TabDPT` | installed package |
| HF revision | `a5ca6e01c0fa09ec68c73e958e5199d1932abb3a` ("Upload tabdpt1_3.safetensors", 2026-09-08) | pins; HF commit list |
| SHA-256 | `97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb` | pins; recomputed on the local HF cache file, matches |
| Settings | CPU, in-context only, 8-member ensemble, `compile=False`, `use_flash=False` | worker, `../fm_extra/adapters.py` |

The Study 13 worker checks this SHA-256 before the first fit. The exploratory DEV
and T1′ adapters load the package default, which is the same file. The local
cache holds only this revision.

## 2. Training corpus

**Weights v1.0 / v1.1 (arXiv 2410.18164v3, App. B, Tab. B.1).** These are 123
OpenML datasets: 93 classification, 29 regression and 1 without a target. Table
B.1 was parsed from the arXiv HTML v3 (2026-01-17). Its IDs are exactly
`data_splits/noleak_training_datasets.csv` in `layer6ai-labs/TabDPT-training`,
which is the list read by `dataset.py:57`. The repository also contains
`noleak_training_datasets_anysize.csv` (79 IDs, 27 of them outside B.1). The
training code does not read that file. Among its extras are DEV `stock` (841),
`BNG(tic-tac-toe)` (137) and 6 HELD-OUT IDs. Contamination analysis (B.1) was run
only against CC18 and CTR23.

**Weights v1.2 / v1.3, i.e. our fm6, use a different and much larger corpus.**
- Source: the TabDPT-Turbo paper, arXiv **2608.01400v2** (2026-08-25), §3.1 and
  App. A.2, Table 2. The v1.2 release notes and the inference README (v1.2 =
  TabDPT-Turbo) point to this paper.
- Size: **1,445 OpenML datasets**, 309.9M rows. The same table gives "TabDPT v1.1:
  112 datasets", which disagrees with the 123 in B.1.
- How it was built:
  - start from all active OpenML datasets;
  - deduplicate against CC18, CTR23 and TabArena with the original TabDPT code;
  - drop tables with fewer than 10 columns or more than 200 columns, and files
    over 300 MB;
  - remove categorical columns with cardinality above 100 and constant columns;
  - deduplicate within the corpus by comparing order-invariant tables.
- **The ID list is not published.** It is absent from the arXiv HTML, the HF card
  and files, the inference repository and its release notes, and the training
  repository, which was last pushed in 2025-08.
- v1.3 (release notes, 2026-09-08) is "similar to v1.2", with "minor
  architectural tweaks and improvements to our training recipe". No corpus is
  stated, so the v1.2 corpus is the best assumption.

So Tab. B.1 does **not** describe the weights we ran. It is reported below
as a lower bound. The relevant check is
eligibility under the published v1.2 filter (§3b).

## 3a. Overlap with Tab. B.1 (v1.0 corpus)

| Set | Size | ID match | Name match | Overlap (judged) | Related (same source family, different table) |
|---|---|---|---|---|---|
| DEV | 20 | 0 | 0 | **0** | 0 |
| T1′ (PMLB/UCI) | 52 | n/a (no OpenML IDs) | 4 | **4** | 3 |
| HELD-OUT | 32 IDs | **0** | not checked (IDs only, by design) | **0** by ID | not checked |

Overlap rows in T1′:
- `GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1` matches OpenML 40646, the same
  table (1600 × 20).
- `mofn_3_7_10` matches OpenML 40680, the same table (1324 × 10).
- `Hill_Valley_without_noise` matches OpenML 1479 `hill-valley`. The OpenML
  description says "This is the non-noisy version, with training and test sets
  merged" (1212 × 100).
- `yeast` matches OpenML 40733 `yeast` v7: "A 4-class version of the yeast
  dataset" (1269 × 8). It has the same UCI rows and features with a coarsened
  target, so it counts as a derived table, like the tic-tac-toe precedent.

Related rows in T1′ (not removed):
- `Hill_Valley_with_noise` and OpenML 1479: same generator, noisy variant, so the
  values differ.
- `chess` (kr-vs-kp, 3196 × 36) and OpenML 1118 `chess`, 184 `kropt`, 1481
  `kr-vs-k` (all KRK, 28056 × 6): same UCI chess family, different endgame and
  table.
- `car_evaluation` and OpenML 1568 `nursery`: same DEX family, as in the
  Real-TabPFN check.

Name hits that were checked and rejected:
- T1′ `churn` vs `KDDCup09_churn`: telecom churn vs Orange, different data.
- DEV `German-Credit…` and `analcatdata_creditscore` vs 44089 `credit`: that is
  Give-Me-Some-Credit, different data.

## 3b. Eligibility under the v1.2/v1.3 corpus filter (unverifiable membership)

Eligible means the table is on OpenML, active, has 10 to 200 features (OpenML
`NumberOfFeatures`, target included), was uploaded before 2026-06, and is in
neither CC18 nor TabArena-v0.1 (suites 99 and 457). Being eligible makes
inclusion plausible, not confirmed, because content deduplication against
CC18/CTR23/TabArena could still remove a table.

| Set | Eligible | Not eligible |
|---|---|---|
| DEV | **7 / 20**:<br>German-Credit-Data-Creditability-Preprocessed-Dataset (46562, 34 features)<br>divorce_prediction (46875, 55)<br>stock (841, 10, at the boundary)<br>Touch2 (42544, 11)<br>SPECTF (337, 45)<br>waveform (47155, 22, uploaded 2026-01-16)<br>mw1 (1071, 38) | 2 in CC18 and therefore deduplicated: wdbc (1510), kc2 (1063).<br>11 with fewer than 10 features. |
| T1′ | No OpenML IDs, so not decidable. 28 of the 42 PMLB tables have at least 10 columns, and most PMLB/UCI tables have OpenML copies. The 10 UCI-API tables were not checked. **Treat as exposed.** | – |
| HELD-OUT | **19 / 32** (count only) | 5 in CC18; the others have fewer than 10 features. |

## 4. Verdict for P1

1. **What changes from the audit.** For our fm6 the audit's "123 IDs" is
   superseded. The weights we ran (v1.3) were pretrained on about 1,445 real
   OpenML tables whose list is not published. No fm6 number on DEV, T1′ or
   HELD-OUT can be certified contamination-free. In particular, 7 of 20 DEV and
   19 of 32 HELD-OUT tables are eligible for that corpus, and T1′ is largely
   exposed (its 4 B.1 overlaps are certain for v1.0 and only likely for v1.3).
   No drop-the-overlap sensitivity, like the Real-TabPFN one, can be built
   without the list.
2. **How much it matters.** fm6 is exploratory on DEV and T1′ and a secondary
   learner in Study 13, and no pre-registered verdict rests on it. P1's quantity
   is the stratified-minus-random gap with the same FM in both arms, so
   memorisation does not simply favour one arm. Its direction on the gap is
   unknown: recognising a table could make the FM less sensitive to which labels
   it sees.
3. (Suggested wording for the paper and internal follow-ups removed.)

## Sources read

- arXiv 2410.18164v3 HTML: §4.1, App. B, Tab. B.1, B.1 contamination analysis.
- arXiv 2608.01400v2 HTML: §3.1, App. A.2, Table 2.
- HF `Layer6/TabDPT`: card at revision a5ca6e0, commit list and file list.
- GitHub `layer6ai-labs/TabDPT-inference`: README and release notes v1.2.0 and
  v1.3.0.
- GitHub `layer6ai-labs/TabDPT-training`: tree, `data_splits/*.csv`,
  `dataset.py`.
- OpenML API: descriptions of 1479, 40733 and 1118; suites 99 and 457;
  qualities of the DEV and HELD-OUT IDs.

Downloads were not copied into the repository.
