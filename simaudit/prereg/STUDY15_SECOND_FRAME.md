# Study 15: second-frame replication of Study 13 test (i) (paper P1)

Status: FROZEN 2026-10-31T03:36:42Z. Drafted the same day. No Study 15 fit exists at the freeze. After the freeze, changes only by dated amendments at the end, each committed before the run or analysis it affects.

Code (committed before this prereg, in its own commit R038; tested on synthetic data under tmp paths only):
- `src/simaudit/data/second_frame.py` (table specs, encoding rule, eligibility, dedup, build, loader);
- `src/simaudit/data/xlsx_lite.py` (standard-library xlsx reader for the untrusted workbooks);
- `src/simaudit/runner/study15.py` (runner; `--set second_frame`), which reuses `runner/study13.py` unchanged;
- `src/simaudit/analysis/study15.py` (analysis), which reuses `analysis/study12.py` and `analysis/study13.py` unchanged;
- `tests/unit/test_study15.py`;
- `data/pool_second_frame.yaml` (the frozen frame: per-table provenance, raw-file md5/SHA-256, content hash).

Outputs: `results/study15/second_frame/{key}_{group}.parquet` (one file per table × learner group, atomic, resumable), `results/study15/study15_results.json`, `paper/p1/study15_numbers.tex`.

## 0. Why, and what has been seen

### 0.1 Why
P1's claim C1 is that the stratified "m labels per class" protocol overstates macro-F1 relative to a random draw of the same budget. Study 13 confirmed test (i) on HELD-OUT. HELD-OUT and DEV were drawn from one frame (small all-numeric OpenML tables, n ≤ 1000; Study 13 Sec. 10, "Same population as DEV"). T1′ is a second frame, but it is not blind (Study 12 Sec. 0). Both are public tables that may sit in the FMs' pre-training or validation corpora. Study 15 tests (i) once more on a **frame that differs in source, era and preprocessing**: 20 tables first published on Zenodo, figshare or Kaggle after 2026-07-01, i.e. after the release of the FM checkpoints listed per table in the pool file.

### 0.2 Seen before this draft (disclosed)
- **DEV, T1′:** everything, as in Study 13 Sec. 0.2.
- **HELD-OUT:** the Study 13 joint results (`results/joint/heldout/joint_results.json`, Study 13 rows). Seen: the (i) means 0.062 / 0.030 / 0.044 / 0.075 (fm1 / fm2 / fm3 / fm5), their SDs 0.047 / 0.041 / 0.041 / 0.052, n = 32, all four "replicated"; (ii-a) replicated, (ii-c) tie established. They are used only for power (Sec. 7) and for a descriptive ratio (Sec. 4.3). `sealed_heldout/` was not opened; HELD-OUT enters the dedup only through the hash list in `data/pool_i2.yaml`.
- **Second frame (this study's data):** only what the build needs, with no model fitted and no feature–label relation computed: file and sheet layout, column names, the source notes (`research_notes/Systematic discovery 2026-10/20_memorisation_free_set.md` §2, read-only), n, d, C and class counts after encoding, and the label strings. The dedup checks (Sec. 1.5) compared rows with DEV and T1′ only.
- **Peer use.** Another study (AUD-43) uses the same SAFE tables as confirmation data for a different claim (conformal coverage). Its fetch log and build script were read (read-only) to copy 12 raw files and their URLs. Its targets were compared with the source notes; Study 15 declares its own targets and drops (Sec. 1.2). No AUD-43 result was seen, and no Study 15 result goes to that study (Sec. 9).

## 1. Data

### 1.1 Frame
- The 20 rows with `status_pool == SAFE` of `research_notes/Autoresearch idea bank/POOL_POS_FM_2026-10-30.csv`, whose first public version is after 2026-07-01: 13 Zenodo, 6 figshare, 1 Kaggle.
- **Study 14 T2.** These rows have no OpenML IDs, and the Study 14 T2 tables are all OpenML, so the two sets cannot overlap. No Study 14 table is used.
- **Raw files.** They are stored in `data/cache/second_frame/raw/<key>/`, a new folder outside git. 12 were copied from the peer cache after their md5 matched its fetch log; the other 8 were downloaded from the original URLs. Every md5 equals the repository's published md5 (Zenodo `checksum`; figshare `computed_md5`), except the Kaggle archive, whose repository publishes none. The pool CSV's `md5_file` column is empty for all 20 rows, so it could not be checked against.
- All raw files are untrusted. They are read as text (csv module) or with `data/xlsx_lite.py` (no formula evaluated, no macro, no external link), by `python -I`, with paths passed as arguments.

### 1.2 Tables: target, declared drops, provenance
**Target choice.**
- The target is the one in the source notes (§2.1/§2.2).
- When the notes list two candidates, the first is used, except for the food-waste table. There the binary target given with its class distribution (`waste_occurrence`, 491/91) is used, because the pool CSV records C = 2.
- `Mortality` (ED mortality) keeps its 6 classes, as the pool CSV records (C = 6). It is not binarised.

**Declared drops** (all from the notes or the column documentation; `second_frame.SPECS` holds the exact lists):
- identifier columns (name rule `ID_NAME`, plus `Tree_ID`);
- model outputs (`Predicted - PVT`, `Prediction`);
- post-outcome and target-defining columns: HPV genotype/assay columns; PPD, CAL and CPI for periodontal status; 2026 tree girths; `*waste*`, `discard_frequency`, `cause_*`, `solution_*`; pain sites and `nrs`; `Resuscitation*`, discharge/death diagnosis; `gr` (the binary collapse of PCOS `group`); `Academic_Discipline_Group` and `College_CAT` for `Nursing_Group`; the Polish "current year of study" column, whose text contains the degree course (the target `kierunek`);
- dates and times;
- one duplicate column (`swb` = `life_satisfaction`).

**Layout rules.**
- Japanese pain survey: only the `women` sheet (n = 3000, as the pool CSV), header on its second row.
- Heart-failure frailty: the published training and testing files are one cohort split 70/30 by the authors. They are concatenated (n = 1498, not the 450 of the testing file alone that the pool CSV records), aligned by column position, with the header on the second row (the first row holds value codes).

| key | source | first public | licence | target | n | d | C | class counts | TabPFN-3.5-safe | raw md5 | content SHA-256 (12) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k11395735_malaria | Kaggle 11395735 | 2026-07-28 | CC0-1.0 | `Result` | 2190 | 12 | 2 | 1068/1122 | no | 3068a32b… (zip); 5b71df58… (csv) | cb6b02710a34 |
| f33111881_frailty | figshare 33111881 | 2026-07-29 | CC BY 4.0 | `Frailty` | 1498 | 81 | 2 | 295/1203 | no | 02151291…; 380417e3… | a6e42d4f5640 |
| f33126371_nonadh | figshare 33126371 | 2026-07-30 | CC BY 4.0 | `Adherence` | 270 | 19 | 2 | 117/153 | no | 4acdc870… | 3becc74f24b9 |
| z21713220_optiaged | Zenodo 21713220 | 2026-07-31 | CC-BY-4.0 | `CJP_COMPOSITE` | 2483 | 18 | 2 | 1799/684 | no | e3b63193… | b14914c6f565 |
| z21872549_vacc | Zenodo 21872549 | 2026-08-10 | CC-BY-4.0 | `kierunek` | 1096 | 16 | 2 | 530/566 | no | e79bcd1a… | dd17a7db49ea |
| z22705412_hpv | Zenodo 22705412 | 2026-08-13 | CC-BY-4.0 | `hpv_result` | 1193 | 30 | 2 | 85/1108 | no | 7ff34981… | 7847906906d4 |
| z22065799_school | Zenodo 22065799 | 2026-08-23 | CC-BY-4.0 | `Q4. Current CGPA` | 209 | 63 | 5 | 24/50/64/48/23 | no | 728863fb… | b79918fd94e4 |
| z22133098_trees | Zenodo 22133098 | 2026-08-27 | CC-BY-4.0 | `Status_2026` | 248 | 3 | 2 | 224/24 | no | 7244ef69… | 6aa77295743c |
| z22172178_perio | Zenodo 22172178 | 2026-08-30 | CC-BY-4.0 | `Periodontal_status` | 211 | 12 | 2 | 29/182 | no | e8efa748… | 33083fe9bb92 |
| z22216010_pvt | Zenodo 22216010 | 2026-08-31 | CC-BY-4.0 | `PVT` | 1463 | 13 | 2 | 1307/156 | no | d666e1a6… | 494d904ab17f |
| z22751055_fairtrade | Zenodo 22751055 | 2026-09-14 | CC-BY-4.0 | `credit_access` | 284 | 5 | 2 | 130/154 | no | 6af4aa99… | 3ef6e0e0cb67 |
| z22767593_edmort | Zenodo 22767593 | 2026-09-15 | CC-BY-4.0 | `Mortality` | 597 | 52 | 6 | 45/16/84/31/263/158 | no | a4b85687… | 962c8bfe9d90 |
| f33868873_adherence | figshare 33868873 | 2026-09-17 | CC BY 4.0 | `adherence` | 597 | 12 | 2 | 463/134 | no | d36cb0aa… | 0ce98536c6c6 |
| z22900584_foodwaste | Zenodo 22900584 | 2026-09-22 | CC-BY-4.0 | `waste_occurrence` | 582 | 24 | 2 | 91/491 | no | b39c568f… | d9ffb0014c64 |
| z23293577_pain | Zenodo 23293577 | 2026-09-24 | CC-BY-4.0 | `pain` | 3000 | 24 | 2 | 2567/433 | yes | ca31e473… | fb0e1afdf6dc |
| f33328002_tea | figshare 33328002 | 2026-10-10 | CC BY 4.0 | `adoption` | 896 | 19 | 2 | 447/449 | yes | ab490cf5… | ee217f22f5bd |
| z23318225_hlit | Zenodo 23318225 | 2026-10-12 | CC-BY-4.0 | `long_standing_condition` | 448 | 68 | 2 | 258/190 | yes | 0896154f… | 56450cc59f0f |
| f34554682_pcos | figshare 34554682 | 2026-10-15 | CC BY 4.0 | `group` | 498 | 20 | 3 | 163/127/208 | yes | 36b1a7b2… | 9ea138402290 |
| z23168913_volunteer | Zenodo 23168913 | 2026-10-19 | CC-BY-4.0 | `Nursing_Group` | 388 | 15 | 2 | 276/112 | yes | 963ba772… | b5baaaabe3e4 |
| f34737363_pneumonia | figshare 34737363 | 2026-10-21 | CC BY 4.0 | `分组` | 234 | 33 | 2 | 154/80 | yes | 100b6b94… | 50b786ad0074 |

- The full URLs, full md5/SHA-256 values and full content hashes are in `data/pool_second_frame.yaml`, with the fetch log in `data/cache/second_frame/fetch_log.json`.
- 17 tables are binary and 3 are multiclass (C = 3, 5, 6). Σn = 18,385.
- "TabPFN-3.5-safe" means that the pool CSV's `apos_FM` lists "TabPFN-3.5", so the table was first published after that checkpoint too.

### 1.3 Encoding rule (the same for every table; `second_frame` module docstring)
1. Cells are stripped. The missing tokens are `""`, `NA`, `N/A`, `nan`, `null`, `#NULL!`, `#N/A`, `-`, `.` and `?`, case-insensitive. ("None" is *not* missing: it is a real `Mortality` level.) Rows whose target is missing are dropped.
2. A feature is numeric when ≥ 95% of its non-missing cells parse as numbers (a decimal comma is accepted); cells that do not parse then become missing. Otherwise it is nominal, coded by the sorted order of its strings.
3. Nominal features with more than 100 levels are dropped (free text, IDs). So are features that are entirely missing or constant.
4. **Missing values are imputed without labels, from all rows:** numeric features by the column median, nominal features as one extra level. This is needed because the classical learners, the integrity screen and the z-scoring cannot take NaN. It is applied identically for every protocol and draw.
5. Labels: the Study 13 Amendment 1 (A) spelling rule (`data.labels`), with codes in the sorted order of the raw strings. One table merges spellings: Kaggle malaria (`Positive | negative | positive` → 2 classes).

### 1.4 Eligibility (Study 13 / pilot I2 rule)
- 2 ≤ C ≤ 10, smallest class ≥ 11, n ≥ 100, d ≥ 1 after encoding. **All 20 tables pass, so no table is excluded.**
- Unlike HELD-OUT, n reaches 3000. There is no panel cap: arm T scores the full complement, as in Study 13 (Sec. 8 costs this).

### 1.5 Dedup (existing tools)
- Each table's (X, y) content hash (`openml_loader.content_hash`) was compared with the 20 DEV and 32 HELD-OUT hashes in `data/pool_i2.yaml` and with the 52 T1′ tables' hashes, computed from the non-sealed npz cache. **No match.**
- The row-overlap share (`integrity.overlap_share`, Study 8 A2.3b; exclusion at ≥ 0.5) was computed against DEV and T1′. **The maximum was 0.00.**
- No HELD-OUT row-level check is possible without opening `sealed_heldout/`, and none was done. Exact HELD-OUT duplicates are excluded by the hash check, and the frame's sources (non-OpenML, post-2026-07) make a HELD-OUT overlap implausible.

### 1.6 Integrity screen, failures
- The Study 13 A2.7 screen runs before any fit and only flags tables (`integrity_screen.csv`). Its flags define SENS-INT and SENS-ID (Sec. 5). No table is excluded and no column is dropped by the screen.
- **Failures.** A failed job is retried once. A table that still fails for an FM is dropped from the contrasts that need that FM (complete case). If more than 10% of the 20 tables (i.e. n < 18) is lost for a contrast, the contrast is "not evaluable", not "not replicated".

## 2. Design (Study 13 Sec. 2, reused unchanged)
- **Identical to Study 13:**
  - row permutation (namespace 35);
  - selection/fit seed `SeedSequence([base_seed, did, m, rep, strategy index, 34, arm])`;
  - redraw seeds;
  - fixed split T_d (namespace 36);
  - base seed 20261020;
  - arm T (stratified 5 reps, random 20 reps, m ∈ {1, 3, 5, 10}, B = m·C, scored on the complement; redraw arm with infeasibility recorded);
  - arm F;
  - learners fm1 (Real-TabPFN, tabpfn 9.2.0 V2 default checkpoint), fm2 (TabICL v2), fm3 (Kumo), fm5 (TabPFN v3.5) and the six classical learners;
  - fm1_bal / fm5_bal;
  - every per-fit metric;
  - CPU, one thread.
- **Table IDs.** did = source offset + repository ID (Zenodo 10¹⁰, figshare 2·10¹⁰, Kaggle 3·10¹⁰). These are disjoint from every OpenML/PMLB/UCI ID, so every seed is new while the namespace scheme stays Study 13's.
- **Not run:** fm6–fm8 (Study 13 Amendments 1/3), to save compute; they carry no P1 claim on this frame.

## 3. Primary hypotheses (confirmatory family of 4)
**Unit.** The table: mean over reps per (table, learner, strategy, m), then mean over m ∈ {1, 3, 5, 10}. Arm T. Macro-F1 over all C classes.

| ID | Contrast | Test | Prediction |
|---|---|---|---|
| (i)-fm1, (i)-fm2, (i)-fm3, (i)-fm5 | stratified − random, per FM | Wilcoxon signed-rank, two-sided | Δ > 0 for every FM |

**Holm over the 4 FMs** at α = 0.05. Robustness is reported for each contrast as in Study 13 Sec. 3 (`analysis.study12.summarize`): mean, median, win rate, bootstrap 95% CI, TOST p, Bayesian signed-rank, leave-one-table-out.

## 4. Decision rules
### 4.1 Per FM (Study 13 Sec. 4.1)
- **Replicated:** Holm p < 0.05 and mean Δ > 0.
- **Reversed:** Holm p < 0.05 and mean Δ < 0.
- **Does not replicate:** otherwise.
- **Not evaluable:** n < 18 (Sec. 1.6).

### 4.2 Frame claim (Study 13 C1 rule)
**C1 replicates on the second frame** iff (i) is replicated for ≥ 3 of the 4 FMs **and** no (i) contrast is reversed. Otherwise it does not replicate.

### 4.3 Magnitude (descriptive)
For each FM, the report gives the ratio of the second-frame mean to the HELD-OUT mean, and whether the HELD-OUT mean lies inside the second-frame CI. A replicated FM with ratio < 0.5 is reported as "replicated, attenuated".

### 4.4 Consequence for P1
- **If C1 replicates,** P1 states that the overstatement holds on a post-FM frame of different provenance.
- **If C1 does not replicate,** P1 states that, with the estimates and CIs, and narrows C1 to the OpenML frames (DEV/HELD-OUT) plus T1′.
- **If any FM is reversed,** that is reported in the abstract-level summary of the robustness section.
- No re-run, re-seed or re-encoding follows any outcome.

## 5. Secondary (no confirmatory weight)
- **(ii-a)** fm2 − fm5 under random and **(ii-c)** fm2 − fm5 under stratified (TOST, margin 0.01), mean over m ∈ {3, 5, 10}, exactly as Study 13. They form their own Holm family of 2, with Study 13's verdict labels.
- **SENS-INT / SENS-ID:** the primary family recomputed without the screen-flagged tables (Holm within the reduced family). A verdict that changes is reported as fragile.
- **Per-budget gaps** per FM, and the accuracy and balanced-accuracy gaps (unadjusted, no direction).
- **S-1 (classical learners):** Study 13's function, Holm over 6.
- Study 13 S-2 to S-14 are not analysed. Arm F is run, because the runner is reused unchanged, but it is not analysed unless an amendment declares it before the analysis.

## 6. Exploratory (descriptive; no test enters any family)
- **TabPFN-3.5-safe subset** (6 tables: pain, tea, hlit, pcos, volunteer, pneumonia; first published after the TabPFN 3.5 checkpoint): the fm5 gap, with mean, CI and unadjusted Wilcoxon p. With n = 6, the smallest possible two-sided p is 0.031.
- **Multiclass subset** (3 tables) and **binary subset** (17 tables): every FM's gap, descriptive only.

## 7. Power (n = 20; from the HELD-OUT SDs, Sec. 0.2)
Wilcoxon two-sided, normal approximation with ARE 0.955, at the strictest Holm step α = 0.05/4:

| FM | HO mean / SD | d | power if HO effect | power if half the HO effect | absolute MDE (80%) |
|---|---|---|---|---|---|
| fm1 | 0.062 / 0.047 | 1.31 | 1.00 | 0.64 | 0.036 |
| fm2 | 0.030 / 0.041 | 0.73 | 0.76 | 0.19 | 0.031 |
| fm3 | 0.044 / 0.041 | 1.06 | 0.98 | 0.42 | 0.032 |
| fm5 | 0.075 / 0.052 | 1.43 | 1.00 | 0.74 | 0.040 |

- The standardised MDE is d = 0.76 (n = 20), or 0.81 if n = 18.
- fm2 has the smallest effect and is the most likely "does not replicate" even if the effect is real. This is stated now.
- C1 needs 3 of 4 FMs, so it stays well powered at the HO effects (P(≥ 3 of 4) ≈ 0.99 if independent).

## 8. Host, run plan, compute
- **Host:** <host> (16 cores, 48 GB), local only, one host. The runner records the host name and `PYTHONHASHSEED` in `provenance.json`.
- **Launch (after the freeze commit):**

  ```
  cd simaudit && PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nohup nice -n 15 uv run python -m simaudit.runner.study15 --set second_frame --workers 6 > logs/study15_second_frame.log 2>&1 &
  ```

  - At most 6 local workers.
  - Group order: fm2, fm5, fm1, fm3, classical.
  - fm5 uses the TabPFN token that the library reads itself.
- **Guard:** the runner refuses to start unless this file reads `Status: FROZEN` and is committed unchanged against HEAD.
- **No interim look.** The analysis (`python -m simaudit.analysis.study15`) runs once, after every FM job has finished or definitively failed.
- **Compute estimate.**
  - Sources: Study 13 Sec. 8 fit times for n ≤ 1000 (TabICL 2.2 s, TabPFN v3.5 3.7 s, TabPFN v2 ≈ 4.4 s, Kumo ≈ 8.8 s; 19.1 s per cell for the four FMs at a mean n of ≈ 418). These are scaled linearly with n (complement scoring), Σn = 18,385.
  - Arm T: 120 × 19.1 s × 18,385 / 418 ≈ 28 worker-hours. Arm F: ≈ 0.6 × that ≈ 17 worker-hours. Classical: ≈ 1.
  - **Total ≈ 46 worker-hours (≈ 70 with a 1.5× margin), ≈ 8–12 wall-hours at 6 workers.**
  - Synthetic smoke (2026-10-31, 12 context rows, 300 queries, one thread): fm2 1.1 s, fm5 4.5 s, fm1 1.2 s, fm3 6.0 s, including model load.

## 9. Threats
- **Contamination.**
  - The frame postdates fm1 (Real-TabPFN v2) and fm2 (TabICL v2) by construction.
  - For fm5 (TabPFN v3.5), 14 of 20 tables predate its checkpoint; hence the exploratory safe subset.
  - Kumo's training cut-off is not documented, so fm3 has no verified contamination status.
- **Pre-publication of summaries.** Class shares may have appeared earlier in an article (source notes §1.3). This matters for LLM priors, much less for in-context FMs.
- **Kaggle repost.** The Kaggle malaria table has no description and could be a repost; the pool CSV marks it SAFE.
- **Different preprocessing.** HELD-OUT had no missing values or nominal features; here imputation and ordinal coding are used. This is part of the frame difference the study is meant to probe, not a confound to remove.
- **Shared use.** AUD-43 uses these tables for a different claim. No Study 15 result (per-table or aggregate) is shared with that study before its analysis is locked. Results live only under `simaudit/results/study15/` and `paper/p1/`.
- **Small n.** 20 tables; the multiclass and safe subsets are descriptive only.
- **One host, shared machine.** Timing does not affect results.

## 10. Deviations from Study 13 (all declared here)
1. New frame and loader (`second_frame`), with a declared encoding rule including label-free imputation (Sec. 1.3); the label spelling rule is applied at build time, not through the sealed raw cache.
2. n up to 3000, no panel cap.
3. Confirmatory family = test (i) only, Holm over 4. (ii-a)/(ii-c) are secondary (Holm over 2). There is no cross-study Holm, because this frame is not shared with any confirmatory study.
4. Guard = frozen-and-committed prereg (no HELD-OUT go variable, no manifest).
5. fm6–fm8 not run; Study 13 S-2 to S-14 not analysed; arm F run but not analysed.

## 11. Log and amendments
- **2026-10-31:** draft, then frozen (2026-10-31T03:36:42Z; code commit R038, `make verify` green: 932 tests, coverage 96.5%, CRAP gate 0 failing). The frame was built (`results/study15/second_frame/frame_build.json`, `data/pool_second_frame.yaml`): 20 of 20 eligible, no hash or row-overlap match, no exclusion.
- Two encoding-rule corrections were made before freezing, on metadata only:
  - "none" was removed from the missing tokens, because it removed the `Mortality` level "None" (n 334 → 597, C 5 → 6);
  - the ≥ 95% numeric rule was added, because a few unparsable cells such as "<5" had turned laboratory columns into nominal ones.
- Neither change involved any model or any feature–label statistic.
