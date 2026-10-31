# Study 7 Part (c) for TabDPT, Mitra and LimiX on DEV: EXPLORATORY plan and value gate

Written and committed 2026-10-29, **before any Part (c) fit of fm6/fm7/fm8** (including the
timing pilot). EXPLORATORY: outside the Study 7 pre-registration and outside every pre-registered
Holm family. DEV only (the 20 tables of `data/pool_i2.yaml`). T1′, HELD-OUT, `sealed_heldout/` and
the Study 14 T2 tables are not touched. `main.tex` is not edited by this work (macros only).

## 1. What
Study 7 Part (c) (dose-response: share s ∈ {0.5, …, 0.9} of the majority or minority class at budget
B = mult·C, mult ∈ {3, 5, 10}, 10 reps; multiclass anchor at share 1/C) was pre-registered and run
for fm1 TabPFN v2, fm2 TabICLv2, fm3 Kumo-Tabular and fm5 TabPFN v3.5. This run adds the three
exploratory FMs that already have DEV Part (a) and T1′ runs:
fm6 TabDPT v1.3, fm7 Mitra-v2 (seeded per fit) and fm8 LimiX-16M (LimiX-2 is not used).

**Design identical to the pre-registered FMs.** The designs are read, read-only, from the stored
`results/study7/c/{did}_fm2.parquet` (cell, arm, share, mult, rep, upweighted, seed, feasible,
counts, labeled; designs and contexts are learner-independent). Each stored context is re-derived
with the runner's `draw_context(y, counts, seed)` and asserted equal before the fit. Designs sharing
a seed (s = 0.5 in both arms of a binary table) are fitted once, as in `runner/study7._dose_cells`.
Each fit goes through `selection.strategies.predict_with_budget` (query = all rows of the table) and
is scored by `analysis.study7.complement_metrics`, raw and after `analysis.study7.prior_correct`
(`pc_*`), i.e. `runner/study7._fit` + `_dose_cells` re-implemented without importing the runner
(it pulls in fm1–fm5's packages). Full X, no column drop (as the main Part (c) run). Learners: the
adapters of `adapters.py` with the settings of the DEV Part (a) runs (CPU, one thread, in-context,
TabDPT 8-member ensemble, Mitra fp32 + `set_seed(cell seed)` as `run_s7a_v2.py`, LimiX default
no-retrieval config fp32), weights from the local cache only, LimiX checkpoint with
`weights_only=True`. Output schema = `results/study7/c`, written to `out/dose_c/{did}_{fm}.parquet`.

## 2. Value gate: which claim of the paper each outcome changes
The paper's Part (c) sentences (Sec. "Prescribed compositions" / "Does a prior correction remove the
drop?", App. `app:dose` inference chain, claims-table row "Drop left after p/π̂", App. `app:fmextra`
"Three further FMs", abstract line on "partly as a prior shift"):

- **S1** "macro-F1 falls in both arms for every FM (Fig. mech b,c)"; per-table raw slopes and their
  CIs "in the 8 FM–arm cells".
- **S2** "after p/π̂ the per-table slopes remain negative in all 8 cells … This residual rules out a
  pure prior shift, the pre-registered alternative, and hence pure H2".
- **S3** rule verdicts under the table-unit chain: trend in all 8 cells; TabICLv2 and Kumo
  inconclusive (minority-arm ratio CI contains 0.5); both TabPFN models unresolved/not beyond
  (ratio CI below 0.5); 0 FMs beyond.
- **S4** App. `app:fmextra`: the further FMs are reported only for Part (a) (DEV) and T1′.
- **S5** abstract: the balance gap "acts partly as a prior shift and … partly through fewer
  labelled rows per class".

**Stated plainly: no outcome of this run can change any pre-registered result or verdict.** S3, the
fig:mech panels, every `\Ssev*`, `\ExRvMc*`, `\ExRvDoseTbl*` and `\ExRvReleve*` macro are computed
from fm1/fm2/fm3/fm5 only and stay as they are. What the run can change is the **scope** of S1 and
S2 (which the text writes as "every FM" while the paper elsewhere has seven FMs, e.g. "All seven
FMs show the gap"), S4, and whether S2/S5 need a model-dependence qualifier.

| Outcome (per new FM, under the analysis of Sec. 3) | Sentence that changes | Change |
|---|---|---|
| O1 Some new FM **"beyond"**: corrected-trend Holm p < 0.05 in both arms and ratio CI > 0.5 in both arms | S4 (+ optional clause in S2) | Add: "under the same chain, X is beyond a prior shift (exploratory)". S2's "rules out a pure prior shift" gains a model where the residual is at least half the drop. S3 is unchanged (pre-registered FMs); the paper may no longer say the rule is never decisive, only that it is not decisive for the pre-registered FMs. |
| O2 **All three inconclusive or not beyond**, raw and corrected slopes negative (Holm p < 0.05 in all 6 cells) | S1, S2 scope; S4 | S1/S2 can say "every one of the seven FMs" (pre-registered 4 + 3 exploratory), with the rule as uninformative for all seven. No verdict changes. This is the expected outcome (the three FMs show the Part (a) gap on DEV and T1′). |
| O3 **Direction reversed**: some new FM's mean raw slope > 0, or its raw Wilcoxon (H1: slope < 0) not significant with slope CI containing or above 0, in some arm | S1 | S1 must be scoped to "every pre-registered FM", and a limitation added: the decline under prescribed composition is model-dependent. Abstract unaffected (it is about the gap, not the dose). |
| O3′ **Corrected slope reversed** (> 0, CI above 0) while the raw slope is negative: p/π̂ over-corrects | S2, claims-table row | S2's "remain negative" scoped to the pre-registered FMs; the row "Drop left after p/π̂" gains "over-corrects for X on DEV" (the same pattern the paper already reports on T1′). |
| O4 **Prior shift**: TOST of the per-table corrected drop passes (margin 0.01) in both arms | S2, S5 | S2's "rules out a pure prior shift … pure H2" holds only for the pre-registered FMs; one further FM is consistent with pure H2. S5 would read "partly (for one further FM, entirely) as a prior shift". This is the outcome with the largest effect on the text. |
| O5 **Mixed**: neither O1–O4 nor clean O2 (e.g. trend in one arm only) | S1/S2 scope, S4 | Report per cell; S1/S2 stay scoped to the pre-registered FMs. |

**Decision if the gate is not passed.** If the run cannot finish under Sec. 5, nothing is reported and
no sentence changes. The value is a generality check (moderate): the most likely outcome (O2) widens
two sentences from four to seven FMs; O3/O3′/O4 would prevent a scope overclaim. Cost: about 22–35
worker-hours on 4 workers (Sec. 6), no HELD-OUT exposure. The run proceeds if the pilot confirms the cost.

## 3. Analysis, fixed now (identical to r14; `analyze_dose_c.py`)
Input: `out/dose_c/{did}_{fm6,fm7,fm8}.parquet`. Step 1 = `r11_partc_common.one_file` unchanged
(rescoring raw and p/π̂ on the complement `comp` and on the rows common to all compositions `com`;
asserts the rescored complement macro-F1 equals the stored one, < 1e-9). Step 2 = r14 unchanged:
per FM × arm (majority, minority) × scoring:
- per table, OLS slope of corrected macro-F1 on s per mult, averaged over mult; one-sided Wilcoxon
  signed-rank over tables (H1: median < 0) and one-sided sign test; bootstrap 95% CI of the mean
  slope over tables (`analysis.report.bootstrap_ci`, seed 14);
- corrected/raw drop ratio (s = 0.5 to largest feasible s, mean over (table, mult)), percentile
  bootstrap 95% CI resampling tables, 10,000 resamples, seed 14;
- paired-t TOST of the per-table corrected drop, margin 0.01 (`analysis.study7.tost_p`);
- verdict per FM and scoring exactly as r14: "beyond" if Holm-Wilcoxon p < 0.05 in both arms and
  ratio CI lower bound > 0.5 in both arms; else "prior shift" if TOST p < 0.05 in both arms; else
  "not beyond" if some arm's ratio CI upper bound < 0.5; else "inconclusive".
- The raw per-table slopes (as `\ExRvDoseTbl*` for S1): exactly `review_reanalysis/acc_imb_dose.dose_table`
  (per table, OLS slope of macro-F1 and of p/π̂ macro-F1 on s over the draws of each mult, mean over
  mult; mean over tables, bootstrap 95% CI, **two-sided** Wilcoxon, Holm per metric over the cells).
  *Amendment 2026-10-29, before any Part (c) output of fm6–fm8 was analysed: the first version said
  "one-sided" here; `dose_table` (the source of S1's numbers) is two-sided, so two-sided is used.*

**Holm family handling = r14's, applied to a separate family.** r14 uses one Holm family per
scoring over the FM × arm cells of the pre-registered FMs (8). Here: one Holm family per scoring
over the **6 new cells** (3 FMs × 2 arms), for the corrected Wilcoxon and the sign test (and the raw
Wilcoxon). The 8 pre-registered cells are not re-adjusted, and none of their numbers changes.
Sensitivity (reported, not used for verdicts): Holm over all 14 cells. Outputs are tagged
EXPLORATORY; macros with prefix `ExDoseX` in `out/dose_c_numbers.tex`.

The script is checked before the run by reproducing `review_reanalysis/out/r14_mech_chain.csv`
exactly from the stored fm1/fm2/fm3/fm5 fits. Pilot fits are not analysed (timing only) and are
not part of the run; all 20 tables × all reps are run fresh.

## 4. Optimizations (only output-preserving ones)
Allowed only if a pilot comparison against the current adapters gives the same probabilities within
1e-6 (max abs) and identical metrics; the evidence is recorded in `out/dose_c_opt_check.json`.

## 5. Stopping rule
- Hard deadline: the run must be **finished by 2026-10-31 18:00 local**. A watchdog writes
  `out/dose_c/STOP` at that time; workers flush their checkpoint and exit after the current fit.
- **HELD-OUT has priority.** If the HELD-OUT opening is authorised, or the joint-opening note is
  committed, before the run ends, `out/dose_c/STOP` is written at once and the run yields all cores.
  The run never delays the opening.
- Incomplete run: if not all 20 tables × 3 FMs finish, the analysis is **not** run on a partial set
  for any FM with missing tables (no subset reporting); a finished FM (all 20 tables) may be
  analysed alone, with its Holm family reduced to its own 2 cells and this stated.
- Compute: ≤ 4 workers, `nice -n 15`, one thread per process (OMP/MKL/OPENBLAS/VECLIB, torch intra-
  and inter-op), memory guard 4 GB available, launched only after the pilot.

## 6. Cost estimate before the pilot
5,350 unique fits per FM (20 tables). From the stored DEV Part (a) seconds at m ∈ {3, 5, 10}:
≈ 5.8 (fm6), 6.7 (fm7), 9.9 (fm8) worker-hours, 22.5 in all; ≈ 6 wall-hours on 4 workers,
≈ 9 with a 1.5× margin for the shared machine. The pilot (2 tables × all levels × reps 0–1 per FM)
replaces this estimate.

## 7. Optimization check and timing pilot (added 2026-10-29, after the plan; no analysis run)
- **Output identity** (`check_dose_opt.py compare` → `out/dose_c_opt_check.json`): `--impl opt` with
  the new thread settings vs `--impl ref` with the DEV Part (a) thread settings, 52 cells per FM
  (tables 46562 and 40496, rep 0, every level and mult, different fit order): max |proba diff| = 0
  and every raw and `pc_*` metric identical for fm6, fm7 and fm8.
- **Thread settings** (`check_dose_opt.py bench`, 46562, mult 10, 6 fits): the historical settings
  leave Accelerate (numpy's and torch's BLAS on this Mac) multi-threaded, 1.16–1.27 cores per
  worker. With `VECLIB_MAXIMUM_THREADS=1` and torch inter-op = 1, CPU s per fit: fm6 5.6 → 2.7,
  fm7 11.5 → 10.2, fm8 13.8 → 12.3; wall s per fit: fm6 4.8 → 2.7, fm7 9.1 → 10.3, fm8 11.9 → 12.5.
- **Memory**: Mitra without the per-fit deepcopy: peak RSS 5.0 → 4.1 GB on 46562 (n = 1000);
  LimiX with the network built once: 1.9 → 1.6 GB. Rejected: Mitra query chunking (256 rows), with
  peak 3.2 → 1.7 GB but +30% time and max |proba diff| 2.4e-5 > 1e-6.
- **Pilot** (`out/dose_c_pilot/`, not analysed): 2 tables × all levels × reps 0–1, one worker per
  FM, opt. 98 fits per FM. Seconds per fit 1.6–3.0 (fm6), 1.9–11.8 (fm7), 1.8–13.6 (fm8); peak RSS
  1.1 / 4.4 / 1.7 GB. Projection (pilot/Part (a) time ratio 0.36 / 0.66 / 0.47 applied to the
  stored Part (a) seconds of all 20 tables): 2.1 + 4.4 + 4.6 = 11.2 worker-hours (pessimistic,
  ratio 1: 22.5); longest (table, mult) job ≤ 0.32 h.
