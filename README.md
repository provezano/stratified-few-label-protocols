# Stratified Few-Label Protocols Can Overstate Macro-Averaged Metrics for Tabular Foundation Models

Code, pre-registrations, stored results and audit materials for the paper of the same title by Felipe Provezano Coutinho (under review at TMLR).

Few-label evaluations of tabular foundation models (FMs) usually draw the labelled set stratified by class, so that every class is present in the context. This repository holds everything needed to check how that choice changes reported macro-averaged metrics compared with drawing the labelled rows at random: the pre-registered studies on public OpenML, PMLB and UCI tables, a second pre-registered replication frame, the analysis code, the stored per-fit scores, and the materials of the audit of protocols reported in the literature. Every number, table and figure of the paper is written by a script in this repository into a LaTeX macro file, a CSV file or a figure, and can be regenerated from the stored results without refitting any model.

## Citation

If you use this code or these results, please cite the paper:

```bibtex
@misc{coutinho2026stratified,
  title         = {Stratified Few-Label Protocols Can Overstate Macro-Averaged Metrics for Tabular Foundation Models},
  author        = {Coutinho, Felipe Provezano},
  year          = {2026},
  eprint        = {XXXX.XXXXX},  % TODO: replace with the arXiv identifier once assigned
  archivePrefix = {arXiv},
  note          = {Under review at Transactions on Machine Learning Research (TMLR)},
  url           = {https://github.com/provezano/stratified-few-label-protocols}
}
```

The arXiv identifier (`arXiv:XXXX.XXXXX`) is a placeholder and will be filled in once assigned. `CITATION.cff` holds the same reference in machine-readable form.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it if needed). Experiments run on CPU.

```bash
git clone https://github.com/provezano/stratified-few-label-protocols.git
cd stratified-few-label-protocols/simaudit
uv sync --frozen                     # Python 3.12, exact versions pinned in uv.lock
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1

# regenerate the main macro file and the per-dataset tables from the stored results (no data, no network)
cd paper/p1 && uv run python make_numbers.py && uv run python make_tables.py && cd ../..
git diff --stat                      # empty: the regenerated files match the committed ones

# download the public tables (checked against pinned SHA-256), needed by scripts that read labels
uv run python ../fetch_data.py --case-study
```

`uv sync --frozen` installs the full environment, including PyTorch and the FM packages (TabPFN, TabICL, Kumo-Tabular). Regenerating numbers, tables and figures from the stored results needs only the scientific Python stack; the continuous-integration workflow (`.github/workflows/ci.yml`) shows a light install with versions taken from `uv.lock`.

## Conventions

Paths are relative to `simaudit/` unless they start with `datasets/`, `environment/`, `seeds/` or `REVISIONS.md`.

**Names used in the files.** "P1" is the internal working name of this paper. DEV, T1′ (the PMLB/UCI set of the paper) and HELD-OUT are the three table sets; the second frame is the set of Study 15. fm1 = TabPFN v2, fm2 = TabICLv2, fm3 = Kumo-Tabular, fm5 = TabPFN v3.5 (the four pre-registered FMs); fm6 = TabDPT v1.3, fm7 = Mitra-v2, fm8 = LimiX-16M (exploratory). Study numbers follow Table `tab:studies` of the paper.

## Layout

| Path | Content |
|---|---|
| `simaudit/src/simaudit/` | The experiment package: data loaders, samplers (stratified, random, selectors), FM wrappers, runners and the frozen analysis code of every study. |
| `simaudit/pyproject.toml`, `simaudit/uv.lock`, `simaudit/Makefile` | The environment (Python 3.12, exact versions) and the main commands. |
| `simaudit/prereg/` | The pre-registrations, with their amendments, addenda, freeze notes and errata at the end of each file (see "Documents"). |
| `simaudit/reports/` | The reports that `make_numbers.py` and other macro generators parse, the scripts that write them, and `study12_ERRATA.md`. |
| `simaudit/results/` | Stored scores of Studies 4, 7 (with S4), 9 (DEV and T1′), 12, 15 and the protocol corrections: one row per fit, with labelled rows, seeds, metrics and run time; the per-row probability columns are left out (see below). |
| `simaudit/data/` | The table pools: `pool_i2.yaml` (DEV only), `pool_t1prime.yaml`, `pool_ext_pmlb.yaml`, `pool_ext_uci.yaml`, `pool_second_frame.yaml`. |
| `simaudit/paper/p1/` | The generators of every macro file, table and figure, with their outputs. |
| `simaudit/paper/p1/exploratory/` | The exploratory analyses: scripts, cited plans, per-table outputs and macro files. |
| `simaudit/paper/p1/supplementary/audit/` | The audit of reported protocols (Section 6.1 and the appendix "Audit of reported protocols"). |
| `simaudit/paper/p1/exploratory/human_check/` | The blind human check of the audit sample, with its adjudication. |
| `datasets/` | • `dev.csv`: OpenML id, version, content hash.<br>• `t1prime.csv`: PMLB/UCI key, DOI, licence, file and content hash, n, C, stratum.<br>• `sources.json`: PMLB commit, UCI API, selection seeds. |
| `environment/` | `freeze.txt` (Python and every installed distribution) and `models.json` (checkpoint file names, revisions, SHA-256). No weights. |
| `seeds/` | The provenance file of every run of Studies 4, 7, 9, 12 and the corrections: base seed, seed namespaces, grids, repetitions, learners, dataset ids, code digest. Study 15: `results/study15/second_frame/provenance.json`. |
| `simaudit/README.md` | Pointer to this file (package metadata). |
| `REVISIONS.md` | The revision identifiers that the appendix "Pre-registration log" cites (R001–R013, R015–R018), each with its date, file and a one-line description. R014 is an uncited alias of R012 (same commit). Later identifiers replace other commit hashes quoted in the files. |
| `fetch_data.py` | Downloads the public DEV, T1′ and case-study data and checks each file against a pinned SHA-256 (no third-party data is included). |
| `LICENSE`, `LICENSE-DATA`, `CITATION.cff` | Licences (see "Licence") and citation metadata. |
| `.github/workflows/ci.yml` | Continuous integration: lock check, byte-compilation and regeneration of five macro files from the stored results. |

## Documents

Only the following Markdown documents are included. Internal working notes, plans that the paper does not cite, and generated summaries (`*_SUMMARY.md`, `TABLES.md`, report prose) are left out: their numbers are in the macro files and CSV outputs. The included documents still name some of these notes (for example a file under `research_notes/`, an `E*_SUMMARY.md` or a `PLAN.md`); those references are kept as written, and the notes are not part of this repository.

| Document | Role |
|---|---|
| `prereg/STUDY7_PROTOCOL_MECHANISM.md` | Study 7 (DEV study), with Addendum 1. |
| `prereg/STUDY9_HOW_MANY_LABELS.md` | Study 9 (learning-curve study), Amendments 1–4, Freeze note 1. |
| `prereg/STUDY12_PROTOCOL_T1PRIME.md` | Study 12 (PMLB/UCI study), Addendum 1, Amendment 2, Erratum 1. |
| `prereg/STUDY13_HELDOUT_CONFIRMATION.md` | Study 13 (HELD-OUT study), Amendments 1–3; run once at the joint opening on 2026-10-30. |
| `prereg/JOINT_OPENING_2026-10-30.md` | The joint HELD-OUT opening note (R018): studies in the event, confirmatory cells, the cross-study Holm family of 19 tests, launch settings. |
| `prereg/STUDY15_SECOND_FRAME.md` | Study 15 (second frame), frozen 2026-10-31 before any fit. |
| `prereg/PILOT_I2.md`, `prereg/STUDY4_LABEL_SELECTION.md`, `prereg/STUDY8_EXPANSION.md` | Context: the pilot defines DEV; Study 4 is the first study (exploratory here); Study 8 builds the pool from which T1′ comes. |
| `reports/STUDY7_REPORT.md`, `STUDY9_T1PRIME_OUTPUT.md`, `STUDY9_T1PRIME_REPORT.md`, `STUDY12_REPORT.md`, `STUDY12_UCI257_EXPLORATORY.md`, `STUDY4_BASELINES.md`, `STUDY4_STAGE_A.md`, `THEORY_CHECKS.md` | Generated reports that the macro generators parse. |
| `reports/P1_PROTOCOL_CORRECTIONS_EXPLORATORY.md` | Plan and criteria of the protocol corrections, committed before any correction was computed (R013), followed by the results; parsed by `make_numbers.py`. |
| `reports/study12_ERRATA.md`, `paper/p1/exploratory/case_study/ERRATA.md` | Errata of Study 12 and of the case study. |
| `paper/p1/heldout_branches.md` | The plan, fixed before the HELD-OUT opening, that maps each HELD-OUT outcome to the wording of each claim (Section 5, Data). |
| `paper/p1/exploratory/b_dose_response_plan.md` | Plan of the dose arms F, S and N on DEV, written before any fit. |
| `paper/p1/exploratory/b_dose/PREREG_T1P_SUBSET.md` | Pre-registration of the dose arms on a T1′ subset (R001). |
| `paper/p1/exploratory/case_study/PREREG.md` | Pre-registration of the TabEBM case study (R002), with its amendments. |
| `paper/p1/exploratory/fm_extra/limix_t1p/PLAN.md`, `fm_extra/dose_plan.md` | Plans of the LimiX run on T1′ and of the dose experiment for fm6–fm8 on DEV, fixed before the runs. |
| `paper/p1/exploratory/seed_indep/PLAN.md` | Plan and reading rule of the independent-seed check, fixed before the run. |
| `paper/p1/exploratory/e17_metrics/e17_report.md` | Section 1 holds the expected directions of the metric panel, written before any E17 metric was computed; the rest reports E17. |
| `paper/p1/exploratory/tabdpt_check/NOTE.md` | The documentary source of the `\Tdpt*` counts (TabDPT pre-training corpus). |
| `paper/p1/exploratory/human_check/ADJUDICATION.md` | Adjudication of the disagreements between the blind human codes and the final codes. |
| `paper/p1/supplementary/audit/00_PROTOCOL.md`, `01_CODING_MANUAL.md`, `fulltext/00_PROTOCOL_FULLTEXT.md`, `fulltext/stage2_coder_instructions.md` | The audit protocol, the coding manual given to every coder, the supplementary full-text protocol and the coder instructions. |

## Where each number comes from

Run every command from `simaudit/` with the package environment (`uv run python ...`), unless the command starts with `cd`. The last column marks the results that are re-scored from the per-fit probabilities. Those scripts do not run from this repository; their derived outputs are included and reproduce the numbers (see the section after the tables).

### Macro files read by the paper

All macro-file paths below are relative to `paper/p1/`.

| Macro file | Macros | Script and command | Inputs | Per-fit probabilities |
|---|---|---|---|---|
| `numbers_p1.tex` | `\Ssev*` (Study 7), `\Snine*` (Study 9), `\Stwelve*` (Study 12), `\Th*` (theory checks), `\Sfour*` (Study 4), `\Cfour*` and `\Cfive*` (protocol corrections), `\SysAudit*` and `\SuppFt*` (audit), `\Repro*`, `\Cal*`, `\Ver*`, `\Held*` | `cd paper/p1 && uv run python make_numbers.py` | `reports/STUDY7_REPORT.md`, `STUDY9_T1PRIME_OUTPUT.md`, `STUDY9_T1PRIME_REPORT.md`, `STUDY12_REPORT.md`, `STUDY12_UCI257_EXPLORATORY.md`, `THEORY_CHECKS.md`, `STUDY4_BASELINES.md`, `STUDY4_STAGE_A.md`, `P1_PROTOCOL_CORRECTIONS_EXPLORATORY.md`; `prereg/STUDY7_PROTOCOL_MECHANISM.md`, `prereg/STUDY9_HOW_MANY_LABELS.md`; `supplementary/audit/{analysis_output.json, audit_numbers.json, coding/coder3_*, fulltext/analysis_supp.json, screening/stage2_decisions.csv}`; `results/*/provenance_*.json` and the `seconds` column of `results/`; `exploratory/e17_metrics/e17_moderators.csv`; `exploratory/verify/out/v_checks.txt`; `exploratory/c4_t1p/out/c4_t1p_numbers.tex`. Four constants (`\DevSpentDatasets`, `\DevSpentEvals`, `\GrinsztajnSorted`, `\TabRepoSorted`) come from an internal note that is not included; their comments say so. | no (it parses the included reports) |
| `tables_p1.tex`, `tables_numbers_p1.tex`, `tables_metrics_p1.tex`, `tables_redraw_p1.tex` | Per-dataset tables, metric-convention table (`tab:metrics`), redraw table (`tab:redraw`) | `cd paper/p1 && uv run python make_tables.py` | `results/study7/a/*.parquet`, `reports/STUDY7_REPORT.md`, `exploratory/out/e10_pools_{dev,t1prime}.csv`, `results/study9/t1prime/*.parquet` | no |
| `exploratory/out/exploratory_numbers.tex` | `\Ex*` (E1–E15, dose arms `\ExB*`, extra FMs `\ExFM*`) | `uv run python paper/p1/exploratory/make_exploratory_numbers.py` | `exploratory/out/*.csv`, `*.parquet` (from `e1_classical.py` … `e15_budget_metrics.py`), `b_dose/out*/*.csv`, `fm_extra/out/{s7a,s7a_v2,t1p}_summary.csv`, `e17_metrics/*.csv` | E2, E7, E8, E15 intermediates: yes (included as `exploratory/out/*.csv`) |
| `exploratory/case_study/out/case_study_numbers.tex` | `\ExCaseEBM*` | `uv run python paper/p1/exploratory/case_study/code/analyze.py` | `case_study/out/stage_a/*.json`, `out/stage_b/*.json`, `out/runs_meta.csv` | no |
| `exploratory/out/e16_numbers.tex` | E16 (multiclass decomposition) | `uv run python paper/p1/exploratory/e16_multiclass/e16_numbers.py` | `e16_multiclass/out/e16_*.csv`, `out/classes.parquet` | extraction `e16_extract.py`: yes |
| `exploratory/out/e17_numbers.tex` | E17 (metric panel) | `uv run python paper/p1/exploratory/e17_metrics/e17_metrics.py` | its draw cache `e17_metrics/e17_draws.parquet`; first build: `results/study7/a`, `results/study9/t1prime` | first build: yes |
| `exploratory/uci257/out/uci257_numbers.tex` | UCI 257 label sensitivity | `uv run python paper/p1/exploratory/uci257/uci257_sensitivity.py` | `results/study9/t1prime/pmlb_1000000257_*.parquet` (included whole), the raw UCI 257 file, the UCI API snapshot (not included) | yes (included for this table) |
| `exploratory/distpfn/out/distpfn_numbers.tex` | Prior corrections on DEV (`tab:distpfn`) | `uv run python paper/p1/exploratory/distpfn/run_distpfn.py` | cache `distpfn/out/runs.parquet` (included); first build: `results/study7/a` | first build: yes |
| `exploratory/distpfn/t1p/out/distpfn_t1p_numbers.tex` | Prior corrections on T1′ (`tab:distpfntp`) | `uv run python paper/p1/exploratory/distpfn/t1p/run_t1p.py` | cache `distpfn/t1p/out/runs.parquet` (included); first build: `results/study12` | first build: yes |
| `exploratory/claims_audit/out/claims_audit_numbers.tex` | `\ExCa*` | `rho_all.py`, `redraw_tost.py`, then `make_claims_numbers.py` (all in `exploratory/claims_audit/`) | `claims_audit/out/{rho_all,redraw_tost}.csv`, `e17_metrics/*.csv`, `reports/STUDY12_REPORT.md`, `fm_extra/out/t1p_summary.csv` | no |
| `exploratory/e1_t1p/out/e1t1p_numbers.tex` | Classical learners on T1′ | `uv run python paper/p1/exploratory/e1_t1p/analyze_e1_t1p.py` (runs: `run_e1_t1p.py`) | `e1_t1p/out/runs/*.parquet`, `results/study12` | no |
| `exploratory/c4_t1p/out/c4_t1p_numbers.tex` | Correction 4 on T1′ | `uv run python paper/p1/exploratory/c4_t1p/run_c4_t1p.py` | cache `c4_t1p/out/ps.parquet` (included), `reports/p1_corrections.py`; first build: `results/study12` | first build: yes |
| `exploratory/fm_extra/out/t1p_fm8_numbers.tex` | `\ExFMeight*` (LimiX-16M on T1′; plan `fm_extra/limix_t1p/PLAN.md`) | `uv run python paper/p1/exploratory/fm_extra/summarize_t1p_fm8.py` | `fm_extra/out/t1p_fm8/*_fm8.parquet`, `fm_extra/out/t1p/*_fm[67].parquet`, `fm_extra/out/s7a/` (included without per-row columns), `results/study12` | no |
| `exploratory/fm_extra/out/dose_c_numbers.tex` | `\ExDoseX*` (Study 7 Part (c) for fm6–fm8, DEV; plan `fm_extra/dose_plan.md`) | `uv run python paper/p1/exploratory/fm_extra/analyze_dose_c.py` | the fm6–fm8 Part (c) fits `fm_extra/out/dose_c/` (not included); derived `out/dose_c_{chain,verdicts,raw}.csv` included | yes |
| `exploratory/review_reanalysis/out/rv_numbers.tex` | `\ExRv*` (re-analyses requested in review) | `uv run python paper/p1/exploratory/review_reanalysis/make_rv_numbers.py`; it concatenates the parts below | `review_reanalysis/out/rv_*.tex` | see parts |
| `exploratory/human_check/out/hc_numbers.tex` | `\ExHc*` (human check of the audit sample) | `cd paper/p1/exploratory/human_check && uv run python hc_analysis.py` (adjudication in `ADJUDICATION.md`) | `human_check/data/{human_check_form,final_coding,adjudication,human_check_sample}.csv` | no |
| `exploratory/prop5/out/prop5_numbers.tex` | Prop. 5(iii) constants | `cd paper/p1/exploratory/prop5 && uv run python make_prop5_numbers.py` | none (theory; `bounds.py`; independent check `verify.py`) | no |
| `exploratory/realtabpfn_check/out/realtabpfn_numbers.tex` | `\ExRtp*` (Real-TabPFN overlap sensitivity, fm1) | `uv run python paper/p1/exploratory/realtabpfn_check/realtabpfn_check.py` | `results/study7/a`, `results/study12`, `fm_extra/out/*_cells.csv`; the HELD-OUT id list (withheld: `overlap.csv` is stored without its HELD-OUT rows, none of which overlaps, so the script does not rerun here) | no |
| `exploratory/tabdpt_check/tabdpt_numbers.tex` | `\Tdpt*` (TabDPT pre-training corpus counts) | none (documentary; `tabdpt_check/NOTE.md`) | the two cited papers and public OpenML metadata | no |
| `exploratory/seed_indep/seed_indep_numbers.tex` | `\ExSeedInd*` (independent learner seed; plan `seed_indep/PLAN.md`) | fits: `run_seed_indep.py`; numbers: `uv run python paper/p1/exploratory/seed_indep/analyze_seed_indep.py` | `seed_indep/out/dev/*.parquet` (included); the HELD-OUT row is an aggregate (fits not included) | no |
| `exploratory/sevenfm/out/sevenfm_b1.tex` … `sevenfm_b11.tex` | `\ExSevFm*` (B1–B11 with seven FMs) | `uv run python paper/p1/exploratory/sevenfm/bN_*.py 7` (`harness.py`) | the original analyses' inputs plus the fm6–fm8 fits; their outputs `sevenfm/out/bN/seven/` are included, including B2's scored-draw cache `b2/seven/draws_scored.parquet` | B2, B3, B5, B6, B10: yes |
| `heldout_numbers.tex`, `heldout_hand.tex`, `exploratory/heldout/out/heldout_{ci,m3plus}_numbers.tex` | `\HO*`, `\ExHO*` (HELD-OUT, joint opening) | `simaudit.analysis.joint_heldout`; `exploratory/heldout/{ci_macros,m3plus,s3_bstar}.py` | the HELD-OUT results (not included; the HELD-OUT ids stay withheld); the macro files and the aggregate `heldout/out/*.csv` are included | no |
| `study15_numbers.tex` | `\SF*` (Study 15, confirmatory family and pre-declared secondary values) | `uv run python -m simaudit.analysis.study15` (also writes `results/study15/study15_results.json`) | `results/study15/second_frame/*.parquet` (metric rows), `integrity_screen.csv`, `data/pool_second_frame.yaml`; the descriptive ratio to HELD-OUT also reads the joint HELD-OUT results (not included; without them the ratio is empty) | no |
| `study15_secondary_numbers.tex` | `\SF*` (further pre-declared secondary values, unadjusted) | `uv run python paper/p1/make_study15_secondary.py` | `results/study15/study15_results.json` | no |
| `exploratory/sf_autopsy/sf_autopsy_numbers.tex`, `sf_sign2x2_numbers.tex` | `\ExSFA*` (Study 15 autopsy, exploratory, post hoc) | `uv run python paper/p1/exploratory/sf_autopsy/sf_autopsy.py`, `sf_sign2x2.py` | `results/study7/a`, `results/study12`, `results/study15/second_frame`, the e10 pool tables, `prereg/STUDY15_SECOND_FRAME.md`; HELD-OUT enters only in aggregate (its results are not included, so the scripts do not rerun here) | no |
| `exploratory/kumo_unused/out/kumo_unused_numbers.tex` | `\ExKumoUnused*` (Kumo, declared-but-unused classes) | fits: `run_kumo_unused.py`; numbers: `uv run python paper/p1/exploratory/kumo_unused/summarize.py` | `kumo_unused/out/runs/*.parquet` (included whole, 2 MB); the fits read `results/study7/a/*_fm3.parquet` | fits: yes; numbers: no |

### Parts of `rv_numbers.tex`

All parts are in `exploratory/review_reanalysis/`. Run the scripts in the order of the table, then run `make_rv_numbers.py`. Every macro is exploratory unless its comment names a pre-registered provenance.

| Part file (`out/`) | Script | Inputs | Per-fit probabilities |
|---|---|---|---|
| `rv_a.tex` | `a_review_subtypes.py` | the final audit coding `human_check/data/final_coding.csv` (default input) and `a_subtype_lookup.csv` | no |
| `rv_bc.tex` | `bc_contrasts.py` | `results/study7/a`, `results/study12`, the fetched data (labels) | no |
| `rv_s9.tex` | `s9_sequential_e10.py` | `results/study9/dev`, `exploratory/verify/out/claim_k_per_table.csv`, the fetched data (labels) | no |
| `rv_ten.tex` | `holm_ten.py` | `out/bc_contrasts.csv`, `fm_extra/out/{t1p_cells,t1p_fm8_cells}.csv` | no |
| `rv_mech.tex` | `mech_recompute.py` | `results/study7/a`, `results/study12` (probabilities) | **yes**; output `out/mech_tables.csv` and `out/mech_summary.csv` (included; Fig. `fig:mech` reads `mech_tables.csv`) |
| `rv_accdose.tex` | `acc_imb_dose.py` | `results/study7/c`, `out/bc_t1_pool_tv.csv`, `exploratory/out/e12_slopes.csv` | no |
| `rv_r4.tex` | `r4_checks.py` | `results/study7/c`, `out/bc_t1_pool_tv.csv`, `exploratory/out/e10_pools_dev.csv` | no |
| `rv_r5.tex` | `r5_checks.py` | `out/r4_rise.csv`, `out/bc_t1_pool_tv.csv`, `e17_metrics/e17_reproduction.csv` | no |
| `rv_r6.tex` | `r6_checks.py` | `results/study7/a` (labelled rows), `out/bc_t1_pool_tv.csv`, the fetched data (labels) | no |
| `rv_r6b.tex` | `r6b_checks.py` | `results/study12`, `human_check/data/*.csv` | **yes** (its loader expects the per-row columns `proba` and `query`; output included) |
| `rv_r7fs.tex` | `r7_fewshot_def.py` | the full texts of the audited papers (copyrighted; not included) | no; output `out/r7_fewshot_def.csv` included |
| `rv_r7.tex` | `r7_checks.py` | `out/r6_accbin.csv`, `out/bc_t1_pool_tv.csv`, `results/study7/a` | no |
| `rv_r8.tex` | `r8_checks.py` | `results/study12` (per-fit query rows, column `query`) and the previous `rv_numbers.tex` | **yes** (per-row column dropped) |
| `rv_r10.tex` | `r10_checks.py` | `out/mech_summary.csv` | no |
| `rv_r11.tex` | `r11_partc_common.py` | `results/study7/c` (probabilities), `data/cache/npz` | **yes**; outputs `out/r11_partc.csv` and the per-fit `out/r11_partc_common.parquet` (included) |
| `rv_r12.tex` | `r12_fisher_vs.py` | none (pure computation: Fisher information of the saturated case-control likelihood) | no |
| `rv_r13.tex` | `r13_mm81.py` | none (pure computation: Manski–McFadden 1981, Eq. 1.11) | no |
| `rv_r14.tex` | `r14_mech_chain.py` | `out/r11_partc_common.parquet`, `out/r11_partc.csv` (run `r11_partc_common.py` first, or use the included parquet) | no; outputs `out/r14_mech_chain.csv`, `out/r14_verdicts.csv` |

### Figures

All figures are written by `cd paper/p1 && uv run python figures/make_figures.py`. Each figure is a vector PDF with a PNG preview.

| Figure (label) | File (in `paper/p1/`) | Inputs |
|---|---|---|
| `fig:protocols` | `figures/fig1_protocols.pdf` | none (illustration with a fixed seed) |
| `fig:gap` | `figures/fig2_gap.pdf` | `results/study7/{a,b,c}`, `results/study12`, `exploratory/fm_extra/out/{s7a,s7a_v2,t1p,t1p_fm8}/*.parquet` |
| `fig:ranks` | `figures/fig3_ranks.pdf` | `results/study7/a`, `results/study12` |
| `fig:mech` | `figures/fig4_mechanism.pdf` | `results/study7/{a,b,c}`, `exploratory/out/e10_pools_dev.csv`, `exploratory/review_reanalysis/out/mech_tables.csv` |
| `fig:forest` | `figures/fig_forest.pdf` | `results/study7/a`, `results/study12` (also `analysis/t1p_heterogeneity.py`, `exploratory/e9_meta_analysis.py`) |
| `fig:accimb` | `exploratory/out/e6_accuracy_imbalance.pdf` | `results/study7/a`, `exploratory/out/e10_pools_dev.csv`, `exploratory/out/e1_gaps.csv`, `exploratory/e1_t1p/out/e1t1p_gaps.csv` |
| `fig:heldout` | `figures/fig8_heldout.pdf` | `results/study13/heldout` (HELD-OUT; not included: `make_figures.py` skips this figure when those results are absent, and the included file is the one built from them) |

### Tables

Every table in the paper is typeset from the macro files above:
- `tab:protocols`, `tab:arms`: `numbers_p1.tex` and `exploratory_numbers.tex`;
- `tab:hyp` (hypotheses), `tab:slopes`, `tab:claims`, `tab:seqt1`: `numbers_p1.tex`, `exploratory_numbers.tex`, `rv_numbers.tex`, `claims_audit_numbers.tex`;
- `tab:distpfn`, `tab:distpfntp`: the two DistPFN macro files;
- `tab:e17`: `e17_numbers.tex`;
- `tab:seedindep`: `seed_indep_numbers.tex`;
- `tab:secondframe`: `study15_numbers.tex` and `study15_secondary_numbers.tex`;
- `tab:prisma`, `tab:auditagree`, `tab:auditsum`, `tab:auditmetric`, `tab:auditex` (audit): `\SysAudit*` from `supplementary/audit/` and `hc_numbers.tex`;
- the per-dataset, metric-convention and redraw tables: `tables_p1.tex`, `tables_metrics_p1.tex` and `tables_redraw_p1.tex`.

### Notes on specific results

- **Study 12, secondary analysis S-2.** The "seen-class AUC" reported for S-2 is the stored column `auc`, which is the AUC over the classes seen in the labelled set. The stored column `seen_auc` is not used. See `src/simaudit/analysis/study12.py`, function `auc_all_analysis`.
- **TabPFN v3.5 (fm5).** Its weights are gated by their provider. To refit fm5, accept the provider's licence and set your own `TABPFN_TOKEN`. This repository includes no weights of any model and no token; `environment/models.json` gives the checkpoint hashes.
- **Exploratory FMs.** Each runs in its own environment; pins, revisions and hashes are in `exploratory/fm_extra/out/pins.json`:
  - TabDPT v1.3 (fm6): `tabdpt` 1.3.1, checkpoint `tabdpt1_3.safetensors`, Apache-2.0;
  - Mitra-v2 (fm7): `mitra-classifier-2` via `autogluon.tabular` 1.6.3, `fine_tune=False`, `n_estimators=1`, float32, seeded per fit (the `s7a_v2` run), Apache-2.0;
  - LimiX-16M (fm8): `LimiX-1_16M.ckpt`, `limix` 2.0.0 at the pinned code commit, config `cls_default_noretrieval.json`, `mix_precision=False`, under the provider's licence.
  - TabICLv2 (fm2) is the default checkpoint of `tabicl` 2.2.0.
- **Study 15 data.** The 20 second-frame tables are public on Zenodo, figshare and Kaggle; `data/pool_second_frame.yaml` gives each source URL, licence, declared target and the SHA-256 of every raw file and of the encoded table. `fetch_data.py` does not download them (the Kaggle file needs an account). `results/study15/second_frame/frame_build.json` records the encoding of each table and `integrity_screen.csv` the integrity flags.

## Results that need the per-fit probabilities

Except for the four UCI 257 files below, the result files in `results/` and the exploratory FM fits are stored without their per-row columns `proba`, `query` and `query_pos`. With those columns, all the files take about 1.9 GB as stored parquet:

| Files | Size |
|---|---|
| `results/study7` | about 250 MB |
| `results/study9/t1prime` | about 580 MB |
| `results/study12` | about 500 MB |
| `results/study15` | about 270 MB |
| fm_extra fits | about 280 MB |

These files are not public; they are available from the author on request. The results below are re-scored from these columns. For each one, the repository includes the derived file that holds every number the paper reports; the script itself does not rerun from this repository:
- the prior corrections: `reports/p1_corrections.py` / `p1_corrections_runs.py` → `results/p1_corrections/`, `reports/P1_PROTOCOL_CORRECTIONS_EXPLORATORY.md`; DistPFN on DEV and T1′ → `exploratory/distpfn/out/runs.parquet`, `exploratory/distpfn/t1p/out/runs.parquet`; Correction 4 on T1′ → `exploratory/c4_t1p/out/ps.parquet`. The three caches are included, so their macro files rebuild without the probabilities.
- the mechanism coverage split (abstract, Figure 4a): `review_reanalysis/mech_recompute.py` → `out/mech_tables.csv`, `out/mech_summary.csv`, `out/rv_mech.tex`;
- E1, E2, E7, E8, E10, E12 and E15: `exploratory/e1_classical.py`, `e2_auc.py`, `e7_common_eval.py`, `e8_schema_aware.py`, `e10_split_and_pools.py`, `e12_small_checks.py`, `e15_budget_metrics.py` → `exploratory/out/*.csv` and `*.parquet` (E15's per-run cache `e15_runs.parquet` is not included; `make_exploratory_numbers.py` reads `out/e15_*.csv`);
- E16: `e16_multiclass/e16_extract.py` → `e16_multiclass/out/classes.parquet`, `out/e16_*.csv`;
- E17: `e17_metrics/e17_metrics.py` (first build) → `e17_metrics/e17_draws.parquet` and `*.csv`;
- Kumo unused classes: `kumo_unused/run_kumo_unused.py` → `kumo_unused/out/runs/*.parquet` (included whole);
- UCI 257: `uci257/uci257_sensitivity.py`. The four probability files it reads (`results/study9/t1prime/pmlb_1000000257_*.parquet`) are included whole. It also needs the UCI API snapshot, which is not included.
- the Study 7 balance check (`balance_check` in `reports/study7_report.py`) → `reports/STUDY7_REPORT.md`; the other report scripts that read the probabilities are `reports/theory_checks.py` and `study12_uci257_exploratory.py` → `THEORY_CHECKS.md`, `STUDY12_UCI257_EXPLORATORY.md` (all included; `make_numbers.py` parses them);
- review part M11-1: `review_reanalysis/r11_partc_common.py` → `out/r11_partc.csv`, `out/r11_partc_common.parquet`, `out/rv_r11.tex`;
- the extra FMs: Part (c) dose (`fm_extra/analyze_dose_c.py` → `out/dose_c_*.csv`) and the seven-FM items B2, B3, B5, B6, B10 (`sevenfm/bN_*.py` → `sevenfm/out/bN/seven/`);
- review parts R8 (query rows) and R6b (its loader expects the per-row columns): `review_reanalysis/r8_checks.py`, `r6b_checks.py` → `out/rv_r8.tex`, `out/rv_r6b.tex`.

## Commands

### 0. Environment and public data

```bash
cd simaudit
uv sync --frozen                     # Python 3.12, versions pinned in uv.lock
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
uv run python ../fetch_data.py --case-study
```

`fetch_data.py` downloads three sets of public data:
- the 20 DEV tables (OpenML, by id);
- the 52 T1′ tables (PMLB at the pinned commit; the official UCI API);
- the five OpenML ARFF files of the case study.

It checks each file against the SHA-256 pinned in `datasets/dev.csv`, `datasets/t1prime.csv` and the script itself, and it stops on any mismatch. The cache goes to `data/cache/` (about 21 MB). Keep that location: the case study reads `data/cache/npz/841.npz` by path. Scripts that only read `results/` or `out/` need no data and no network.

### 1. Numbers, tables and figures from the stored results (no model fits)

The commands in the macro-file map regenerate each file. The committed `.tex`, `.csv`, figure and `out/` files are the reference outputs to compare with (`git diff` shows any difference). We checked, from a fresh copy with the network disabled, that `make_numbers.py`, `make_tables.py`, `make_exploratory_numbers.py`, `hc_analysis.py`, `make_prop5_numbers.py`, `kumo_unused/summarize.py`, `fm_extra/summarize_t1p_fm8.py`, `make_study15_secondary.py`, the review parts that need no per-fit probabilities and `make_rv_numbers.py`, and `figures/make_figures.py` reproduce their included macro files byte for byte and their figures pixel for pixel (the PDF metadata differs). Three review parts (`bc_contrasts.py`, `s9_sequential_e10.py`, `r6_checks.py`) read labels, so they need `fetch_data.py` first.

### 2. Re-running the experiments (CPU; hours to days)

```bash
cd simaudit
uv run simaudit build-pool-pmlb                      # T1' tables (PMLB commit in datasets/sources.json)
uv run simaudit snapshot-uci && uv run simaudit build-pool-uci
uv run simaudit study7 --workers 3                   # Study 7 (DEV)
uv run simaudit study7 --datasets 694 --drop 694:1 --out-name study7_s4
uv run simaudit study9 --set t1prime --workers 2     # Study 9 (T1')
uv run simaudit study9 --set dev
uv run python -m simaudit.runner.study12 --workers 7 # Study 12 (T1')
uv run python -m simaudit.runner.study15 --set second_frame --workers 6   # Study 15 (encoded tables in data/cache/second_frame/npz)
uv run python reports/p1_corrections_runs.py         # protocol corrections (exploratory)
# exploratory runs: see the "Usage" line at the top of each script, e.g.
uv run python paper/p1/exploratory/b_dose/run_dose_t1p.py --workers 6
uv run python paper/p1/exploratory/e1_t1p/run_e1_t1p.py
uv run python paper/p1/exploratory/kumo_unused/run_kumo_unused.py
```

These runners regenerate the full result files, including the per-row probabilities. Labelled sets are cached per table under `results/<study>/selections/` and reused.

The LUCoS selector needs the third-party TabClustPFN code and checkpoint (research-only licence; not included). The cached selections let the FM fits run without them.

The TabEBM case study (`exploratory/case_study/`) runs in two stages, both from `simaudit/` with `code/run_all.sh`:
- stage A runs in a separate environment: TabEBM at the pinned commit, tabpfn 2.1.2 and imbalanced-learn 0.12.4 (versions and hashes in `PREREG.md`);
- stage B runs in the package environment.

Study 13 (HELD-OUT) was run once, at the joint opening on 2026-10-30. Its pre-registration, the opening note and its macro files are included; its data, ids and per-fit results are not. Study 15 was run once after its freeze on 2026-10-31; its encoded tables are built by `simaudit.data.second_frame` from the raw files listed in `data/pool_second_frame.yaml`, a build that also checks content hashes against HELD-OUT (withheld), so the included `frame_build.json` records its result.

## What is not included, and why

- **HELD-OUT.** Its tables, its ids (the HELD-OUT list of `data/pool_i2.yaml` is removed, as are the HELD-OUT rows of `realtabpfn_check/out/overlap.csv` and the per-table file `heldout/out/s3_bstar_cells.csv`), its per-fit results and any access log. Only aggregates over the HELD-OUT tables are included.
- **The pilot's exclusion list.** The pilot pre-registration (`prereg/PILOT_I2.md`) excluded 124 OpenML ids from the DEV frame because an earlier study had used them. That study is called `PRIORSEL` here, and the list is not included. The DEV ids are given in `data/pool_i2.yaml` and `datasets/dev.csv`, so no analysis needs the list.
- **Raw third-party data.** The OpenML, PMLB and UCI files, the case-study ARFF files, the second-frame raw files and the UCI API snapshot are not included, because their licences vary. Run `fetch_data.py` to download the first three.
- **Model weights of any kind and access tokens.** This includes the TabPFN v3.5, TabDPT, Mitra and LimiX checkpoints and the TabClustPFN checkpoint. The hashes are in `environment/models.json` and `pins.json`.
- **Third-party code and environments.** Not included: the TabEBM repository, the tabpfn 2.1.2 and imbalanced-learn wheels, the TabClustPFN/LUCoS code and every virtual environment. Packages such as `kmedoids` are installed by `uv sync`, not included.
- **Per-row arrays.** See "Results that need the per-fit probabilities"; also the case study's `out/arrays/`.
- **Logs, process markers and caches.** Run logs, PID and DONE markers, `__pycache__` and E15's per-run cache `exploratory/out/e15_runs.parquet` are left out.
- **Internal working material.** The internal research notes, working summaries and report prose, plans that the paper does not cite, the review's `extract_numbers.py` (it reads internal files; its output `audit_numbers.json` is included), and code and checks that no number of the paper uses (a timing benchmark of companion work, label-budget theory notes, platform checks). Also the material of later studies that are not part of this paper.
- **The paper source.** See the paper itself (citation above).
- **Development history.** This repository is a snapshot of the research code base, published as a single commit. Commit hashes are replaced by the identifiers of `REVISIONS.md`, and absolute paths by repository-relative ones or `<HOME>`/`<workspace>`. The mapping from revision identifiers to commits of the development history is not public, because that history references the sealed HELD-OUT set; it is available from the author on request.

Some files differ from their originals in the development history, so a hash that a pre-registration or a provenance file cites may not match the copy here:
- neutral revision identifiers, paths and names;
- the HELD-OUT ids (and the HELD-OUT rows of one overlap table) and the name of the earlier study;
- Portuguese labels and internal note names, translated into English;
- in the kept documents, lines about internal process (who approved a step, which working session relayed a decision, which machine ran a job, to-do files, suggested wording for the paper) are replaced by neutral wording or a short "removed" marker. Dates and "before any run or access" statements are kept.

No design value, number, threshold, hypothesis or decision rule was changed. The list of altered files, with their original SHA-256, is available from the author on request.

## Licence

- **Code** (`*.py`, `*.sh`, `Makefile`s, `pyproject.toml`, `uv.lock`, the CI workflow): MIT licence, see [`LICENSE`](LICENSE).
- **Results, documents and figures** (everything else: stored results and derived outputs under `results/` and `out/`, macro files `*.tex`, CSV/JSON/YAML files, pre-registrations, reports and other Markdown documents, figures): Creative Commons Attribution 4.0 International (CC BY 4.0), see [`LICENSE-DATA`](LICENSE-DATA).

The public tables that `fetch_data.py` downloads, the FM packages and their weights, and the papers listed in the audit materials keep their own licences and terms; `datasets/t1prime.csv` and `data/pool_second_frame.yaml` give the licence of each table. TabPFN v3.5 weights are gated by their provider and are not included.

## Contact

Felipe Provezano Coutinho (Independent Researcher). Contact: open an issue, or see ORCID 0000-0001-7206-5856. Requests for the per-fit probability files, the revision-to-commit mapping or the list of altered files can be made the same way.
