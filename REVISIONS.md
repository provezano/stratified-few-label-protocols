# Revision identifiers

Git commit hashes are replaced by neutral identifiers. R001 to R018 are the revisions that the paper
cites (appendix "Pre-registration log"); they are listed with the commit date, the file each one
freezes or changes, and a one-line description. R014 is not cited: it is an alias of R012, whose commit
holds both Study 13 Amendment 1 and Study 9 Amendment 2; the files of this repository use R012. Identifiers from R019 on replace
other commit hashes quoted inside the files of this repository, with the commit date and message. The mapping
to the commits of the development history is not public (that history references the sealed HELD-OUT set); it is available from the author on request.

## Revisions cited in the paper

| ID | Date | File | Description |
|---|---|---|---|
| R001 | 2026-10-28 | `paper/p1/exploratory/b_dose/PREREG_T1P_SUBSET.md` | Pre-registration of the dose arms F/S/N on a 16-table T1' subset, committed before any T1' fit (exploratory) |
| R002 | 2026-10-28 | `paper/p1/exploratory/case_study/PREREG.md` | Pre-registration of the TabEBM case study (realistic vs stratified labelled sets), frozen before any download or fit |
| R003 | 2026-10-21 | `prereg/STUDY9_HOW_MANY_LABELS.md` | Code state whose analysis-code SHA-256 hashes Study 9 Freeze note 1 records (frozen analysis code) |
| R004 | 2026-10-27 | `tools/heldout_dedup_t1prime.py` | Overlap checks between T1' and HELD-OUT (hash, rows, signatures), committed before they ran |
| R005 | 2026-10-27 | `prereg/STUDY8_EXPANSION.md` | Result of the T1'/HELD-OUT overlap checks (passed, no exclusion), recorded before any T1' analysis |
| R006 | 2026-10-27 | `reports/study9_t1prime_report.py` | Study 9 T1' report driver (frozen analysis unchanged plus pre-specified extras), committed before the analysis ran |
| R007 | 2026-10-27 | `prereg/STUDY12_PROTOCOL_T1PRIME.md` | Study 12 pre-registration (Study 7 Part (a) on T1'), frozen and committed alone before any run |
| R008 | 2026-10-27 | `prereg/STUDY12_PROTOCOL_T1PRIME.md` | Study 12 Addendum 1: integrity sensitivities SENS-INT and SENS-ID, before any outcome |
| R009 | 2026-10-27 | `prereg/STUDY12_PROTOCOL_T1PRIME.md` | Study 12 Amendment 2: redraw repetitions without a covering draw recorded as infeasible (runner/study12.py) |
| R010 | 2026-10-28 | `prereg/STUDY12_PROTOCOL_T1PRIME.md` | Study 12 Erratum 1: T1' composition is 35 binary / 17 multiclass tables, corrected before any metric was read |
| R011 | 2026-10-27 | `prereg/STUDY13_HELDOUT_CONFIRMATION.md` | Study 13 pre-registration (HELD-OUT replication of Study 7 Part (a)), frozen before any HELD-OUT access |
| R012 | 2026-10-28 | `prereg/STUDY13_HELDOUT_CONFIRMATION.md` | Study 13 Amendment 1 (label normalisation, TabDPT and Mitra as secondary learners, E17 metric families), before HELD-OUT access |
| R013 | 2026-10-27 | `reports/P1_PROTOCOL_CORRECTIONS_EXPLORATORY.md` | Plan, estimands and criteria of the protocol corrections, committed before any correction was computed |
| R014 | 2026-10-28 | `prereg/STUDY9_HOW_MANY_LABELS.md` | Study 9 Amendment 2 (label normalisation on HELD-OUT), before HELD-OUT access; committed together with R012, which the paper cites for it (alias of R012, same commit; not cited in the paper) |
| R015 | 2026-10-28 | `prereg/STUDY9_HOW_MANY_LABELS.md`; `prereg/STUDY13_HELDOUT_CONFIRMATION.md` | Study 9 Amendment 3 and Study 13 Amendment 2 (unverified labels on HELD-OUT), before HELD-OUT access |
| R016 | 2026-10-28 | `prereg/STUDY9_HOW_MANY_LABELS.md` | Study 9 Amendment 4 (integrity screen on HELD-OUT, flag-only; SENS-INT/SENS-ID), before HELD-OUT access |
| R017 | 2026-10-29 | `prereg/STUDY13_HELDOUT_CONFIRMATION.md` | Study 13 Amendment 3 (LimiX-16M as third secondary learner; S-10 and S-11 Holm over 3; code in a separate earlier commit), before HELD-OUT opening |
| R018 | 2026-10-30 | `prereg/JOINT_OPENING_2026-10-30.md` | Joint HELD-OUT opening note (Studies 9, 13, 10; cross-study Holm family of 19 tests; launch settings), committed before any HELD-OUT access |

## Other revisions quoted in the repository files

| ID | Date | Commit message |
|---|---|---|
| R019 | 2026-10-28 | docs(paper): P1 exploratory dose-response arms F/S/N |
| R020 | 2026-10-28 | docs(p1-exploratory): plan for Correction 4 (reweighting/GREG, coverage-only) on T1' Study 12 runs, before the run (audit row 22) |
| R021 | 2026-10-28 | docs(p1-exploratory): DistPFN plan on stored Study 7 DEV contexts (post hoc, before any run) |
| R022 | 2026-10-28 | docs(p1-exploratory): plan for post-hoc prior corrections (DistPFN, EM, p/pi) on T1' Study 12 probabilities, before the run |
| R023 | 2026-10-29 | plan(p1-exploratory): Kumo declared-but-unused class test on DEV (claims-audit row 26), written before any DEV fit |
| R024 | 2026-10-30 | docs(p1): TabDPT v1.3 (fm6) pretraining-contamination note: B.1 overlap and v1.2/v1.3 corpus eligibility |
| R025 | 2026-10-30 | docs(p1): heldout_branches synced to blind review 4 - new finding (2) wording with the (ii-c) verdict word, flip over six learners, tab:main column for HELD-OUT, conclusion without emph lead-ins, literal mechanism verdicts unaffected by HELD-OUT |
| R026 | 2026-10-30 | docs(p1): M7 reproducibility paragraph: Study 7 (a) probabilities shipped, rest at camera-ready; REVISIONS.md content; TabPFN v3.5 access |
| R027 | 2026-10-30 | docs(prereg): freeze Study 10 (decision of 2026-10-29; delta, Sec. 8 table and budget confirmed; G4 passed with exceptions E-T7-1/E-T7-2) |
| R028 | 2026-10-27 | feat(p1-exploratory): fm_extra feasibility (TabDPT v1.3, Mitra-v2, LimiX-16M adapters, probes, smoke) |
| R029 | 2026-10-29 | plan(p1-exploratory): LimiX-16M on T1' with Study 12 draws (design identical to fm6/fm7), after timing pilot, before the run |
| R030 | 2026-10-29 | results(p1-exploratory): LimiX-16M on T1' per plan R029 (52 tables, Study 12 draws): gap +0.067 [+0.056,+0.079] mean over m, positive at every m; ExLimixTp macros |
| R031 | 2026-10-29 | results(p1-exploratory): LimiX-16M T1' accuracy gap (same method as macro-F1; added later, not in plan R029) + LimiX DEV accuracy gap; ExLimixTpAcc*/ExLimixDevAcc* macros |
| R032 | 2026-10-29 | docs(p1-exploratory): value gate and fixed analysis for Study 7 Part (c) on TabDPT, Mitra and LimiX (DEV, before any fit) |
| R033 | 2026-10-29 | results(p1-exploratory): Part (c) fm6-fm8 optimization check (identical outputs) and timing pilot; no analysis |
| R034 | 2026-10-29 | results(p1-exploratory): Study 7 Part (c) for TabDPT, Mitra, LimiX on DEV: decline and corrected residual in all 6 cells; 0 beyond, 2 inconclusive, 1 not beyond (outcome O2) |
| R035 | 2026-10-29 | feat(study13): fm8 LimiX-16M as third secondary FM in the worker and runner (Amendment 3 code) |
| R036 | 2026-10-31 | feat(study15): second-frame loader, runner and analysis for the post-FM replication of Study 13 test (i) |
| R037 | 2026-10-21 | docs(prereg): Study 8 Amendment 1, Tier 2 (PMLB + OpenML remainder) and Tier 3, before any download |
| R038 | 2026-10-21 | docs(prereg): Study 8 Amendment 2, UCI Tier 4, OpenML-free set T1', HELD-OUT checks and integrity screens |
| R039 | 2026-10-21 | feat(pool): UCI Tier 4 loader, integrity screens and T1' assembly (Study 8 Amendment 2) |
| R040 | 2026-10-21 | fix(pool): one-directional, rounding- and recoding-robust row overlap; pass integrity CSV to build-t1prime |
| R041 | 2026-10-21 | docs(study9): DEV dry-run report and freeze of the analysis code |
| R042 | 2026-10-28 | feat(p1-exploratory): UCI 257 label-spelling sensitivity (Very Low/very_low) on stored Study 9 fits |
| R043 | 2026-10-28 | docs(report): Study 12, Study 7 replication on T1' (52 tables, 4 FMs) |
| R044 | 2026-10-31 | docs(p1): fix meaning changes found by readability verifier 2 |
