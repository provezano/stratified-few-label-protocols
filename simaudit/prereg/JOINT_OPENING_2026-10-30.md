# Joint HELD-OUT opening note (2026-10-30)

This note is required by Study 13 Sec. 0.3 (item 3) and Sec. 4.4. It is committed before any runner touches `sealed_heldout/`. It follows the decision of 2026-10-29 to freeze Study 10 and open HELD-OUT. No HELD-OUT file has been read by any study before this commit.

## 1. Studies in the event

| Study | Prereg | State at this commit | SHA-256 of the prereg |
|---|---|---|---|
| Study 9, Sec. 9 | `prereg/STUDY9_HOW_MANY_LABELS.md` | FROZEN, Amendments 1–4 | `ed97591a70dd61f5082d92f19d5f492ec89c6743349612c51418f1164df019b0` |
| Study 13 | `prereg/STUDY13_HELDOUT_CONFIRMATION.md` | FROZEN 2026-10-28, Amendments 1–3 | `949ce46c7973df757c2f61d161162eef166646b109e200bbae9139afb0d4e375` |
| Study 10 | `prereg/STUDY10_SELECTOR_CONFIRMATION.md` | FROZEN 2026-10-30 (commit `R029`), before the deadline of 2026-10-31 23:59 UTC | `87ea00fc021dafd481b8acc960c6b785cb798fa363f8d4cff55bbd2acb942544` |

The hashes above are of the files as committed before this note. Each prereg's log gets a one-line pointer to this note in the same commit, so those three files change only by that appended line.

## 2. Confirmatory cells (Study 13 Sec. 0.3, item 4)

No outcome of any study is analysed before every cell below is complete or has definitively failed.

- **Study 9:** every HELD-OUT job of the sealed driver `runner/study9_heldout` (all four FMs, as its Sec. 9 requires).
- **Study 13:** every job of `runner/study13 --set heldout`, all groups: the four pre-registered FMs and the classical learners in arms T and F, and the secondary FMs fm6, fm7 and fm8 (Amendments 1 and 3; S-10 and S-11).
- **Study 10:** the H1 cells only, that is fm2 on HO ∪ T1′, every arm × mult × rep (Study 10 Sec. 12). The secondary learners of Study 10 run afterwards and are analysed after the joint analysis.

While runners are going, the only things checked are process state, job counts, failure counts and logs. No metric is computed or read.

## 3. Cross-study Holm family (Study 13 Sec. 4.4), 19 p-values at α = 0.05

1. **Study 13, 6 tests** (Sec. 3): (i) for fm1, fm2, fm3 and fm5; (ii-a); (ii-c) TOST.
2. **Study 9, 9 tests** (its Sec. 6 family, computed on HELD-OUT with FM2 by the pinned `analysis/study9.py`): H1b; H2a × 3 (S-KMn, S-LUn, S-ST); H2b × 2; H3b; H4 × 2 (A, D). A test that the frozen analysis cannot compute on HELD-OUT enters the family with p = 1.
3. **Study 10, 4 tests** (its Sec. 7.1): H1a–H1d, the non-inferiority tests with δ = 0.01.

Each study's within-study verdicts follow its own prereg. The wording rule of Study 13 Sec. 4.4 applies: "confirmed on HELD-OUT" only under the cross-study Holm. Otherwise the claim is "replicated (study-wise), not under the cross-study correction". Both adjusted p-values are reported (`analysis.study13.cross_study_holm`).

## 4. Launch

- Every runner starts with `P1_HELDOUT_GO=1`, `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=MKL_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1` and `nice -n 15`. The thread settings do not change outputs (paper/p1/exploratory/veclib_check/NOTE.md).
- Workers, at most 10 in total: Study 9, 4; Study 13, 3; Study 10, 3 (fm2 on HO, then fm2 on T1′).
- Each runner checks its guards, the manifest and the pool hashes itself, and writes its own access log. A crashed runner may be resumed from its checkpoints with unchanged code and settings, but never re-run with changes.
