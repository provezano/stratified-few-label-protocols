# LimiX-16M on T1′ — EXPLORATORY plan

Written and committed 2026-10-29 after a timing pilot (below) and before the analysed run.
EXPLORATORY, not pre-registered. T1′ was already exposed by Study 12 and by the fm6/fm7
T1′ runs; HELD-OUT, `sealed_heldout/` and the Study 14 T2 tables are not touched;
`main.tex` is not edited (macros only).

## What and why
The paper ran LimiX-16M (`stable-ai/LimiX-1_16M` @ 3a7c18ac, Stable AI licence v1.0,
checked by the authors for internal research benchmarking; LimiX-2 is not used) only on DEV
("LimiX-16M only on DEV", "LimiX was not run on T1′"; Fig. 2 caption). TabDPT v1.3 (fm6)
and Mitra-v2 (fm7) were also run on T1′ with Study 12's draws. This run closes that
asymmetry: LimiX-16M on the same 52 T1′ tables with the identical design.

## Design (identical to fm6/fm7 on T1′, `run_t1p.py`)
- Tables: the 52 of `data/pool_t1prime.yaml`.
- Row permutation of `runner.study12` (namespace 33); stored Study 12 selections and seeds
  (`results/study12/selections/{key}.parquet`, read-only), **stratified (5 reps) and random
  (20 reps)** at m ∈ {1, 3, 5, 10}: 100 fits per table, 5,200 in all. No redraw arm.
- Panel E = first min(n, 2000) permuted rows; metrics on E ∖ L via
  `analysis.study12.panel_metrics` (+ `pc_*` prior-corrected); missing classes as in
  `runner.study9.predict_on` (fit on seen classes, unseen = 0; one-class context = constant).
- Learner: `adapters.limix_predict`, the DEV wrapper unchanged (default no-retrieval
  classification config, fp32, CPU, one torch thread, seed = cell seed). The only change is
  loading the checkpoint with `torch.load(..., weights_only=True)` (file = plain
  `{state_dict, config}` dict; already-downloaded weights, `local_files_only=True`,
  `HF_HUB_OFFLINE=1`). CPU as on DEV (the DEV run did not use MPS).
- Runner `run_t1p_fm8.py` (imports `run_t1p.py` unchanged); outputs `out/t1p_fm8/`
  (not committed). Pilot outputs (`out/t1p_fm8_pilot/`) are **not** used in the analysis;
  all 52 tables are re-run.

## Timing pilot (before this plan)
3 tables, 3 workers, same runner, output `out/t1p_fm8_pilot/` (stopped cleanly via STOP once
the timing was clear; 131 fits):
- pmlb_891171204 (435 × 16, 2 classes): all 100 fits, 2.6 s/fit (m = 1: 1.7 s, m = 10: 3.4 s);
  TabDPT on the same table 2.9 s, Mitra 2.4 s.
- pmlb_402875677 (1212 × 100, 2 classes): 18 fits (m = 1), 42 s/fit; TabDPT same cells 6.7 s.
- pmlb_211571049 (2000 × 76, 10 classes): 13 fits (m = 1), 59 s/fit; TabDPT same cells 9.3 s.
Estimate: LimiX/TabDPT time ratio ≈ 0.9 at a 435-row panel and ≈ 6.5 at ≥ 1212 rows,
interpolated linearly in panel size and applied to TabDPT's stored per-table T1′ seconds
(6.4 worker-hours in all): **≈ 31 worker-hours** (pessimistic, ratio 8 everywhere: 51).
With 6 workers ≈ 5–9 wall-hours (< 40 h cap), so the run goes ahead. CPU only: the DEV run
was on CPU (`device=cpu`), so MPS is not used, for comparability.

## Analysis (fixed before the run; `summarize_t1p_fm8.py`, which imports `summarize_t1p.py`)
Primary metric macro-F1 on E ∖ L, exactly as for fm6/fm7: per table, mean over reps of
stratified and of random; gap = stratified − random; per m ∈ {1, 3, 5, 10} and for the mean
over m: mean over tables, 10k bootstrap 95% CI over tables (seed 0), two-sided Wilcoxon
signed-rank p over tables **unadjusted** (this is the test used for fm6/fm7 on T1′; no Holm
was applied there), plus Holm over LimiX's four per-budget p values; share of tables with a
positive gap; binary (35) vs multiclass (17); sensitivity without UCI 257 (51 tables).
Shown side by side with fm1–fm5 (stored Study 12) and fm6/fm7 (out/t1p), and with LimiX's
DEV gap. No hypothesis test against the other FMs.

Reading: the paper's statement "All seven FMs show the gap" is extended to T1′ for LimiX if
the mean-over-m gap is positive with a CI excluding 0; at each m the gap is reported as is.

## Compute
6 workers (the machine cap), `nice -n 15`, OMP/MKL/OPENBLAS/VECLIB = 1, memory guard 4 GB.
Done-file `out/t1p_fm8/DONE`; watched on PID. Macros: `out/t1p_fm8_numbers.tex`, prefix
`ExLimixTp`, all tagged EXPLORATORY.
