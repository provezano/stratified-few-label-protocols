# Independent learner seed (Lemma 1 assumption) — EXPLORATORY plan

Written and committed 2026-10-31, before any fit below was run.
EXPLORATORY, not pre-registered. `main.tex` and `prereg/` are not edited.

## Why
Lemma 1 (`lem:composition`) needs the learner's randomness to be independent of the
labelled set L. In the stored runs the FM seed equals the selection seed:
- Study 7 Part (a) (DEV): `runner/study7.py::_strategy_cells` passes `meta["seed"]`
  (= `selection_seed(...)`, namespace 10) to `_fit`, i.e. the same integer seeds
  `np.random.default_rng` in the sampler and `random_state` of the FM.
- Study 13 (HELD-OUT and DEV): `cell_seed` is "Seed of one selection and all its fits"
  (namespace 34); `_fit_cell` uses `meta["seed"]`.
- Study 12 (T1′): same pattern (`selection_seed`, namespace 33, used for the fit).
- Study 9 is not affected: it has a separate `fm_seed` (different entropy from
  `selection_seed`).
The sampler stream (PCG64 from `SeedSequence(seed)`) and the FM's stream
(`random_state=seed` → numpy/torch generators) are different generators, but both are
deterministic functions of the same integer, so the lemma holds only approximately.
Question: do the stratified − random gaps change when the FM seed is drawn independently
of the selection?

## Design
Keep exactly the stored labelled sets (same rows for S and R, same row order, same
features); replace only the learner seed.
- New fit seed: `SeedSequence([base_seed, seed_did(did), mult, rep,
  study4.ORDER.index(strategy), 0xF00D]).spawn(1)[0].generate_state(1)[0] % (2**31 − 1)`.
  Namespace 0xF00D (61453) is never used for selection (selection namespaces: 5, 10, 33,
  34, 35, 36). The driver asserts, before any fit: (i) fit seed ≠ selection seed in every
  cell; (ii) the set of new fit seeds is disjoint from the set of all stored selection
  seeds of the job (all strategies); (iii) the first 64 draws of
  `default_rng(selection seed)` and `default_rng(fit seed)` differ.
- Labelled sets are read from the stored caches, never recomputed:
  DEV `results/study7/selections/{did}_{stratified,random}.parquet` (5 S, 20 R draws per m,
  m ∈ {1,3,5,10}), and the driver checks they equal the `labeled` column of
  `results/study7/a/{did}_{fm}.parquet`. Fit/metrics: `runner.study7._fit` unchanged
  (`complement_metrics`, scored on E ∖ L).
- Reproduction control (same host as the run): per FM, 20 cells (one S and one R draw at
  m = 3, rep 0, on 10 tables) refitted with the ORIGINAL seed; report max |Δ macro-F1| and
  max |Δ proba| vs the stored values. This separates seed effects from host/library drift.

## Scope (in this order)
1. DEV, Study 7 Part (a), arms S and R, all m, all reps (2000 fits per learner):
   fm2 (TabICL), fm1 (TabPFN v2), fm5 (TabPFN v3.5, only if the library finds its
   token; no `.env` is read by this driver), fm3 (Kumo).
2. DEV, the six classical learners of E1 (lr, knn1, knn3, gnb, lda, hgb) on the same
   draws with the E1 conventions (`e1_classical.predictor`, pool z-score); originals from
   `paper/p1/exploratory/out/e1_runs.parquet`. Only hgb takes a seed (and with
   `early_stopping=False` and no feature subsampling it should not use it), so the
   expected change is exactly zero; run because it is cheap.
3. If time allows: HELD-OUT, Study 13 arm T, stratified + random, fm2 only, LOCAL only.
   Read only through the study13 path: `check_heldout_go` (P1_HELDOUT_GO=1),
   `verify_manifest` (byte hashes), `sealed_loader(..., normalise_labels=True)` with
   content-hash checks, the study13 row permutation, and the stored selections
   `results/study13/heldout/selections/{key}.parquet`; fits by `study13.fit_record`
   (`panel_metrics`, query = E ∖ L). Accesses go to a separate access log under
   `results/seed_indep/heldout/` (git-ignored); raw HELD-OUT fits stay there (git-ignored);
   only aggregates go to `out/`. Nothing HELD-OUT is copied to another machine.

## Outcomes
Gap per table = mean over m ∈ {1,3,5,10} of [mean over S draws − mean over R draws]
(`analysis.study12.cell_means`/`delta`, as in the paper). Per learner, original seeds vs
independent seeds:
- gap mean over tables, 95% table-bootstrap CI (`report.bootstrap_ci`, 10k, seed 0),
  Wilcoxon signed-rank p;
- difference of gaps d_t = gap_indep_t − gap_orig_t: mean, 95% table-bootstrap CI,
  Wilcoxon p; also per m (descriptive);
- share of fit cells with |macro-F1_indep − macro-F1_orig| > 0.01 (and mean |Δ|);
- coverage/balance split (observed parts as in E1 `theory_table`: miss part =
  mean_R[(M̄^S − M^R)·1{R misses a class}], balance part = same over covered R draws),
  orig vs indep, difference with CI. These are computed from the stored metrics.
Primary metric macro-F1; accuracy and balanced accuracy secondary.

## Reading (fixed before the run)
- **Materially unchanged**: for every learner run, the 95% CI of the difference of gaps
  (macro-F1, mean over m) lies within ±0.01.
- **Changed**: some learner's difference CI excludes 0 and its mean exceeds 0.01 in
  absolute value.
- Otherwise **inconclusive** for that learner (CI crosses ±0.01).
The same rule is reported (descriptively) for the coverage and balance parts and for
the HELD-OUT fm2 run.

## Compute
DEV and HELD-OUT both on this Mac (host recorded in `out/provenance.json`): at most
3 worker processes, `nice -n 15`, 1 BLAS/torch thread each. A second machine is not used (one host for the whole DEV part keeps the reproduction control meaningful).
Code: `run_seed_indep.py` (driver), `analyze_seed_indep.py`. No change to `src/`.
Outputs: `out/dev/*.parquet` (raw DEV fits, no probabilities), `out/repro.csv`,
`out/summary_*.csv`, `out/SUMMARY.md`, `seed_indep_numbers.tex` (prefix `ExSeedInd`,
every macro tagged EXPLORATORY).
