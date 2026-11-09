Post-release tables and balance probe (added November 2026)

This folder holds the code, the frozen analysis plans, the per-table results and the per-fit scores of the two analyses
that the paper reports on tables first published after every model was released:

  post_release/   the confirmation on post-release tables (inclusion rule, harvest and curation records, frame, frozen
                  pre-data and mechanism models, predictions, unit runner, analysis, results; Section "post-release
                  tables" of the supplementary material)
  balance_probe/  the composition experiment that splits the balance part (frozen plan, probe, analysis, per-unit scores;
                  Section "balance mechanism" of the supplementary material), and host_check/, an exploratory check that
                  each of the two hosts reproduces its own outputs bit for bit (the small cross-host difference comes from
                  the platform's math library, not from run-to-run randomness)

What is and is not included
- Plans and deviation records are public copies (.txt). Host names, local paths and internal note names are replaced;
  nothing else is changed. The SHA-256 digests in plan/PLAN.sha256 and harvest/INCLUSION_RULE.sha256 refer to the
  unredacted frozen files, which were frozen before the runs they govern. They were not lodged in a public registry.
- Per-fit files leave out the per-row probability and query-index columns, as elsewhere in this repository. Host names
  in the score files are replaced by "host A" and "host B" (two Apple-silicon CPUs).
- No third-party data. Every table is public; harvest/specs.jsonl gives its source, record, DOI, URL, licence and
  first-publication date, and harvest/SHA256SUMS the hash of each encoded table. harvest/tools/ rebuilds the encoded
  tables from the sources.
- The scripts expect the simaudit package of this repository (set SIMAUDIT_ROOT to the simaudit/ folder) and the
  encoded tables under post_release/harvest/npz/ and balance_probe/data/npz/.

The DOI of the archived release that contains this folder is given in the paper's data statement.
