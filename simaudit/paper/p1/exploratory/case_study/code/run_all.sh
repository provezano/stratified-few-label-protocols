#!/bin/bash
# Full case-study run (PREREG Amendment 2): S (gate) then R; stage A in the TabEBM venv, stage B in the P1 venv.
set -u
CS=$(pwd)/paper/p1/exploratory/case_study  # run from simaudit/
VA=$CS/.venv_tabebm/bin/python
VB=$(pwd)/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 HF_HUB_OFFLINE=1
cd $CS
for P in S R; do
  nice -n 15 $VA -I code/run_stage_a.py --protocol $P --workers 6 >> out/log_stage_a_$P.txt 2>&1
  nice -n 15 $VB -I code/run_stage_b.py --protocol $P --workers 6 >> out/log_stage_b_$P.txt 2>&1
done
date -u > out/ALL.done
