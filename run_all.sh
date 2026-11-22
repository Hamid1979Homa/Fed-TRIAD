#!/usr/bin/env bash
# Reproduce every number, table and figure of Section 6 and the numerical checks of the proofs.
# About 1-2 hours on a 2-core machine (230 simulation runs of ~30 s each).
set -euo pipefail
cd "$(dirname "$0")/src"
JOBS="${JOBS:-2}"

echo "== 1/4 numerical checks of the proofs (each check has a failure control)"
python theory_checks.py

echo "== 2/4 checks of the simulator (reproduces the first study bit for bit; planted faults must fail)"
python test_sim.py ../reference/drift_m12_first_study_seeds1-5.npy

echo "== 3/4 main study and sensitivity study (23 settings x 10 seeds; resumable)"
python run_sweep.py --out ../results --jobs "$JOBS"

echo "== 4/4 summary numbers and figures"
python analyze.py ../results ../outputs
echo "done: see outputs/numbers.json and outputs/*.png"
