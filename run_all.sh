#!/usr/bin/env bash
# Reproduce every number, table and figure of Section 6 and the numerical checks of the proofs.
# About 1-2 hours on a 2-core machine (230 simulation runs of ~30 s each).
# Needs bash (Linux, macOS, WSL, or Git Bash on Windows); on plain Windows run the five commands of the README table instead.
set -euo pipefail
cd "$(dirname "$0")/src"
JOBS="${JOBS:-2}"

echo "== 1/5 numerical checks of the proofs (each check has a failure control)"
python theory_checks.py

echo "== 2/5 checks of the simulator (reproduces the first study to a relative tolerance of 1e-4; planted faults must fail)"
python test_sim.py ../reference/drift_m12_first_study_seeds1-5.npy

echo "== 3/5 main study and sensitivity study (23 settings x 10 seeds; resumable)"
python run_sweep.py --out ../results --jobs "$JOBS"

echo "== 4/5 summary numbers and figures"
python analyze.py ../results ../outputs

echo "== 5/5 comparison of the recomputed numbers with the reference numbers of the paper"
python check_against_reference.py ../outputs/numbers.json ../reference/numbers_reference.json

echo "done: see outputs/numbers.json and outputs/*.png"
