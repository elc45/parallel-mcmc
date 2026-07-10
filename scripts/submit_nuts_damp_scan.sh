#!/bin/bash

# Submit the damp-factor array job and a plot job that runs after it completes.
#
# Usage:
#   bash scripts/submit_nuts_damp_scan.sh
#   STEP_SIZE=0.1 RANDOM_SEEDS="1 2 3 4 5" bash scripts/submit_nuts_damp_scan.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

# Propagate overrides into both jobs.
export STEP_SIZE="${STEP_SIZE:-0.05}"
export DAMP_FACTORS="${DAMP_FACTORS:-0.001 0.01 0.25 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
export RANDOM_SEEDS="${RANDOM_SEEDS:-123 456 789 1011 1213}"

IFS=' ' read -r -a _DAMP <<< "${DAMP_FACTORS}"
LAST_IDX=$(( ${#_DAMP[@]} - 1 ))

ARRAY_ID=$(sbatch --parsable --array="0-${LAST_IDX}" scripts/nuts_damp_scan.slurm)
PLOT_ID=$(sbatch --parsable --dependency="afterok:${ARRAY_ID}" \
  --export=ALL,STEP_SIZE,RESULTS_DIR,OUTPUT \
  scripts/nuts_damp_scan_plot.slurm)

echo "Submitted damp-factor array job: ${ARRAY_ID} (tasks 0-${LAST_IDX})"
echo "Submitted plot job (after array): ${PLOT_ID}"
echo "step_size=${STEP_SIZE}"
echo "random_seeds=${RANDOM_SEEDS}"
echo "Results: experiments/nuts/scans/damp_factor/blr_german_credit/results/eps_${STEP_SIZE}/"
echo "Plot:    experiments/nuts/scans/damp_factor/blr_german_credit/results/eps_${STEP_SIZE}/n_iters_vs_damp_factor.png"
