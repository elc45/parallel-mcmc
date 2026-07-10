#!/bin/bash

# Combine nuts_damp_scan array results into a single plot.
# CPU-only: uses .venv/bin/python and a matplotlib-only script (no JAX import).
#
# Plots whatever task_*.json files are present (any number of damp factors /
# seeds). Override RESULTS_DIR / OUTPUT / STEP_SIZE at submit time if needed.
#
# Submit after the array job, e.g. via scripts/submit_nuts_damp_scan.sh

#SBATCH --job-name=nuts_damp_plot
#SBATCH --output=logs/nuts_damp_plot_%j.out
#SBATCH --error=logs/nuts_damp_plot_%j.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

module purge
module load python

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

STEP_SIZE="${STEP_SIZE:-0.05}"
SCAN_DIR="${REPO_ROOT}/experiments/nuts/scans/damp_factor/blr_german_credit"
RESULTS_DIR="${RESULTS_DIR:-${SCAN_DIR}/results/eps_${STEP_SIZE}}"
OUTPUT="${OUTPUT:-${RESULTS_DIR}/n_iters_vs_damp_factor.png}"
PYTHON="${REPO_ROOT}/.venv/bin/python"

mkdir -p logs

if [[ ! -x "${PYTHON}" ]]; then
  echo "Project venv not found at ${PYTHON}" >&2
  echo "Run the GPU array job first so .venv exists on a compute node." >&2
  exit 1
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Python:       ${PYTHON}"
echo "Results dir:  ${RESULTS_DIR}"
echo "Output plot:  ${OUTPUT}"

"${PYTHON}" -c "import matplotlib; print('matplotlib ok')"

"${PYTHON}" examples/plot_nuts_damp_scan.py \
  --results-dir "${RESULTS_DIR}" \
  --output "${OUTPUT}"
