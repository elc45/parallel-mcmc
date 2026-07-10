#!/bin/bash

# SLURM array job: one DEER damp_factor per task for NUTS on blr_german_credit.
# Each task averages Newton iterations over RANDOM_SEEDS (default: 5 seeds).
#
# Submit array + plot job:
#   bash scripts/submit_nuts_damp_scan.sh
#
# Or manually:
#   ARRAY_ID=$(sbatch --parsable scripts/nuts_damp_scan.slurm)
#   sbatch --dependency=afterok:${ARRAY_ID} scripts/nuts_damp_scan_plot.slurm
#
# Per-task results:
#   experiments/nuts/scans/damp_factor/blr_german_credit/results/.../task_XX.json
#
# Final plot (after plot job):
#   experiments/nuts/scans/damp_factor/blr_german_credit/results/eps_<STEP_SIZE>/n_iters_vs_damp_factor.png

#SBATCH -p gpu
#SBATCH --job-name=nuts_damp_scan
#SBATCH --output=logs/nuts_damp_scan_%A_%a.out
#SBATCH --error=logs/nuts_damp_scan_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gpus-per-task=1
#SBATCH --array=0-12

set -euo pipefail

module purge
module load python uv

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p logs
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export SLURM_STDOUT_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}.out"
  export SLURM_STDERR_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}.err"
fi

IFS=' ' read -r -a DAMP_FACTORS <<< "${DAMP_FACTORS:-0.001 0.01 0.25 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
IFS=' ' read -r -a RANDOM_SEEDS <<< "${RANDOM_SEEDS:-123 456 789 1011 1213}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#DAMP_FACTORS[@]} )); then
  echo "Array task ${TASK_ID} out of range for ${#DAMP_FACTORS[@]} damp factors" >&2
  exit 1
fi
DAMP_FACTOR="${DAMP_FACTORS[$TASK_ID]}"

STEP_SIZE="${STEP_SIZE:-0.05}"
SCAN_DIR="${REPO_ROOT}/experiments/nuts/scans/damp_factor/blr_german_credit"
RESULTS_DIR="${SCAN_DIR}/results/eps_${STEP_SIZE}"
mkdir -p "${RESULTS_DIR}"

if [[ "${TASK_ID}" == "0" ]]; then
  python - "${SCAN_DIR}" "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}" \
    "${STEP_SIZE}" "${DAMP_FACTORS[*]}" "${RANDOM_SEEDS[*]}" <<'PY'
import json
import sys
from pathlib import Path

scan_dir = Path(sys.argv[1])
array_job_id = sys.argv[2]
step_size = float(sys.argv[3])
damp_factors = [float(x) for x in sys.argv[4].split()]
random_seeds = [int(x) for x in sys.argv[5].split()]

meta = {
    "scan": "damp_factor",
    "sampler": "nuts",
    "target": "blr_german_credit",
    "step_size": step_size,
    "damp_factors": damp_factors,
    "random_seeds": random_seeds,
    "slurm_array_job_id": array_job_id,
}
(scan_dir / "scan.json").write_text(json.dumps(meta, indent=2) + "\n")
PY
fi

echo "Job ${SLURM_JOB_ID:-local} task ${TASK_ID} on $(hostname)"
echo "damp_factor=${DAMP_FACTOR}  step_size=${STEP_SIZE}"
echo "random_seeds=${RANDOM_SEEDS[*]}"

uv run examples/experiment_nuts_damp_scan.py run \
  --target blr_german_credit \
  --step-size "${STEP_SIZE}" \
  --damp-factor "${DAMP_FACTOR}" \
  --task-id "${TASK_ID}" \
  --results-dir "${RESULTS_DIR}" \
  --random-seeds "${RANDOM_SEEDS[@]}"
