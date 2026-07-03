#!/bin/bash

# Override target / configs at submit time:
#   sbatch scripts/single_run.slurm banana
#   sbatch scripts/single_run.slurm gaussian_2d examples/configs/samplers/nuts.json

#SBATCH -p gpu
#SBATCH --job-name=nuts_test
#SBATCH --output=logs/nuts_test_%j.out
#SBATCH --error=logs/nuts_test_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gpus-per-task=1

set -euo pipefail

module purge
module load python uv

# SLURM sets SLURM_SUBMIT_DIR to the directory you ran sbatch from.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p logs
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export SLURM_STDOUT_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
  export SLURM_STDERR_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

TARGET="${1:-banana}"
SAMPLER_CONFIG="${2:-examples/configs/samplers/nuts.json}"
DEER_CONFIG="${DEER_CONFIG:-examples/configs/deer/default.json}"

if [[ ! -f "${SAMPLER_CONFIG}" ]]; then
  echo "Sampler config not found: ${SAMPLER_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${DEER_CONFIG}" ]]; then
  echo "DEER config not found: ${DEER_CONFIG}" >&2
  exit 1
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Target:         ${TARGET}"

uv run examples/run_nuts.py \
  --target "${TARGET}" \
  --sampler-config "${SAMPLER_CONFIG}" \
  --deer-config "${DEER_CONFIG}"

# Copy logs again after Python exits so the final lines are included.
LATEST_RUN="${REPO_ROOT}/experiments/nuts/individual_runs/latest_run.txt"
if [[ -n "${SLURM_JOB_ID:-}" && -f "${LATEST_RUN}" ]]; then
  RUN_DIR=$(cat "${LATEST_RUN}")
  for pair in "out:slurm.out" "err:slurm.err"; do
    ext="${pair%%:*}"
    dest_name="${pair##*:}"
    src="${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${ext}"
    if [[ -f "${src}" && -d "${RUN_DIR}" ]]; then
      cp "${src}" "${RUN_DIR}/${dest_name}"
    fi
  done
fi
