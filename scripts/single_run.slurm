#!/bin/bash

# Override target / configs at submit time:
#   sbatch scripts/single_run.slurm banana
#   sbatch scripts/single_run.slurm gaussian_2d examples/configs/samplers/mala.json
#
# Override branch (default: microcanonical):
#   GIT_BRANCH=microcanonical sbatch scripts/single_run.slurm
#
# Check status:  squeue -u "$USER"
# Cancel:         scancel <job_id>

#SBATCH -p gpu
#SBATCH --job-name=mala_test
#SBATCH --output=logs/mala_test_%j.out
#SBATCH --error=logs/mala_test_%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gpus-per-task=1

set -euo pipefail

# --- Cluster-specific setup (edit these for your site) -----------------------
#
# Many clusters use environment modules. Uncomment and adjust names from
# `module avail` on your login node, e.g.:

module purge
module load python uv

# SLURM sets SLURM_SUBMIT_DIR to the directory you ran sbatch from.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

BRANCH="${GIT_BRANCH:-microcanonical}"
if ! git checkout "${BRANCH}" 2>/dev/null; then
  git fetch origin "${BRANCH}"
  git checkout "${BRANCH}"
fi
echo "On branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

TARGET="${1:-banana}"
SAMPLER_CONFIG="${2:-examples/configs/samplers/mala.json}"
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
echo "Repo:           ${REPO_ROOT}"
echo "Target:         ${TARGET}"
echo "Sampler config: ${SAMPLER_CONFIG}"
echo "DEER config:    ${DEER_CONFIG}"
echo "Python: $(which python)"
python -V

uv run examples/run_mala.py \
  --target "${TARGET}" \
  --sampler-config "${SAMPLER_CONFIG}" \
  --deer-config "${DEER_CONFIG}"
