#!/bin/bash

# SLURM array job: sweep MALA step size (sampler config key: epsilon).
#
# Submit:
#   sbatch scripts/step_size_scan.slurm
#   sbatch scripts/step_size_scan.slurm banana
#   sbatch scripts/step_size_scan.slurm banana examples/configs/samplers/mala.json
#
# Override step sizes (comma-separated, no spaces):
#   STEP_SIZES=0.01,0.02,0.05 sbatch scripts/step_size_scan.slurm
#
# Override branch:
#   GIT_BRANCH=microcanonical sbatch scripts/step_size_scan.slurm
#
# Outputs land under:
#   experiments/mala/scans/step_size/<target>/epsilon_<value>/
# Scan metadata: experiments/mala/scans/step_size/scan.json
#
# Check status:  squeue -u "$USER"
# Cancel:         scancel <job_id>

#SBATCH -p gpu
#SBATCH --job-name=mala_step_scan
#SBATCH --output=logs/mala_step_scan_%A_%a.out
#SBATCH --error=logs/mala_step_scan_%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gpus-per-task=1
#SBATCH --array=0-5

set -euo pipefail

module purge
module load python uv

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

BRANCH="${GIT_BRANCH:-microcanonical}"
if ! git checkout "${BRANCH}" 2>/dev/null; then
  git fetch origin "${BRANCH}"
  git checkout "${BRANCH}"
fi
echo "On branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p logs
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export SLURM_STDOUT_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}.out"
  export SLURM_STDERR_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}.err"
fi

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

# Default sweep; override with STEP_SIZES=0.01,0.02,... at submit time.
IFS=',' read -r -a STEP_SIZES <<< "${STEP_SIZES:-0.005,0.01,0.02,0.05,0.1,0.2}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#STEP_SIZES[@]} )); then
  echo "Array task ${TASK_ID} out of range for ${#STEP_SIZES[@]} step sizes" >&2
  exit 1
fi
EPSILON="${STEP_SIZES[$TASK_ID]}"

SCAN_ROOT="${REPO_ROOT}/experiments/mala/scans/step_size"
TARGET_SCAN_DIR="${SCAN_ROOT}/${TARGET}"
RUN_LABEL="epsilon_${EPSILON}"
RUN_DIR="${TARGET_SCAN_DIR}/${RUN_LABEL}"

mkdir -p "${TARGET_SCAN_DIR}"

python - "${SCAN_ROOT}" "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}" "${TARGET}" "${SAMPLER_CONFIG}" "${DEER_CONFIG}" "${STEP_SIZES[*]}" <<'PY'
import json
import sys
from pathlib import Path

scan_root = Path(sys.argv[1])
array_job_id = sys.argv[2]
target = sys.argv[3]
sampler_config = sys.argv[4]
deer_config = sys.argv[5]
step_sizes = [float(x) for x in sys.argv[6].split()]

meta = {
    "scan": "step_size",
    "sampler_parameter": "epsilon",
    "step_sizes": step_sizes,
    "target": target,
    "sampler_config": sampler_config,
    "deer_config": deer_config,
    "slurm_array_job_id": array_job_id,
}
(scan_root / "scan.json").write_text(json.dumps(meta, indent=2) + "\n")
PY

echo "Job ${SLURM_JOB_ID:-local} task ${TASK_ID} on $(hostname)"
echo "Repo:           ${REPO_ROOT}"
echo "Target:         ${TARGET}"
echo "Sampler config: ${SAMPLER_CONFIG}"
echo "DEER config:    ${DEER_CONFIG}"
echo "Scan root:      ${SCAN_ROOT}"
echo "Target scan:    ${TARGET_SCAN_DIR}"
echo "Run dir:        ${RUN_DIR}"
echo "Epsilon:        ${EPSILON}"
echo "Python: $(which python)"
python -V

uv run examples/run_mala.py \
  --target "${TARGET}" \
  --sampler-config "${SAMPLER_CONFIG}" \
  --deer-config "${DEER_CONFIG}" \
  --epsilon "${EPSILON}" \
  --run-dir "${RUN_DIR}" \
  --latest-run-parent "${SCAN_ROOT}"
