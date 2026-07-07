#!/bin/bash

# Run a DEER parallel MCMC example by sampler name.
#
# Usage (sampler-first):
#   sbatch scripts/single_run.slurm nuts
#   sbatch scripts/single_run.slurm nuts banana
#   sbatch scripts/single_run.slurm hmc ill_conditioned_gaussian
#   sbatch scripts/single_run.slurm nuts banana examples/configs/samplers/nuts.json
#   sbatch scripts/single_run.slurm nuts banana examples/configs/samplers/nuts.json examples/configs/deer/quasi-damped.json
#
# Override DEER config via environment when arg 4 is omitted:
#   DEER_CONFIG=examples/configs/deer/full-damped.json sbatch scripts/single_run.slurm nuts banana
#
# Legacy target-first invocation (defaults to nuts):
#   sbatch scripts/single_run.slurm banana
#   sbatch scripts/single_run.slurm banana examples/configs/samplers/nuts.json
#
# Supported samplers: nuts, mala, hmc, mclmc, langevin,
#                     hamiltonian_leapfrog, microcanonical_leapfrog

#SBATCH -p gpu
#SBATCH --job-name=deer_run
#SBATCH --output=logs/deer_run_%j.out
#SBATCH --error=logs/deer_run_%j.err
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

echo "Python: $(command -v python)"
python -V
echo "uv: $(command -v uv)"
uv -V

# The login-node .venv can point at an unavailable Python (e.g. Nix store paths).
# Recreate/sync the project environment on the compute node before running.
if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  echo "Project .venv is missing or invalid; running uv sync..."
else
  echo "Syncing project environment with uv..."
fi
uv sync --frozen

mkdir -p logs
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export SLURM_STDOUT_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
  export SLURM_STDERR_PATH="${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

is_known_sampler() {
  case "$1" in
    nuts|mala|hmc|mclmc|langevin)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_sampler() {
  local sampler="$1"
  case "${sampler}" in
    nuts)
      RUN_SCRIPT="examples/run_nuts.py"
      DEFAULT_SAMPLER_CONFIG="examples/configs/samplers/nuts.json"
      LATEST_RUN="${REPO_ROOT}/experiments/nuts/individual_runs/latest_run.txt"
      ;;
    mala)
      RUN_SCRIPT="examples/run_mala.py"
      DEFAULT_SAMPLER_CONFIG="examples/configs/samplers/mala.json"
      LATEST_RUN="${REPO_ROOT}/experiments/mala/runs/latest_run.txt"
      ;;
    hmc)
      RUN_SCRIPT="examples/run_hmc.py"
      DEFAULT_SAMPLER_CONFIG="examples/configs/samplers/hmc.json"
      LATEST_RUN="${REPO_ROOT}/experiments/hmc_runs/latest_run.txt"
      ;;
    mclmc)
      RUN_SCRIPT="examples/run_mclmc.py"
      DEFAULT_SAMPLER_CONFIG="examples/configs/samplers/mclmc.json"
      LATEST_RUN="${REPO_ROOT}/experiments/mclmc_runs/latest_run.txt"
      ;;
    langevin)
      RUN_SCRIPT="examples/run_langevin.py"
      DEFAULT_SAMPLER_CONFIG="examples/configs/samplers/langevin.json"
      LATEST_RUN="${REPO_ROOT}/experiments/langevin_runs/latest_run.txt"
      ;;
    *)
      echo "Unknown sampler: ${sampler}" >&2
      echo "Choose from: nuts, mala, hmc, mclmc, langevin" >&2
      exit 1
      ;;
  esac
}

if [[ $# -ge 1 ]] && is_known_sampler "$1"; then
  SAMPLER="$1"
  TARGET="${2:-banana}"
  SAMPLER_CONFIG="${3:-}"
  DEER_CONFIG="${4:-${DEER_CONFIG:-examples/configs/deer/default.json}}"
else
  SAMPLER="nuts"
  TARGET="${1:-banana}"
  SAMPLER_CONFIG="${2:-}"
  DEER_CONFIG="${3:-${DEER_CONFIG:-examples/configs/deer/default.json}}"
fi

resolve_sampler "${SAMPLER}"

if [[ -z "${SAMPLER_CONFIG}" ]]; then
  SAMPLER_CONFIG="${DEFAULT_SAMPLER_CONFIG}"
fi

if [[ ! -f "${SAMPLER_CONFIG}" ]]; then
  echo "Sampler config not found: ${SAMPLER_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${DEER_CONFIG}" ]]; then
  echo "DEER config not found: ${DEER_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Run script not found: ${RUN_SCRIPT}" >&2
  exit 1
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Sampler:        ${SAMPLER}"
echo "Run script:     ${RUN_SCRIPT}"
echo "Target:         ${TARGET}"
echo "Sampler config: ${SAMPLER_CONFIG}"
echo "DEER config:    ${DEER_CONFIG}"
echo "Latest run:     ${LATEST_RUN}"

uv run "${RUN_SCRIPT}" \
  --target "${TARGET}" \
  --sampler-config "${SAMPLER_CONFIG}" \
  --deer-config "${DEER_CONFIG}"

# Copy logs again after Python exits so the final lines are included.
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
