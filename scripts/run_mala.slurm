#!/bin/bash
# Submit from the repo root:
#   mkdir -p logs
#   sbatch scripts/run_mala.slurm
#
# Override the config at submit time:
#   sbatch scripts/run_mala.slurm examples/configs/banana.json
#
# Override branch (default: microcanonical):
#   GIT_BRANCH=microcanonical sbatch scripts/run_mala.slurm
#
# Check status:  squeue -u "$USER"
# Cancel:         scancel <job_id>

#SBATCH --partition=gpu
#SBATCH --job-name=mala_test
#SBATCH --output=logs/mala_test_%j.out
#SBATCH --error=logs/mala_test_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-task=1
#SBATCH --account=ecarlson

set -euo pipefail

# --- Cluster-specific setup (edit these for your site) -----------------------
#
# Many clusters use environment modules. Uncomment and adjust names from
# `module avail` on your login node, e.g.:

module purge
module load cuda/12.8.0
module load python uv


# Activate the venv

uv venv
source .venv/bin/activate


# --- Repo paths --------------------------------------------------------------

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

CONFIG="${1:-examples/configs/mala.json}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Repo:   ${REPO_ROOT}"
echo "Config: ${CONFIG}"
echo "Python: $(which python)"
python -V

python - <<'PY'
import jax
print("JAX version:", jax.__version__)
print("Default backend:", jax.default_backend())
print("Devices:", jax.devices())
if jax.default_backend() != "gpu":
    raise SystemExit(
        "JAX is not using the GPU. Install CUDA jaxlib in your venv "
    )
PY

srun python examples/run_mala.py --config "${CONFIG}"
