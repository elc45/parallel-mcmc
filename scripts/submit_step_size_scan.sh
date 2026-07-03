#!/bin/bash
# Submit a step-size scan with array bounds matched to STEP_SIZES.
#
#   ./scripts/submit_step_size_scan.sh
#   ./scripts/submit_step_size_scan.sh banana
#   STEP_SIZES=0.01,0.02,0.05 ./scripts/submit_step_size_scan.sh banana

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEP_SIZES="${STEP_SIZES:-0.01,0.02,0.05,0.005}"
IFS=',' read -r -a _SIZES <<< "${STEP_SIZES}"
LAST_IDX=$(( ${#_SIZES[@]} - 1 ))

export STEP_SIZES
exec sbatch --array="0-${LAST_IDX}" "${REPO_ROOT}/scripts/step_size_scan.slurm" "$@"
