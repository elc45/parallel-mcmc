#!/bin/bash

#SBATCH -p gpu
#SBATCH --job-name=nuts_test
#SBATCH --output=logs/nuts_warmup_%j.out
#SBATCH --error=logs/nuts_warmup_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gpus-per-task=1

set -euo pipefail

module purge
module load python uv

TARGET="${1:-banana}"

uv run examples/run_nuts_warmup.py \
  --target "${TARGET}" \