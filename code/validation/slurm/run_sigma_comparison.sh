#!/bin/bash
#SBATCH --job-name=sigma_compare
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/validation/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/validation/slurm/logs/%x_%j.err

set -euo pipefail
module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

echo "===================================================================="
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "===================================================================="

# Locally-weighted conformal bands with global vs local transition-width sigma,
# on the exchangeable jan/feb/jun_2025 split.
python -u code/validation/generate_sigma_comparison.py \
    --n-viz 10 \
    --viz-seed 7 \
    --device auto \
    "$@"

echo "===================================================================="
echo "Finished: $(date)"
echo "===================================================================="
