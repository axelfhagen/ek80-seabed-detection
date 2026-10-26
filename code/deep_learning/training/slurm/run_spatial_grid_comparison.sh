#!/bin/bash
# Spatial grid comparison of predicted vs label seabed depth across train/val/test.
# Runs full-echogram inference over each split, caches per-ping tables, and renders
# the geographic disagreement maps in Figures/spatial_grid_comparison/.
# Adapted from slurm_template.sh.
# ── SBATCH directives ────────────────────────────────────────────────────────
#SBATCH --job-name=spatial_grid_comparison
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

echo "════════════════════════════════════════════════════════════════════"
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname)"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
fi
echo "Start:  $(date)"
echo "════════════════════════════════════════════════════════════════════"

python code/data_visualization/gen_spatial_grid_comparison.py \
    --splits train val test \
    --lat-step 0.001 \
    --lon-step 0.002 \
    --min-count 10

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
