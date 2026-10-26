#!/bin/bash
#SBATCH --job-name=line-mask
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/line_mask_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/line_mask_%j.err

# Real run: boundary (thin-line) bottom segmentation instead of the default
# 'below' region fill. Canonical survey split, full epoch budget with early
# stopping. Line-mode patches cache separately from the below-mode caches.

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
CACHE_DIR="${PROJECT}/Datasets/patch_datasets/11_line_mask"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_line_mask"

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

echo "============================================================"
echo "Line-mask (boundary) training — canonical survey split"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Cache dir:      $CACHE_DIR"
echo "============================================================"

python code/deep_learning/training/train.py \
    --split survey \
    --mask-mode line \
    --line-thickness 3 \
    --patch-width 512 \
    --stride 512 \
    --target-size 512 512 \
    --epochs 250 \
    --batch-size 8 \
    --freeze encoder \
    --early-stopping-patience 15 \
    --scheduler-type plateau \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Finished: $(date)"
