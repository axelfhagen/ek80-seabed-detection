#!/bin/bash
#SBATCH --job-name=unet_line_tversky_a085_t2_pw512_resnet34
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/line_tversky_fn_t2_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/line_tversky_fn_t2_%j.err

# Line-mode (boundary) bottom detection with a false-negative-heavy Tversky
# loss (alpha=0.85, beta=0.15). On a thin line the dominant failure is a
# sparse/broken prediction (missed line pixels = FN), so penalising FN much
# harder than FP pushes the model to keep the line continuous. Thickness=2
# gives a little more positive signal for the overlap-based loss to latch
# onto. Checkpoint selection on bottom_mae.

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
CACHE_DIR="${PROJECT}/Datasets/patch_datasets/line_t2_pw512"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_line_tversky_fn_t2"

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

echo "============================================================"
echo "Line Tversky FN-heavy (a0.85/b0.15) thickness=2 — survey split"
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
    --line-thickness 2 \
    --patch-width 512 \
    --stride 512 \
    --target-size 512 512 \
    --loss-type tversky \
    --tversky-alpha 0.85 \
    --tversky-beta 0.15 \
    --monitor-metric bottom_mae \
    --epochs 300 \
    --batch-size 8 \
    --freeze encoder \
    --early-stopping-patience 25 \
    --scheduler-type plateau \
    --lr-patience 8 \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Finished: $(date)"
