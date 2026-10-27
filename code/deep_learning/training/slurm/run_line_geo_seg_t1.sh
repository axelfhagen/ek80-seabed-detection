#!/bin/bash
#SBATCH --job-name=unet_line_geo_seg_t1_pw512_resnet34
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/line_geo_seg_t1_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/line_geo_seg_t1_%j.err

# Line-mode (boundary) bottom detection with the geometry-aware loss.
# geo_seg adds a differentiable per-ping depth-regression term plus an
# anti-spike curvature term on top of Dice+BCE — directly optimising the
# bottom-LINE position rather than pixel overlap. Reg/smooth weights are
# bumped so the geometry terms dominate the (overlap-useless on a 1px line)
# Dice backbone. Thinnest line (thickness=1) since the soft-argmax depth is
# cleanest on a single ridge. Checkpoint selection on bottom_mae, which stays
# sensitive after IoU saturates near zero on thin structures.

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
CACHE_DIR="${PROJECT}/Datasets/patch_datasets/line_t1_pw512"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_line_geo_seg_t1"

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

echo "============================================================"
echo "Line geo_seg thickness=1 — canonical survey split"
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
    --line-thickness 1 \
    --patch-width 512 \
    --stride 512 \
    --target-size 512 512 \
    --loss-type geo_seg \
    --bce-weight 0.5 \
    --geo-reg-weight 0.5 \
    --geo-smooth-weight 0.2 \
    --geo-depth-scale 0.1 \
    --geo-huber-delta 5.0 \
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
