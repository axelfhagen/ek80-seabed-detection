#!/bin/bash
#SBATCH --job-name=line-mask-eval
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/line_mask_eval_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/line_mask_eval_%j.err

# Eval-only: load the current best line-mask checkpoint (from the timed-out run
# 24540331) and generate evaluation metrics + the full diagnostics plot suite
# without any further training. Data/model args must match the training run so
# the val split and model architecture are reconstructed identically.

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
CACHE_DIR="${PROJECT}/Datasets/patch_datasets/11_line_mask"
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_20260520_113942_line_mask"

cd "$PROJECT"

echo "============================================================"
echo "Line-mask (boundary) EVAL-ONLY on current checkpoint"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Diagnostics ->  ${CHECKPOINT_DIR}/diagnostics"
echo "============================================================"

python code/deep_learning/training/train.py \
    --eval-only \
    --split survey \
    --mask-mode line \
    --line-thickness 3 \
    --patch-width 512 \
    --stride 512 \
    --target-size 512 512 \
    --batch-size 8 \
    --freeze encoder \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --resume "$CHECKPOINT_DIR"

echo "Finished: $(date)"
