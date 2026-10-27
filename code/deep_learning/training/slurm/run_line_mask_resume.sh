#!/bin/bash
#SBATCH --job-name=line-mask-resume
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/line_mask_resume_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/line_mask_resume_%j.err

# Resume the boundary (thin-line) bottom segmentation run that hit the 4h wall
# limit (job 24540331). Continues from the saved best checkpoint into the SAME
# checkpoint dir so training_history.json stays continuous. All training args
# must match the original run exactly so the optimizer param groups line up.

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
CACHE_DIR="${PROJECT}/Datasets/patch_datasets/11_line_mask"
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_20260520_113942_line_mask"

cd "$PROJECT"

echo "============================================================"
echo "Line-mask (boundary) training — RESUME"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Resume from:    ${CHECKPOINT_DIR}/smp_unet_resnet34_best.pth"
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
    --cache-dir "$CACHE_DIR" \
    --resume "$CHECKPOINT_DIR"

echo "Finished: $(date)"
