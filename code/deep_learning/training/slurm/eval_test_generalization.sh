#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Held-out test-survey evaluation of a trained checkpoint.
#
# Scores any baseline-recipe checkpoint on the canonical TEST surveys
# (feb_2025 + jun_2025) via --eval-split test. Those surveys are excluded from
# BOTH the survey split and the random_pool split, so this is a fair, clean
# generalization number for either training scheme. Run it on the survey
# baseline checkpoint AND each baseline_random_split checkpoint, then compare:
# the random-split runs typically post rosy validation IoU but worse test IoU.
#
# Usage:
#   sbatch eval_test_generalization.sh <RESUME_DIR> <NAME>
#     RESUME_DIR  checkpoint dir (containing *_best.pth) or a .pth file
#     NAME        label for the output subdir (e.g. survey_baseline, rand_seed42)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=eval_test_generalization
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

RESUME_DIR="${1:?Usage: sbatch eval_test_generalization.sh <RESUME_DIR> <NAME>}"
NAME="${2:?Usage: sbatch eval_test_generalization.sh <RESUME_DIR> <NAME>}"
OUT_DIR="${PROJECT}/code/checkpoints/test_generalization/${NAME}"
mkdir -p "$OUT_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "Held-out test-survey eval: ${NAME} (${SLURM_JOB_ID})"
echo "Resume: ${RESUME_DIR}"
echo "Out:    ${OUT_DIR}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "════════════════════════════════════════════════════════════════════"

python code/deep_learning/training/evaluate_checkpoint.py \
    --split survey \
    --eval-split test \
    --no-diagnostics \
    --resume "$RESUME_DIR" \
    --checkpoint-dir "$OUT_DIR" \
    --cache-dir "${PROJECT}/Datasets/patch_cache" \
    --num-workers 4 \
    --architecture smp_unet --encoder resnet34 --encoder-weights imagenet \
    --in-channels 3 --base-filters 32 --model-depth 4 \
    --target-size 512 512 --patch-width 512 --stride 512 \
    --freq1 38000 --freq2 200000

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
