#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Common-set re-evaluation of a top checkpoint.
#
# All 5 "best" checkpoints were originally evaluated on DIFFERENT validation
# sets (single-survey random splits vs. the canonical oct_2024 val split), so
# their logged IoUs are not directly comparable. This re-evaluates each one on
# the SAME held-out set — the canonical survey VAL split (oct_2024, 378 files),
# which none of these 5 models trained on — so the metrics line up.
#
# Writes eval_metrics.json into code/checkpoints/common_eval_oct2024/<name>/
# (a fresh dir) so each checkpoint's training-time eval_metrics.json is preserved.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=eval_common_dec2023
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:40:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

NAME=dec_2023
RESUME_DIR="${PROJECT}/code/checkpoints/exp_20260506_163907_survey_dec_2023"
OUT_DIR="${PROJECT}/code/checkpoints/common_eval_oct2024/${NAME}"
mkdir -p "$OUT_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "Common-set eval: ${NAME} (${SLURM_JOB_ID})"
echo "Resume: ${RESUME_DIR}"
echo "Out:    ${OUT_DIR}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "════════════════════════════════════════════════════════════════════"

python code/deep_learning/training/train.py \
    --zarr-dir "${PROJECT}/Datasets/echopype_zarr" \
    --split survey \
    --eval-only \
    --no-diagnostics \
    --resume "$RESUME_DIR" \
    --checkpoint-dir "$OUT_DIR" \
    --cache-dir "${PROJECT}/Datasets/patch_cache" \
    --num-workers 4 \
    --results-log "" \
    --architecture smp_unet --encoder resnet34 --encoder-weights imagenet \
    --in-channels 3 --base-filters 32 --model-depth 4 \
    --target-size 512 512 --patch-width 512 --stride 512 \
    --freq1 38000 --freq2 200000

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
