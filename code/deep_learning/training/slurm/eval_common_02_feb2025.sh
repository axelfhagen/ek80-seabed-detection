#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Common-set re-evaluation on the canonical survey VAL split (oct_2024, 378
# files). See eval_common_01_dec2023.sh for the rationale. Writes into a fresh
# output dir so the checkpoint's training-time eval_metrics.json is preserved.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=eval_common_feb2025
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

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"

NAME=feb_2025
RESUME_DIR="${PROJECT}/code/checkpoints/exp_20260506_193527_survey_feb_2025"
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
