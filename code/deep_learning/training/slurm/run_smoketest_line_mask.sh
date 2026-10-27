#!/bin/bash
#SBATCH --job-name=smoke-line-mask
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/smoke_line_mask_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/smoke_line_mask_%j.err

# Smoketest: train with mask_mode='line' (thin bottom boundary) instead of the
# default 'below' region fill. Small file subset + few epochs so it finishes
# inside the 20-minute / 16 GB / any-GPU budget and schedules fast.

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_datasets/smoke_line_mask"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_smoke_line_mask"

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

# Small subset for a quick run: first 10 zarr files.
mapfile -t ZARR_PATHS < <(ls -d "${ZARR_DIR}"/*.zarr | head -n 10)

echo "============================================================"
echo "Smoketest: line-mask (boundary) training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Files:  ${#ZARR_PATHS[@]} zarr files"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Cache dir:      $CACHE_DIR"
echo "============================================================"

python code/deep_learning/training/train.py \
    --zarr-paths "${ZARR_PATHS[@]}" \
    --split random \
    --val-fraction 0.2 \
    --mask-mode line \
    --line-thickness 3 \
    --target-size 512 512 \
    --epochs 10 \
    --batch-size 4 \
    --freeze encoder \
    --early-stopping-patience 5 \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Finished: $(date)"
