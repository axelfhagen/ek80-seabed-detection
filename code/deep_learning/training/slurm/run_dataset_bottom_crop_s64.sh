#!/bin/bash
#SBATCH --job-name=ds-bcrop-s64
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/ds_bcrop_s64_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/ds_bcrop_s64_%j.err

# ── Environment ───────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate master_env

# ── Configuration ─────────────────────────────────────────────────────
PROJECT=/cluster/work/axelfh/Master
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_datasets/15_bottom_crop_s64"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/ds_bcrop_s64_${SLURM_JOB_ID}.out" "${LOG_DIR}/ds_bcrop_s64_${TIMESTAMP}_${SLURM_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/ds_bcrop_s64_${SLURM_JOB_ID}.err" "${LOG_DIR}/ds_bcrop_s64_${TIMESTAMP}_${SLURM_JOB_ID}.err" 2>/dev/null || true
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_ds_bcrop_s64"

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

echo "============================================================"
echo "Dataset Experiment: 15_bottom_crop_s64"
echo "  Dynamic crop around bottom line, dense overlap (stride 64)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Cache dir:      $CACHE_DIR"
echo "============================================================"

python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --patch-width 256 \
    --stride 64 \
    --crop-mode bottom \
    --crop-padding 64 \
    --target-size 256 256 \
    --epochs 250 \
    --batch-size 4 \
    --freeze encoder \
    --early-stopping-patience 15 \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Finished: $(date)"
