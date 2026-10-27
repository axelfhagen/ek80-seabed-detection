#!/bin/bash
#SBATCH --job-name=ds-aspect
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/ds_aspect_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/ds_aspect_%j.err

# ── Environment ───────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate master_env

# ── Configuration ─────────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_datasets/03_preserve_aspect"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/ds_aspect_${SLURM_JOB_ID}.out" "${LOG_DIR}/ds_aspect_${TIMESTAMP}_${SLURM_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/ds_aspect_${SLURM_JOB_ID}.err" "${LOG_DIR}/ds_aspect_${TIMESTAMP}_${SLURM_JOB_ID}.err" 2>/dev/null || true
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_ds_aspect"

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

echo "============================================================"
echo "Dataset Experiment: 03_preserve_aspect"
echo "  Target 1024x256 — preserves natural echogram aspect ratio"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Cache dir:      $CACHE_DIR"
echo "============================================================"

python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --target-size 1024 256 \
    --epochs 250 \
    --batch-size 4 \
    --freeze encoder \
    --early-stopping-patience 15 \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Finished: $(date)"
