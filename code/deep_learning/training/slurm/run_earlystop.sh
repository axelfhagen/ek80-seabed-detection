#!/bin/bash
#SBATCH --job-name=unet-earlystop
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/earlystop_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/earlystop_%j.err

# ── Environment ───────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate master_env

# ── Configuration ─────────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/earlystop_${SLURM_JOB_ID}.out" "${LOG_DIR}/earlystop_${TIMESTAMP}_${SLURM_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/earlystop_${SLURM_JOB_ID}.err" "${LOG_DIR}/earlystop_${TIMESTAMP}_${SLURM_JOB_ID}.err" 2>/dev/null || true
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_earlystop"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Checkpoint dir: $CHECKPOINT_DIR"

# ── Training ──────────────────────────────────────────────────────────
python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --epochs 250 \
    --batch-size 4 \
    --freeze encoder \
    --early-stopping-patience 15 \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Finished: $(date)"
