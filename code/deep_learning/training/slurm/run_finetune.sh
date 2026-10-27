#!/bin/bash
#SBATCH --job-name=unet-finetune
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/finetune_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/finetune_%j.err

# ── Environment ───────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate master_env

# ── Configuration ─────────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/finetune_${SLURM_JOB_ID}.out" "${LOG_DIR}/finetune_${TIMESTAMP}_${SLURM_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/finetune_${SLURM_JOB_ID}.err" "${LOG_DIR}/finetune_${TIMESTAMP}_${SLURM_JOB_ID}.err" 2>/dev/null || true
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_finetune"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Checkpoint dir: $CHECKPOINT_DIR"

# ── Training ──────────────────────────────────────────────────────────
# Phase 1: encoder frozen (epochs 1-25)
# Phase 2: encoder unfreezes at epoch 25, trains with lower encoder LR
python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --epochs 250 \
    --batch-size 4 \
    --freeze encoder \
    --unfreeze-after 25 \
    --encoder-lr 1e-5 \
    --decoder-lr 1e-3 \
    --early-stopping-patience 15 \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Finished: $(date)"
