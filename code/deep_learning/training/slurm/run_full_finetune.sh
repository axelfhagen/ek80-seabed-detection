#!/bin/bash
#SBATCH --job-name=unet-full-ft
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/full_finetune_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/full_finetune_%j.err

# ── Environment ───────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate master_env

# ── Configuration ─────────────────────────────────────────────────────
PROJECT=/cluster/work/axelfh/Master
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/full_finetune_${SLURM_JOB_ID}.out" "${LOG_DIR}/full_finetune_${TIMESTAMP}_${SLURM_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/full_finetune_${SLURM_JOB_ID}.err" "${LOG_DIR}/full_finetune_${TIMESTAMP}_${SLURM_JOB_ID}.err" 2>/dev/null || true
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_full_finetune"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Checkpoint dir: $CHECKPOINT_DIR"

# ── Training ──────────────────────────────────────────────────────────
# No freezing at all — train the full model from the start with low LRs.
# The previous finetune run showed that unfreezing at epoch 26 caused a
# destabilizing jump in val loss (0.03 → 0.07). Training everything from
# scratch with a uniformly low encoder LR (5e-5) avoids that transition
# shock. The decoder starts at 5e-4 (lower than prior runs) since the
# encoder features will be shifting from epoch 1. Longer patience (25)
# accommodates the slower convergence expected with full fine-tuning.
python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --epochs 250 \
    --batch-size 4 \
    --freeze none \
    --encoder-lr 5e-5 \
    --decoder-lr 5e-4 \
    --early-stopping-patience 25 \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Finished: $(date)"
