#!/bin/bash
#SBATCH --job-name=unet_ablation_target_h1024_w512_tall
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/ablation_target_h1024_w512_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/ablation_target_h1024_w512_%j.err

# ── Environment ───────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate master_env

# ── Configuration ─────────────────────────────────────────────────────
# Ablation: baseline config (resnet34 / dice_bce / bce-weight 0.5,
# patch-width=stride=512, freeze encoder, plateau scheduler) with the
# model input resized to an ASYMMETRIC 1024(H) x 512(W) instead of the
# 512x512 default — i.e. keep the 512-ping width, double the depth/range
# height to 1024. target_size is (height, width), so this is "1024 512".
# The patch cache is shared with the baseline because the resize to
# target_size happens on the fly in the dataset; only the network input
# shape changes. Batch size is halved (8 -> 4) to absorb the doubled
# activation footprint from the taller input.
PROJECT=/cluster/work/axelfh/Master
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/ablation_target_h1024_w512_${SLURM_JOB_ID}.out" \
   "${LOG_DIR}/ablation_target_h1024_w512_${TIMESTAMP}_${SLURM_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/ablation_target_h1024_w512_${SLURM_JOB_ID}.err" \
   "${LOG_DIR}/ablation_target_h1024_w512_${TIMESTAMP}_${SLURM_JOB_ID}.err" 2>/dev/null || true
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_ablation_target_h1024_w512"

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

echo "============================================================"
echo "Ablation: asymmetric target size 1024(H) x 512(W)  (baseline = 512x512)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Cache dir:      $CACHE_DIR"
echo "============================================================"

python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --encoder resnet34 \
    --loss-type dice_bce \
    --bce-weight 0.5 \
    --patch-width 512 \
    --stride 512 \
    --target-size 1024 512 \
    --epochs 250 \
    --batch-size 4 \
    --freeze encoder \
    --scheduler-type plateau \
    --early-stopping-patience 15 \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Finished: $(date)"
