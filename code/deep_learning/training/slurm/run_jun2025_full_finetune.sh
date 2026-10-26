#!/bin/bash
#SBATCH --job-name=jun25_ft
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/jun2025_full_ft_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/jun2025_full_ft_%j.err

# Variant: no encoder freezing — full fine-tune from the start.
# With 1727 files the dataset may be large enough to justify this.
# Submit with: sbatch run_jun2025_full_finetune.sh

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR="code/checkpoints/exp_${TIMESTAMP}_jun2025_full_ft"

echo "============================================================"
echo "Jun 2025 — full finetune (no freeze) | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "============================================================"

python code/deep_learning/training/train.py \
    --zarr-dir Datasets/echopype_zarr \
    --cache-dir Datasets/patch_cache \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --encoder resnet34 \
    --freeze none \
    --encoder-lr 1e-04 \
    --decoder-lr 0.001 \
    --scheduler-type plateau \
    --epochs 250 \
    --batch-size 8 \
    --num-workers 4 \
    --early-stopping-patience 20 \
    --patch-width 512 \
    --stride 512 \
    --target-size 512 512 \
    --in-channels 3 \
    --freq1 38000 \
    --freq2 200000 \
    --bce-weight 0.5 \
    --seed 42

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
