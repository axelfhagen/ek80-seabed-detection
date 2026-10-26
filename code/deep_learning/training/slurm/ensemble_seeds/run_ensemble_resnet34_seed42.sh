#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Deep ensemble member — ResNet-34 U-Net, seed 42
# Replicates chunk_arch_sweep_2 exactly; only --seed differs across members.
# Members share parent dir code/checkpoints/deep_ensemble_resnet34_seeds/
# (per-seed subfolder avoids overwriting smp_unet_resnet34_best.pth across runs).
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=unet_ensemble_resnet34_seed42
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

CHECKPOINT_DIR="${PROJECT}/code/checkpoints/deep_ensemble_resnet34_seeds/member_seed42"
mkdir -p "$CHECKPOINT_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Seed:   42"
echo "Acct:   share-ie-idi"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Ckpt:   $CHECKPOINT_DIR"
echo "Start:  $(date)"
echo "════════════════════════════════════════════════════════════════════"

# Architecture and hyperparameters identical to chunk_arch_sweep_2.
# Only --seed differs across the 5 members.
# --fast-cache consumes pre-built 512x512 patches (no resizing).
python code/deep_learning/training/train.py \
    --zarr-dir "${PROJECT}/Datasets/echopype_zarr" \
    --cache-dir "${PROJECT}/Datasets/patch_cache" \
    --fast-cache \
    --patch-width 512 \
    --stride 512 \
    --target-size 512 512 \
    --architecture smp_unet \
    --encoder resnet34 \
    --encoder-weights none \
    --freeze none \
    --encoder-lr 1e-3 \
    --decoder-lr 1e-3 \
    --batch-size 8 \
    --epochs 250 \
    --scheduler-type plateau \
    --loss-type dice_bce \
    --early-stopping-patience 15 \
    --num-workers 4 \
    --seed 42 \
    --checkpoint-dir "$CHECKPOINT_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
