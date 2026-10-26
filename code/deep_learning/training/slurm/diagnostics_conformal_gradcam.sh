#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Conformal-band + Grad-CAM diagnostics over a set of checkpoints.
#
# Runs the standalone diagnostics CLI on each checkpoint below. The diagnostics
# suite now adds, on top of its usual plots, locally-weighted conformal
# prediction bands (calibrated on the canonical survey VAL split) and Grad-CAM
# attention panels. Each run writes into the checkpoint's own diagnostics/ dir.
#
# The 5 checkpoints are the canonical-survey-split top performers (all
# smp_unet / resnet34, directly comparable on the oct_2024 val split).
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=diag_conformal_gradcam_top5_survey
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

CHECKPOINTS=(
    chunk_bce_weight_sweep_2
    chunk_arch_sweep_2
    chunk_tversky_sweep_5
    chunk_freeze_sweep_3
    exp_20260513_094152_arch_sweep_1
)

echo "════════════════════════════════════════════════════════════════════"
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Models: ${#CHECKPOINTS[@]}"
echo "Start:  $(date)"
echo "════════════════════════════════════════════════════════════════════"

for d in "${CHECKPOINTS[@]}"; do
    CKPT_DIR="${PROJECT}/code/checkpoints/${d}"
    echo ""
    echo "──────────────────────────────────────────────────────────────────"
    echo "Diagnostics: ${d}"
    echo "──────────────────────────────────────────────────────────────────"
    python -m deep_learning.training.diagnostics \
        --checkpoint-dir "$CKPT_DIR" \
        --split survey \
        --patch-width 512 --stride 512 \
        || echo "WARN: diagnostics failed for ${d}"
done

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
