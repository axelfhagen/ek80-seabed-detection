#!/bin/bash
#SBATCH --job-name=unet_bottomline_regression_8h
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

# 8-hour bottom-line REGRESSION experiment.
#
# Instead of segmenting the seabed per pixel, the U-Net's single output channel
# is read per ping (column) as a distribution over depth rows (softmax over
# height); its expectation is the predicted bottom row (soft-argmax / integral
# regression). Trained directly on per-ping depth (Huber + curvature) via
# RegressionLoss -- no per-pixel mask supervision. See
# architectures/regression_unet.py and unet_model.RegressionLoss.
#
# Early stopping patience = 20 on val bottom_mae (the per-ping bottom-row MAE),
# which is the natural regression objective. Checkpoint dir is stable so a
# requeue after preemption resumes in place.
#
# Submit:  sbatch code/deep_learning/training/slurm/run_train_regression_8h.sh

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/bottomline_regression_8h"

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

# Already finished (early-stopped + evaluated)?
if [ -f "${CHECKPOINT_DIR}/eval_metrics.json" ]; then
    echo "Run already has eval_metrics.json -- nothing to do."
    exit 0
fi

# Resume in place if a chunk already ran (e.g. after a preemption/requeue).
RESUME_ARG=""
if ls "${CHECKPOINT_DIR}"/*_last.pth >/dev/null 2>&1 \
   || ls "${CHECKPOINT_DIR}"/*_best.pth >/dev/null 2>&1; then
    RESUME_ARG="--resume ${CHECKPOINT_DIR}"
    echo "Resuming from ${CHECKPOINT_DIR}"
else
    echo "Fresh start in ${CHECKPOINT_DIR}"
fi

echo "============================================================"
echo "Bottom-line regression (8h) | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Ckpt:   ${CHECKPOINT_DIR}  (${RESUME_ARG:-fresh})"
echo "Start:  $(date)"
echo "============================================================"

python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --split survey \
    --num-workers 4 \
    --fast-cache \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --task regression \
    --architecture unet_regression \
    --encoder resnet34 \
    --encoder-weights imagenet \
    --freeze encoder \
    --loss-type regression \
    --monitor-metric bottom_mae \
    --geo-depth-scale 1.0 \
    --geo-huber-delta 5.0 \
    --geo-smooth-weight 0.1 \
    --mask-mode below \
    --patch-width 512 \
    --stride 512 \
    --target-size 512 512 \
    --batch-size 8 \
    --encoder-lr 1e-4 \
    --decoder-lr 1e-3 \
    --scheduler-type plateau \
    --epochs 300 \
    --early-stopping-patience 20 \
    $RESUME_ARG

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
