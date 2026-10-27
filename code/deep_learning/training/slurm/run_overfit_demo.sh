#!/bin/bash
#SBATCH --job-name=overfit_demo
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/overfit_demo_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/overfit_demo_%j.err

# Deliberately overfit, to produce a textbook train-vs-val divergence curve.
# Levers (all the things that suppressed overfitting in the dataloader bench):
#   * --freeze none        : full encoder capacity to memorise (not just decoder)
#   * augmentation OFF      : model sees identical patches every epoch -> can memorise
#   * constant LR           : plateau scheduler with huge patience never decays,
#                             so train loss keeps falling instead of being cut off
#   * small train set       : 12 files (random split), easy to memorise
#   * 400 epochs, no early stopping
# Uses --fast-cache so 400 epochs finish quickly.

set -uo pipefail
module purge
module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"
cd "$PROJECT"

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${PROJECT}/code/checkpoints/overfit_demo_${TS}"
mkdir -p "$RUN_DIR"

# Small fixed file subset (deterministic).
mapfile -t ZARR_PATHS < <(ls -d "${ZARR_DIR}"/*.zarr | sort | head -n 16)

echo "============================================================"
echo "Overfit demo | Job ${SLURM_JOB_ID}"
echo "Node: $(hostname)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Files: ${#ZARR_PATHS[@]}   Run dir: ${RUN_DIR}"
echo "Start: $(date)"
echo "============================================================"

python code/deep_learning/training/train.py \
    --architecture smp_unet \
    --encoder resnet34 \
    --encoder-weights imagenet \
    --freeze none \
    --in-channels 3 \
    --freq1 38000 --freq2 200000 \
    --target-size 512 512 \
    --patch-width 512 --stride 512 \
    --batch-size 16 \
    --num-workers 4 \
    --fast-cache \
    --split random --val-fraction 0.30 --seed 42 \
    --epochs 400 \
    --encoder-lr 3e-4 --decoder-lr 1e-3 \
    --scheduler-type plateau --lr-patience 100000 \
    --p-hflip 0 --p-gauss-noise 0 --p-brightness 0 \
    --p-channel-dropout 0 \
    --p-ping-dropout-zero 0 --p-ping-dropout-shuffle 0 --p-ping-dropout-noise 0 \
    --early-stopping-patience 100000 \
    --no-eval --no-diagnostics \
    --results-log "" \
    --zarr-paths "${ZARR_PATHS[@]}" \
    --cache-dir "$CACHE_DIR" \
    --checkpoint-dir "$RUN_DIR"

echo "============================================================"
echo "Finished: $(date)  | run dir: ${RUN_DIR}"
echo "============================================================"
