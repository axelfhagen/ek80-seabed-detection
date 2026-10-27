#!/bin/bash
#SBATCH --job-name=smoke-pipeline
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/smoke_pipeline_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/smoke_pipeline_%j.err

# Pipeline smoketest: 1 epoch, default 'below' mask, small file subset reusing
# the existing patch_cache (config-hashed NPZ from the default params, so the
# cache should hit and data loads fast — if it misses, fresh extraction is still
# a valid test). Exercises the full path: data load -> 1-epoch train -> eval ->
# diagnostics/plots. Sized for any-GPU / 16 GB / 10 min so it schedules quickly.

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_smoke_pipeline"

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

# Small subset: first 10 zarr files (these match already-cached NPZ basenames).
mapfile -t ZARR_PATHS < <(ls -d "${ZARR_DIR}"/*.zarr | head -n 10)

echo "============================================================"
echo "Smoketest: full pipeline, 1 epoch"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Files:  ${#ZARR_PATHS[@]} zarr files"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Cache dir:      $CACHE_DIR"
echo "============================================================"

python code/deep_learning/training/train.py \
    --zarr-paths "${ZARR_PATHS[@]}" \
    --split random \
    --val-fraction 0.2 \
    --epochs 1 \
    --batch-size 4 \
    --freeze encoder \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Finished: $(date)"
