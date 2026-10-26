#!/bin/bash
#SBATCH --job-name=extract-missing
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/extract_missing_%A_%a.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/extract_missing_%A_%a.err

# Extract patch caches for the calibration (jan_2025) and test (feb_2025,
# jun_2025) splits, using the canonical convention (pw512/s512, full crop,
# 'below' mask, minmax -80/-34) so they match the existing train/val caches.
#
# 8-way array job; ensure_cached() takes a per-file lock so shards are safe.
# Submit with: sbatch code/deep_learning/training/slurm/run_extract_missing.sh

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

echo "============================================================"
echo "Extract missing patches | Job: ${SLURM_ARRAY_JOB_ID} Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "============================================================"

python code/deep_learning/training/extract_missing_patches.py \
    --splits calibration test \
    --cache-dir Datasets/patch_cache \
    --zarr-dir Datasets/echopype_zarr

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
