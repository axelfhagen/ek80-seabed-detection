#!/bin/bash
#SBATCH --job-name=extract-line-mask
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --array=1-20%20
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/extract_line_mask_%A_%a.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/extract_line_mask_%A_%a.err

# CPU pre-extraction of line-mode (boundary) patch caches for the train+val
# splits, sharded across the array. Bottom detection + patch extraction is
# CPU-bound, so doing this off-GPU frees the A100 to only train. Caches are
# keyed by mask_mode, so these land alongside (not overwriting) below-mode
# caches and are picked up by run_line_mask.sh via the matching cache dir.

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
CACHE_DIR="${PROJECT}/Datasets/patch_datasets/11_line_mask"

cd "$PROJECT"

echo "============================================================"
echo "Extract line-mask patches | task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
echo "Job: ${SLURM_ARRAY_JOB_ID}  Node: $(hostname)  Start: $(date)"
echo "Cache dir: $CACHE_DIR"
echo "============================================================"

python code/deep_learning/training/extract_missing_patches.py \
    --splits train val \
    --mask-mode line \
    --line-thickness 3 \
    --cache-dir "$CACHE_DIR"

echo "Finished: $(date)"
