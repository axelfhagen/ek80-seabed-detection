#!/bin/bash
#SBATCH --job-name=aug_abl
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/aug_abl_%A_%a.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/aug_abl_%A_%a.err

# Augmentation ablation on a 100-file random subset of the full dataset (frozen
# in aug_ablation_file_list.txt). Each array task trains one config from the
# `aug_ablation` grid. Patches are 256/64, random file-level split
# (val-fraction 0.2 -> 80 train / 20 val files). The fast cache stores patches
# pre-resized to a fixed 512x512 (no resize at training time). All tasks share
# one dedicated fast cache (augmentation is applied on-the-fly, never cached),
# so the cache should be pre-warmed once before submitting the array.

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
CACHE_DIR="${PROJECT}/Datasets/patch_cache_aug_ablation_fast"
GRID_NAME="aug_ablation"
TASK_ID=$SLURM_ARRAY_TASK_ID
FILE_LIST="${PROJECT}/code/deep_learning/training/slurm/aug_ablation_file_list.txt"

cd "$PROJECT"

# 100-file random subset (frozen list, identical for every task and the prewarm).
mapfile -t ZARR_NAMES < "$FILE_LIST"
ZARR_PATHS=()
for n in "${ZARR_NAMES[@]}"; do
    [ -n "$n" ] && ZARR_PATHS+=( "${PROJECT}/Datasets/echopype_zarr/${n}" )
done
echo "Selected ${#ZARR_PATHS[@]} zarr stores"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/aug_abl_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out" \
   "${LOG_DIR}/aug_abl_${TASK_ID}_${TIMESTAMP}_${SLURM_ARRAY_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/aug_abl_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.err" \
   "${LOG_DIR}/aug_abl_${TASK_ID}_${TIMESTAMP}_${SLURM_ARRAY_JOB_ID}.err" 2>/dev/null || true

CONFIG_JSON=$(python code/deep_learning/training/slurm/experiment_grid.py \
    --grid "$GRID_NAME" --index "$TASK_ID")
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to get config for grid=$GRID_NAME index=$TASK_ID" >&2
    exit 1
fi

CLI_ARGS=$(python -c "
import json, sys
cfg = json.loads(sys.argv[1])
args = []
for k, v in cfg.items():
    args.append(f'--{k}')
    args.append(str(v))
print(' '.join(args))
" "$CONFIG_JSON")

CHECKPOINT_DIR="code/checkpoints/exp_${TIMESTAMP}_${GRID_NAME}_${TASK_ID}"

echo "============================================================"
echo "Aug ablation | Task: ${TASK_ID} | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Config: ${CONFIG_JSON}"
echo "Start:  $(date)"
echo "============================================================"

python code/deep_learning/training/train.py \
    --num-workers 4 \
    --no-diagnostics \
    --fast-cache \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR" \
    $CLI_ARGS \
    --zarr-paths "${ZARR_PATHS[@]}"

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
