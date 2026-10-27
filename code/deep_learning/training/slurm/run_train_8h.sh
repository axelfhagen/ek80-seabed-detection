#!/bin/bash
#SBATCH --job-name=t8h
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/t8h_%A_%a.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/t8h_%A_%a.err

# 8h training chunk that auto-resumes.
#
# Keep wall time bounded so jobs backfill and rarely get preempted, then
# continue the SAME run on the next submission. To make that work the
# checkpoint dir is STABLE (no timestamp) -- re-submitting this script for the
# same grid+task picks up exactly where the last chunk left off. 8h is long
# enough that most runs finish (or early-stop) within a single chunk.
#
# Usage (array, one task per grid index):
#   cd code/deep_learning/training/slurm
#   sbatch --array=1-7%5 run_train_8h.sh loss_sweep
#
# Re-submit the identical command to advance every task by another 8h chunk.
# A task that has already produced eval_metrics.json is finished and skips fast.
#
# NOTE: train.py writes *_last.pth every epoch and *_best.pth on val-IoU
# improvement. Resume prefers *_last.pth, so a chunk continues from exactly
# where it stopped (no rewind to the best epoch, no re-trained gap). Evaluation
# still loads *_best.pth.

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"
GRID_NAME="${1:?Usage: sbatch --array=1-N run_train_8h.sh <grid_name>}"
TASK_ID=$SLURM_ARRAY_TASK_ID
CHECKPOINT_PARENT="${CHECKPOINT_PARENT:-${PROJECT}/code/checkpoints}"

cd "$PROJECT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/t8h_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out" \
   "${LOG_DIR}/t8h_${GRID_NAME}_${TASK_ID}_${TIMESTAMP}_${SLURM_ARRAY_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/t8h_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.err" \
   "${LOG_DIR}/t8h_${GRID_NAME}_${TASK_ID}_${TIMESTAMP}_${SLURM_ARRAY_JOB_ID}.err" 2>/dev/null || true

# Stable, timestamp-free checkpoint dir so chunks accumulate in place.
CHECKPOINT_DIR="${CHECKPOINT_PARENT}/chunk_${GRID_NAME}_${TASK_ID}"
mkdir -p "$CHECKPOINT_DIR"

# Already finished? eval_metrics.json is written only after evaluation.
if [ -f "${CHECKPOINT_DIR}/eval_metrics.json" ]; then
    echo "Task ${GRID_NAME}/${TASK_ID} already has eval_metrics.json -- nothing to do."
    exit 0
fi

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

# Resume only if a checkpoint already exists in the stable dir. Check both
# names: train.py resumes from _last.pth (continue where it stopped) and only
# falls back to _best.pth, so either one being present means there is state to
# continue from.
RESUME_ARG=""
if ls "${CHECKPOINT_DIR}"/*_last.pth >/dev/null 2>&1 \
   || ls "${CHECKPOINT_DIR}"/*_best.pth >/dev/null 2>&1; then
    RESUME_ARG="--resume ${CHECKPOINT_DIR}"
    echo "Resuming from ${CHECKPOINT_DIR}"
else
    echo "Fresh start in ${CHECKPOINT_DIR}"
fi

echo "============================================================"
echo "8h chunk: ${GRID_NAME} | Task: ${TASK_ID} | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Config: ${CONFIG_JSON}"
echo "Ckpt:   ${CHECKPOINT_DIR}  (${RESUME_ARG:-fresh})"
echo "Start:  $(date)"
echo "============================================================"

# --fast-cache: serve patches from pre-resized, uncompressed, mmap-able .npy
# (pre-built by slurm/convert_cache_to_fast.sh). ~20x faster dataloading;
# data is bit-identical to the .npz path, so resumed runs and cross-run
# comparisons stay valid. A file missing its .npy is built once from the .npz.
python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --num-workers 4 \
    --fast-cache \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR" \
    $RESUME_ARG \
    $CLI_ARGS

echo "============================================================"
echo "Finished chunk: $(date)"
echo "============================================================"
