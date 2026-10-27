#!/bin/bash
#SBATCH --job-name=resume_aug_abl
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/resume_aug_abl_%A_%a.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/resume_aug_abl_%A_%a.err

# Resume interrupted aug_ablation tasks from their timestamped checkpoint dirs.
# Mirrors launch_aug_ablation.sh (same 100-file list, fast cache, --fast-cache,
# --no-diagnostics) but globs the latest exp_*_aug_ablation_N dir and resumes.

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
CACHE_DIR="${PROJECT}/Datasets/patch_cache_aug_ablation_fast"
GRID_NAME="aug_ablation"
TASK_ID=$SLURM_ARRAY_TASK_ID
CHECKPOINT_PARENT="${CHECKPOINT_PARENT:-${PROJECT}/code/checkpoints}"
FILE_LIST="${PROJECT}/code/deep_learning/training/slurm/aug_ablation_file_list.txt"

cd "$PROJECT"

mapfile -t ZARR_NAMES < "$FILE_LIST"
ZARR_PATHS=()
for n in "${ZARR_NAMES[@]}"; do
    [ -n "$n" ] && ZARR_PATHS+=( "${PROJECT}/Datasets/echopype_zarr/${n}" )
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/resume_aug_abl_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out" \
   "${LOG_DIR}/resume_aug_abl_${TASK_ID}_${TIMESTAMP}_${SLURM_ARRAY_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/resume_aug_abl_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.err" \
   "${LOG_DIR}/resume_aug_abl_${TASK_ID}_${TIMESTAMP}_${SLURM_ARRAY_JOB_ID}.err" 2>/dev/null || true

CHECKPOINT_DIR=$(ls -d "${CHECKPOINT_PARENT}/exp_"*"_${GRID_NAME}_${TASK_ID}"/ 2>/dev/null | sort | tail -1)
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: no checkpoint dir matching exp_*_${GRID_NAME}_${TASK_ID}" >&2
    exit 1
fi
CHECKPOINT_DIR="${CHECKPOINT_DIR%/}"

if ! ls "${CHECKPOINT_DIR}"/*_best.pth >/dev/null 2>&1; then
    echo "ERROR: no *_best.pth in ${CHECKPOINT_DIR}" >&2
    exit 1
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

echo "============================================================"
echo "Resume aug_ablation | Task: ${TASK_ID} | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Config: ${CONFIG_JSON}"
echo "Resume: ${CHECKPOINT_DIR}"
echo "Zarr:   ${#ZARR_PATHS[@]} stores"
echo "Start:  $(date)"
echo "============================================================"

python code/deep_learning/training/train.py \
    --num-workers 4 \
    --no-diagnostics \
    --fast-cache \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --resume "$CHECKPOINT_DIR" \
    $CLI_ARGS \
    --zarr-paths "${ZARR_PATHS[@]}"

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
