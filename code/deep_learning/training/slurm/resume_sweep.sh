#!/bin/bash
#SBATCH --job-name=resume
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/resume_%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/resume_%x_%j.err

# Resume an interrupted sweep from a task's existing best checkpoint.
# The original arch_sweep run (job 24460646) was cut off mid-training: 8/9
# tasks hit the 8h wall limit on slow P100 GPUs and task 6 was OOM-killed.
# This pins A100s and continues each task from its saved checkpoint dir so
# the runs reach early-stopping and evaluation.
#
# Submit as individual jobs (easier to backfill-schedule than one big array),
# sizing memory per model. The task id is the second argument:
#   cd code/deep_learning/training/slurm
#   sbatch --job-name=resume_arch_4 resume_sweep.sh arch_sweep 4
#   sbatch --job-name=resume_arch_6 --mem=32G resume_sweep.sh arch_sweep 6
#
# It also still works as an array job (SLURM_ARRAY_TASK_ID takes precedence):
#   sbatch --array=1-9%5 resume_sweep.sh arch_sweep
#
# The checkpoint dir for the task is discovered by globbing
# code/checkpoints/exp_*_<grid>_<taskid>; --resume + --checkpoint-dir both
# point at it so training continues in place and the history stays continuous.

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"
GRID_NAME="${1:?Usage: sbatch resume_sweep.sh <grid_name> <task_id>}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${2:?Usage: sbatch resume_sweep.sh <grid_name> <task_id>}}"
CHECKPOINT_PARENT="${CHECKPOINT_PARENT:-${PROJECT}/code/checkpoints}"

cd "$PROJECT"

# Locate the existing checkpoint dir for this task
CHECKPOINT_DIR=$(ls -d "${CHECKPOINT_PARENT}/exp_"*"_${GRID_NAME}_${TASK_ID}"/ 2>/dev/null | head -1)
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: no checkpoint dir matching exp_*_${GRID_NAME}_${TASK_ID} in ${CHECKPOINT_PARENT}" >&2
    exit 1
fi
CHECKPOINT_DIR="${CHECKPOINT_DIR%/}"

if ! ls "${CHECKPOINT_DIR}"/*_best.pth >/dev/null 2>&1; then
    echo "ERROR: no *_best.pth inside ${CHECKPOINT_DIR}" >&2
    exit 1
fi

# Get config for this array index (same grid the run was launched with)
CONFIG_JSON=$(python code/deep_learning/training/slurm/experiment_grid.py \
    --grid "$GRID_NAME" --index "$TASK_ID")
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to get config for grid=$GRID_NAME index=$TASK_ID" >&2
    exit 1
fi

# Convert JSON config to CLI args
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
echo "Resume: ${GRID_NAME} | Task: ${TASK_ID} | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Config: ${CONFIG_JSON}"
echo "Resume: ${CHECKPOINT_DIR}"
echo "Start:  $(date)"
echo "============================================================"

python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --resume "$CHECKPOINT_DIR" \
    $CLI_ARGS

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
