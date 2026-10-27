#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:45:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/eval_%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/eval_%x_%j.err

# Eval-only backfill for sweep tasks whose training timed out before the
# evaluation/diagnostics stage. Loads each task's saved *_best.pth and runs
# run_evaluation + generate_diagnostics (the 13-plot panel) without training.
# Writes eval_metrics.json + diagnostics/ into the checkpoint dir. Note: train.py
# intentionally does NOT append to results.csv in --eval-only mode; the canonical
# results.csv row is written by the full 6h resume run (resume_sweep.sh).
#
# Usage (per task):
#   cd code/deep_learning/training/slurm
#   sbatch --job-name=eval_arch_1 eval_checkpoints.sh arch_sweep 1

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"
GRID_NAME="${1:?Usage: sbatch eval_checkpoints.sh <grid_name> <task_id>}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${2:?Usage: sbatch eval_checkpoints.sh <grid_name> <task_id>}}"
CHECKPOINT_PARENT="${CHECKPOINT_PARENT:-${PROJECT}/code/checkpoints}"

cd "$PROJECT"

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
    if isinstance(v, list):
        args.extend(str(x) for x in v)
    else:
        args.append(str(v))
print(' '.join(args))
" "$CONFIG_JSON")

echo "============================================================"
echo "Eval-only: ${GRID_NAME} | Task: ${TASK_ID} | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Resume: ${CHECKPOINT_DIR}"
echo "Start:  $(date)"
echo "============================================================"

python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --resume "$CHECKPOINT_DIR" \
    --eval-only \
    $CLI_ARGS

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
