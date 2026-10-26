#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/sweep_%A_%a.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/sweep_%A_%a.err

# ── Environment ───────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate master_env

# ── Configuration ─────────────────────────────────────────────────────
PROJECT=/cluster/work/axelfh/Master
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"
GRID_NAME="${1:?Usage: sbatch --array=1-N launch_sweep.sh <grid_name>}"
TASK_ID=$SLURM_ARRAY_TASK_ID

cd "$PROJECT"

# Rename log files with timestamp for easier identification
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/sweep_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.out" \
   "${LOG_DIR}/sweep_${GRID_NAME}_${TASK_ID}_${TIMESTAMP}_${SLURM_ARRAY_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/sweep_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.err" \
   "${LOG_DIR}/sweep_${GRID_NAME}_${TASK_ID}_${TIMESTAMP}_${SLURM_ARRAY_JOB_ID}.err" 2>/dev/null || true

# Get config for this array index (1-based)
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

CHECKPOINT_PARENT="${CHECKPOINT_PARENT:-code/checkpoints}"
CHECKPOINT_DIR="${CHECKPOINT_PARENT}/exp_${TIMESTAMP}_${GRID_NAME}_${TASK_ID}"

echo "============================================================"
echo "Sweep: ${GRID_NAME} | Task: ${TASK_ID} | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Config: ${CONFIG_JSON}"
echo "Start:  $(date)"
echo "============================================================"

python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR" \
    $CLI_ARGS

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
