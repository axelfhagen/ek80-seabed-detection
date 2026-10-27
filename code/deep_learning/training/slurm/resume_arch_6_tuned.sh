#!/bin/bash
#SBATCH --job-name=resume_arch_6_tuned
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/resume_%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/resume_%x_%j.err

# Resume arch_sweep task 6 (vanilla_unet_large) from its best checkpoint.
# This model's training/validation curves track each other closely (unlike the
# other arch_sweep runs that diverge after a few epochs), so it is worth
# training further. The saved checkpoint is the best epoch (25), not the
# early-stop point, so resuming continues from peak weights.
#
# Two overrides vs. the original grid: a longer early-stopping patience (25)
# and a slower LR-plateau scheduler (lr-patience 10) to give the run more room
# before it stops and before each LR reduction.

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"
GRID_NAME="arch_sweep"
TASK_ID=6
CHECKPOINT_PARENT="${PROJECT}/code/checkpoints"

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
    args.append(str(v))
print(' '.join(args))
" "$CONFIG_JSON")

echo "============================================================"
echo "Resume (tuned): ${GRID_NAME} | Task: ${TASK_ID} | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Config: ${CONFIG_JSON}"
echo "Resume: ${CHECKPOINT_DIR}"
echo "Overrides: early-stopping-patience=25 lr-patience=10"
echo "Start:  $(date)"
echo "============================================================"

# Trailing overrides win over $CLI_ARGS (argparse keeps the last value).
python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --resume "$CHECKPOINT_DIR" \
    $CLI_ARGS \
    --early-stopping-patience 25 \
    --lr-patience 10

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
