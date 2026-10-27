#!/bin/bash
#SBATCH --job-name=resume_lowlr
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/resume_%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/resume_%x_%j.err

# Non-destructive resume at a lower LR, to test whether a converged sweep task
# keeps improving with more training. Unlike resume_sweep.sh (which writes back
# into the original exp dir), this COPIES the task's checkpoint dir to a
# "<orig>_lowlr" sibling and trains the copy — the original *_best.pth and
# training_history.json are never overwritten.
#
# The copied optimizer state would normally pin the old LR on resume, so we pass
# --reset-lr-on-resume to re-apply the CLI --decoder-lr/--encoder-lr.
#
# Usage:
#   cd code/deep_learning/training/slurm
#   sbatch --job-name=lowlr_geo_2 resume_lower_lr.sh geo_seg_test 2
#   DECODER_LR=1e-4 ES_PATIENCE=40 sbatch --job-name=lowlr_geo_3 resume_lower_lr.sh geo_seg_test 3
#
# Env overrides (with defaults):
#   DECODER_LR=3e-4   ENCODER_LR=(unset, keeps grid default)   ES_PATIENCE=30

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"
GRID_NAME="${1:?Usage: sbatch resume_lower_lr.sh <grid_name> <task_id>}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${2:?Usage: sbatch resume_lower_lr.sh <grid_name> <task_id>}}"
CHECKPOINT_PARENT="${CHECKPOINT_PARENT:-${PROJECT}/code/checkpoints}"
DECODER_LR="${DECODER_LR:-3e-4}"
ES_PATIENCE="${ES_PATIENCE:-30}"

cd "$PROJECT"

# Locate the original checkpoint dir for this task
ORIG_DIR=$(ls -d "${CHECKPOINT_PARENT}/exp_"*"_${GRID_NAME}_${TASK_ID}"/ 2>/dev/null | head -1)
if [ -z "$ORIG_DIR" ]; then
    echo "ERROR: no checkpoint dir matching exp_*_${GRID_NAME}_${TASK_ID} in ${CHECKPOINT_PARENT}" >&2
    exit 1
fi
ORIG_DIR="${ORIG_DIR%/}"
if ! ls "${ORIG_DIR}"/*_best.pth >/dev/null 2>&1; then
    echo "ERROR: no *_best.pth inside ${ORIG_DIR}" >&2
    exit 1
fi

# Copy to a sibling dir so the original is never written to. Reuse an existing
# copy if this task was already launched (so a re-run continues the lowlr run).
LOWLR_DIR="${ORIG_DIR}_lowlr"
if [ ! -e "$LOWLR_DIR" ]; then
    echo "Copying ${ORIG_DIR} -> ${LOWLR_DIR}"
    cp -r "$ORIG_DIR" "$LOWLR_DIR"
else
    echo "Reusing existing copy ${LOWLR_DIR}"
fi

# Config for this task (same grid the run was launched with)
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

# Optional encoder-lr override (only when set; grid uses frozen encoder by default)
ENCODER_LR_ARG=""
if [ -n "${ENCODER_LR:-}" ]; then
    ENCODER_LR_ARG="--encoder-lr ${ENCODER_LR}"
fi

echo "============================================================"
echo "Lower-LR resume: ${GRID_NAME} | Task: ${TASK_ID} | Job: ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Original:    ${ORIG_DIR}  (read-only, not overwritten)"
echo "Writing to:  ${LOWLR_DIR}"
echo "decoder_lr=${DECODER_LR}  ${ENCODER_LR_ARG}  early-stop=${ES_PATIENCE}"
echo "Start:       $(date)"
echo "============================================================"

# CLI args after $CLI_ARGS override the grid values (argparse: last wins).
python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --num-workers 4 \
    --cache-dir "$CACHE_DIR" \
    --checkpoint-dir "$LOWLR_DIR" \
    --resume "$LOWLR_DIR" \
    --reset-lr-on-resume \
    $CLI_ARGS \
    --decoder-lr "$DECODER_LR" \
    $ENCODER_LR_ARG \
    --early-stopping-patience "$ES_PATIENCE"

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
