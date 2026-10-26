#!/bin/bash
#SBATCH --job-name=retrain_arch6_vanilla_unet_large_fromscratch
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

# Clean from-scratch retrain of arch_sweep task 6 (vanilla_unet_large), the
# architecture that previously reached val_IoU=0.9965 before a resume-time NaN
# collapse (2026-05-21, job 24549068). That collapse was an infrastructure bug,
# not an architecture problem: the original run was stable for 31+ epochs and
# only stopped on the 4h wall. The fixes (resume-stable RNG, grad clipping,
# non-finite-loss guard) landed in commit 44f06fa the next day.
#
# This script trains in a SINGLE job (--time 10h) so there is no resume boundary
# at all — the cleanest way to produce a trustworthy curve for the thesis.
# grad-clip-norm is passed explicitly (it is also the config default of 1.0) so
# a single runaway step can never blow the run to NaN again.

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"
GRID_NAME="arch_sweep"
TASK_ID=6

cd "$PROJECT"

# Fresh, timestamped checkpoint dir — never overwrite the old (intact) best.pth.
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_${SLURM_JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CHECKPOINT_DIR"

# Pull the exact arch_6 config from the grid so it stays in sync with the sweep.
CONFIG_JSON=$(python code/deep_learning/training/slurm/experiment_grid.py \
    --grid "$GRID_NAME" --index "$TASK_ID")
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
echo "Retrain (from scratch): ${GRID_NAME} | Task: ${TASK_ID} | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Config: ${CONFIG_JSON}"
echo "Ckpt:   ${CHECKPOINT_DIR}"
echo "Start:  $(date)"
echo "============================================================"

# Trailing overrides win (argparse keeps the last value):
#   --grad-clip-norm 1.0     explicit anti-NaN guard (also the default)
#   --early-stopping-patience 25  a little more room than the grid's 15
python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR" \
    $CLI_ARGS \
    --grad-clip-norm 1.0 \
    --early-stopping-patience 25

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
