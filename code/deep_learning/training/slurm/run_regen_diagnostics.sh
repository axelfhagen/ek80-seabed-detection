#!/bin/bash
# Regenerate post-training diagnostics for one checkpoint dir per array task.
#
# Reads a manifest file (one checkpoint directory per line) and runs the
# diagnostics CLI on the directory named on line $SLURM_ARRAY_TASK_ID. Use the
# companion submit_regen_diagnostics.sh wrapper to build the manifest and size
# the array; it is the intended entry point.
#
#   sbatch --array=1-N%6 run_regen_diagnostics.sh <manifest> [split]
#
# split defaults to "survey" (the canonical train/val/cal/test split). Single-
# survey or jun2025 runs used a different val set and should not be regenerated
# with the survey split — see submit_regen_diagnostics.sh.
#SBATCH --job-name=regen_diag
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%A_%a.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%A_%a.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

MANIFEST="${1:?Usage: sbatch --array=1-N run_regen_diagnostics.sh <manifest> [split]}"
SPLIT="${2:-survey}"

CKPT_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: no checkpoint dir on line ${SLURM_ARRAY_TASK_ID} of ${MANIFEST}" >&2
    exit 1
fi

echo "════════════════════════════════════════════════════════════════════"
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})"
echo "Target: ${CKPT_DIR}"
echo "Split:  ${SPLIT}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start:  $(date)"
echo "════════════════════════════════════════════════════════════════════"

python code/deep_learning/training/diagnostics.py \
    --checkpoint-dir "$CKPT_DIR" \
    --split "$SPLIT"

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
