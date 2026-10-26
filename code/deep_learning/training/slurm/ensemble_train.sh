#!/bin/bash
#SBATCH --job-name=ens-train
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/ens_train_%A_%a.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/ens_train_%A_%a.err

# ── Environment ───────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate master_env

# ── Configuration ─────────────────────────────────────────────────────
PROJECT=/cluster/work/axelfh/Master
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_datasets/ensemble_shared"

# ENSEMBLE_ID is passed via --export from submit_ensemble.sh
ENSEMBLE_ID="${ENSEMBLE_ID:?ERROR: ENSEMBLE_ID not set}"
MEMBER_ID="${SLURM_ARRAY_TASK_ID}"
SEED=$((42 + MEMBER_ID - 1))

CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_ensemble_${ENSEMBLE_ID}/member_${MEMBER_ID}"

# Rename log files with ensemble ID
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/ens_train_${SLURM_ARRAY_JOB_ID}_${MEMBER_ID}.out" \
   "${LOG_DIR}/ens_train_${ENSEMBLE_ID}_m${MEMBER_ID}_${SLURM_ARRAY_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/ens_train_${SLURM_ARRAY_JOB_ID}_${MEMBER_ID}.err" \
   "${LOG_DIR}/ens_train_${ENSEMBLE_ID}_m${MEMBER_ID}_${SLURM_ARRAY_JOB_ID}.err" 2>/dev/null || true

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

echo "============================================================"
echo "Ensemble Training — Member ${MEMBER_ID}"
echo "  Ensemble ID:    ${ENSEMBLE_ID}"
echo "  Seed:           ${SEED}"
echo "  Job ID:         ${SLURM_ARRAY_JOB_ID}_${MEMBER_ID}"
echo "  Node:           $(hostname)"
echo "  GPU:            $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Start:          $(date)"
echo "  Checkpoint dir: ${CHECKPOINT_DIR}"
echo "  Cache dir:      ${CACHE_DIR}"
echo "  Extra args:     ${EXTRA_ARGS}"
echo "============================================================"

python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --target-size 512 512 \
    --epochs 250 \
    --batch-size 4 \
    --freeze encoder \
    --early-stopping-patience 15 \
    --num-workers 4 \
    --seed "$SEED" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR" \
    $EXTRA_ARGS

echo "Member ${MEMBER_ID} finished: $(date)"
