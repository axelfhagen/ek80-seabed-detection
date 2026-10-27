#!/bin/bash
#SBATCH --job-name=ens-eval
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/ens_eval_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/ens_eval_%j.err

# ── Environment ───────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate master_env

# ── Configuration ─────────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"

ENSEMBLE_ID="${ENSEMBLE_ID:?ERROR: ENSEMBLE_ID not set}"

CKPT_PARENT="${PROJECT}/code/checkpoints/exp_ensemble_${ENSEMBLE_ID}"
OUTPUT_DIR="${PROJECT}/code/ensemble_results/ensemble_${ENSEMBLE_ID}"

# Rename log files
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/ens_eval_${SLURM_JOB_ID}.out" \
   "${LOG_DIR}/ens_eval_${ENSEMBLE_ID}_${SLURM_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/ens_eval_${SLURM_JOB_ID}.err" \
   "${LOG_DIR}/ens_eval_${ENSEMBLE_ID}_${SLURM_JOB_ID}.err" 2>/dev/null || true

cd "$PROJECT"

# Collect checkpoints
CHECKPOINTS=$(find "$CKPT_PARENT" -name "unet_*_best.pth" | sort)
N_FOUND=$(echo "$CHECKPOINTS" | wc -l)

echo "============================================================"
echo "Ensemble Evaluation"
echo "  Ensemble ID:  ${ENSEMBLE_ID}"
echo "  Checkpoints:  ${N_FOUND} found"
echo "  Job ID:       ${SLURM_JOB_ID}"
echo "  Node:         $(hostname)"
echo "  GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Start:        $(date)"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "============================================================"
echo ""
echo "Checkpoint paths:"
echo "$CHECKPOINTS"
echo ""

if [ "$N_FOUND" -lt 2 ]; then
    echo "ERROR: Found only ${N_FOUND} checkpoint(s), need at least 2."
    exit 1
fi

python code/deep_learning/training/evaluate_ensemble.py \
    --checkpoints $CHECKPOINTS \
    --zarr-dir "$ZARR_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "Ensemble evaluation finished: $(date)"
