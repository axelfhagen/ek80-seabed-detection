#!/bin/bash
#SBATCH --job-name=loss-sweep
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/loss_sweep_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/loss_sweep_%j.err

# ── Environment ───────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate master_env

# ── Configuration ─────────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/loss_sweep_${SLURM_JOB_ID}.out" "${LOG_DIR}/loss_sweep_${TIMESTAMP}_${SLURM_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/loss_sweep_${SLURM_JOB_ID}.err" "${LOG_DIR}/loss_sweep_${TIMESTAMP}_${SLURM_JOB_ID}.err" 2>/dev/null || true

cd "$PROJECT"

COMMON_ARGS=(
    --zarr-dir "$ZARR_DIR"
    --epochs 250
    --batch-size 4
    --freeze encoder
    --early-stopping-patience 15
    --num-workers 4
    --cache-dir "$CACHE_DIR"
)

echo "============================================================"
echo "Loss Function Sweep — Job $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "============================================================"

# ── Experiment 1: Dice-heavy (BCE 0.3 / Dice 0.7) ────────────────────
EXP=loss_01_dice_heavy
echo -e "\n>>> $EXP"
python code/deep_learning/training/train.py "${COMMON_ARGS[@]}" \
    --loss-type dice_bce \
    --bce-weight 0.3 \
    --checkpoint-dir "code/checkpoints/exp_${TIMESTAMP}_${EXP}"

# ── Experiment 2: BCE-heavy (BCE 0.7 / Dice 0.3) ─────────────────────
EXP=loss_02_bce_heavy
echo -e "\n>>> $EXP"
python code/deep_learning/training/train.py "${COMMON_ARGS[@]}" \
    --loss-type dice_bce \
    --bce-weight 0.7 \
    --checkpoint-dir "code/checkpoints/exp_${TIMESTAMP}_${EXP}"

# ── Experiment 3: Focal loss (α=0.25, γ=2) ───────────────────────────
EXP=loss_03_focal
echo -e "\n>>> $EXP"
python code/deep_learning/training/train.py "${COMMON_ARGS[@]}" \
    --loss-type focal \
    --focal-alpha 0.25 \
    --focal-gamma 2.0 \
    --checkpoint-dir "code/checkpoints/exp_${TIMESTAMP}_${EXP}"

# ── Experiment 4: Focal loss aggressive (α=0.5, γ=3) ─────────────────
EXP=loss_04_focal_aggressive
echo -e "\n>>> $EXP"
python code/deep_learning/training/train.py "${COMMON_ARGS[@]}" \
    --loss-type focal \
    --focal-alpha 0.5 \
    --focal-gamma 3.0 \
    --checkpoint-dir "code/checkpoints/exp_${TIMESTAMP}_${EXP}"

# ── Experiment 5: Focal + Dice hybrid (α=0.25, γ=2) ──────────────────
EXP=loss_05_focal_dice
echo -e "\n>>> $EXP"
python code/deep_learning/training/train.py "${COMMON_ARGS[@]}" \
    --loss-type focal_dice \
    --focal-alpha 0.25 \
    --focal-gamma 2.0 \
    --checkpoint-dir "code/checkpoints/exp_${TIMESTAMP}_${EXP}"

echo -e "\n============================================================"
echo "All experiments finished: $(date)"
echo "============================================================"
