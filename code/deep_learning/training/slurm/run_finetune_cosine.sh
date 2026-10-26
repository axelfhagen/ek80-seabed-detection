#!/bin/bash
#SBATCH --job-name=unet-ft-cosine
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/finetune_cosine_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/finetune_cosine_%j.err

# ── Environment ───────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate master_env

# ── Configuration ─────────────────────────────────────────────────────
PROJECT=/cluster/work/axelfh/Master
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/finetune_cosine_${SLURM_JOB_ID}.out" "${LOG_DIR}/finetune_cosine_${TIMESTAMP}_${SLURM_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/finetune_cosine_${SLURM_JOB_ID}.err" "${LOG_DIR}/finetune_cosine_${TIMESTAMP}_${SLURM_JOB_ID}.err" 2>/dev/null || true
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_finetune_cosine"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Checkpoint dir: $CHECKPOINT_DIR"

# ── Training ──────────────────────────────────────────────────────────
# Combines the two best strategies: finetune (best bottom error at 3.27m)
# with a gentle cosine scheduler (fixed). Phase 1: frozen encoder for 20
# epochs with cosine (T_0=20 matches the phase length). Phase 2: unfreeze
# at epoch 20, cosine resets with the new optimizer, T_0=30 gives a long
# annealing cycle for the full model. The original finetune's ReduceLR
# was too reactive post-unfreeze; cosine provides a smoother LR trajectory.
python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --epochs 250 \
    --batch-size 4 \
    --freeze encoder \
    --unfreeze-after 20 \
    --encoder-lr 1e-5 \
    --decoder-lr 1e-3 \
    --early-stopping-patience 25 \
    --scheduler-type cosine \
    --cosine-t0 30 \
    --cosine-t-mult 2 \
    --cosine-eta-min 1e-6 \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Finished: $(date)"
