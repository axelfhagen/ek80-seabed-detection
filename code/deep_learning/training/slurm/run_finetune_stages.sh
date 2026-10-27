#!/bin/bash
#SBATCH --job-name=unet-finetune-stages
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/finetune_stages_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/finetune_stages_%j.err

# ── Environment ───────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate master_env

# ── Configuration ─────────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/finetune_stages_${SLURM_JOB_ID}.out" "${LOG_DIR}/finetune_stages_${TIMESTAMP}_${SLURM_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/finetune_stages_${SLURM_JOB_ID}.err" "${LOG_DIR}/finetune_stages_${TIMESTAMP}_${SLURM_JOB_ID}.err" 2>/dev/null || true
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_${TIMESTAMP}_finetune_stages"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"

cd "$PROJECT"
mkdir -p "$CHECKPOINT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "Checkpoint dir: $CHECKPOINT_DIR"

# ── Training ──────────────────────────────────────────────────────────
# The finetune run's val loss got very noisy after unfreezing all 24M params
# at epoch 26. This uses the 'stages' strategy: freeze only the first 3
# stages (conv1+bn1, layer1, layer2 — universal features), while layer3
# and layer4 train from the start alongside the decoder. This gives ~40%
# trainable params — a middle ground that avoids the shock of a full unfreeze.
# Lower encoder LR (1e-5) protects the trainable encoder stages.
python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --epochs 250 \
    --batch-size 4 \
    --freeze stages \
    --encoder-lr 1e-5 \
    --decoder-lr 1e-3 \
    --early-stopping-patience 20 \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Finished: $(date)"
