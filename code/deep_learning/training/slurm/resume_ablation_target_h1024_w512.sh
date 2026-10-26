#!/bin/bash
#SBATCH --job-name=resume_ablation_target_h1024_w512
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/resume_ablation_target_h1024_w512_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/resume_ablation_target_h1024_w512_%j.err

# Resume the asymmetric 1024(H)x512(W) ablation run that timed out at 6h.
# Picks up from the best checkpoint in exp_20260522_214015_ablation_target_h1024_w512.

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
ZARR_DIR="${PROJECT}/Datasets/echopype_zarr"
CACHE_DIR="${PROJECT}/Datasets/patch_cache"
CHECKPOINT_DIR="${PROJECT}/code/checkpoints/exp_20260522_214015_ablation_target_h1024_w512"

cd "$PROJECT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/resume_ablation_target_h1024_w512_${SLURM_JOB_ID}.out" \
   "${LOG_DIR}/resume_ablation_target_h1024_w512_${TIMESTAMP}_${SLURM_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/resume_ablation_target_h1024_w512_${SLURM_JOB_ID}.err" \
   "${LOG_DIR}/resume_ablation_target_h1024_w512_${TIMESTAMP}_${SLURM_JOB_ID}.err" 2>/dev/null || true

echo "============================================================"
echo "Resume: asymmetric target 1024(H) x 512(W)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Resume: $CHECKPOINT_DIR"
echo "Start:  $(date)"
echo "============================================================"

python code/deep_learning/training/train.py \
    --zarr-dir "$ZARR_DIR" \
    --encoder resnet34 \
    --loss-type dice_bce \
    --bce-weight 0.5 \
    --patch-width 512 \
    --stride 512 \
    --target-size 1024 512 \
    --epochs 250 \
    --batch-size 4 \
    --freeze encoder \
    --scheduler-type plateau \
    --early-stopping-patience 15 \
    --num-workers 4 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --resume "$CHECKPOINT_DIR"

echo "Finished: $(date)"
