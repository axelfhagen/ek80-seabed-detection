#!/bin/bash
#SBATCH --job-name=all_surveys
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/all_surveys_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/all_surveys_%j.err

# Train on ALL surveys combined using the best-run hyperparameters.
# Patches are extracted on-the-fly and cached, so first run takes longer.
# Submit with: sbatch run_all_surveys.sh

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR="code/checkpoints/exp_${TIMESTAMP}_all_surveys"

echo "============================================================"
echo "All-survey training | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "============================================================"

python code/deep_learning/training/train.py \
    --zarr-dir Datasets/echopype_zarr \
    --cache-dir Datasets/patch_cache \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --encoder resnet34 \
    --freeze stages \
    --encoder-lr 1e-05 \
    --decoder-lr 0.001 \
    --scheduler-type plateau \
    --epochs 250 \
    --batch-size 4 \
    --num-workers 4 \
    --early-stopping-patience 20 \
    --patch-width 512 \
    --stride 512 \
    --target-size 512 512 \
    --in-channels 3 \
    --freq1 38000 \
    --freq2 200000 \
    --bce-weight 0.5 \
    --seed 42

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
