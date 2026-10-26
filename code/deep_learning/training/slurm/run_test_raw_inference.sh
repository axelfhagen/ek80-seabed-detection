#!/bin/bash
#SBATCH --job-name=test-raw-infer
#SBATCH --account=share-ie-imf
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/test_raw_infer_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/test_raw_infer_%j.err

# Run U-Net inference on test_raw and generate 3x3 grid plots.
# Skips echogram generation (handled by the CPU job).
# Submit with: sbatch run_test_raw_inference.sh

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

echo "============================================================"
echo "Test raw inference | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "============================================================"

python code/infer_test_raw.py \
    --raw-dir /cluster/work/axelfh/test_raw \
    --checkpoint code/checkpoints/exp_20260511_191547_all_surveys_jun2025/smp_unet_resnet34_best.pth \
    --output-dir code/plots_test_raw \
    --max-pings 2000 \
    --skip-echograms

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
