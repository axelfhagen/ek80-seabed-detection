#!/bin/bash
#SBATCH --job-name=plot_ensemble_report_batch_resnet34_5files
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/%x_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"
mkdir -p code/data_visualization/slurm/logs

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python code/data_visualization/plot_ensemble_report_batch.py \
    --device cuda \
    --output-dir "${PROJECT}/Figures/ensemble/report_batch_v1" \
    --files \
        D20250201-T055359 \
        D20250213-T114505 \
        D20250213-T223315 \
        D20250213-T083929 \
        D20250626-T084615

echo "Finished: $(date)"
