#!/bin/bash
#SBATCH --job-name=viz-labels
#SBATCH --account=share-ie-imf
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=code/deep_learning/training/slurm/logs/viz_labels_%j.out
#SBATCH --error=code/deep_learning/training/slurm/logs/viz_labels_%j.err

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT}/code/deep_learning/training/slurm/logs"
mv "${LOG_DIR}/viz_labels_${SLURM_JOB_ID}.out" "${LOG_DIR}/viz_labels_${TIMESTAMP}_${SLURM_JOB_ID}.out" 2>/dev/null || true
mv "${LOG_DIR}/viz_labels_${SLURM_JOB_ID}.err" "${LOG_DIR}/viz_labels_${TIMESTAMP}_${SLURM_JOB_ID}.err" 2>/dev/null || true

python code/data_visualization/visualize_labels.py \
    --zarr-dir "${PROJECT}/Datasets/echopype_zarr" \
    --raw-dir "${PROJECT}/Datasets/raw_files/drive-download-20260317T113707Z-1-001" \
    --output-dir "${PROJECT}/label_overview"
