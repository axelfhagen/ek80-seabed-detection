#!/bin/bash
#SBATCH --job-name=worst_bins_spotlight_pred
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/%x_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"

PRED_PARQUET="${PRED_PARQUET:-Figures/conformal/labelless_pred_fullrange/ping_table.parquet}"

# Regenerates Figures/failure_mining/worst_bins_spotlight_pred.pdf from the
# full-range predicted-bottom crossing table.
python code/data_visualization/analyze_failure_mining.py \
    --source pred \
    --pred-parquet "$PRED_PARQUET"
