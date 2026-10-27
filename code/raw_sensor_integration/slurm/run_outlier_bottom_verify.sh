#!/bin/bash
#SBATCH --job-name=outlier-bottom-verify
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/raw_sensor_integration/slurm/logs/outlier_bottom_verify_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/raw_sensor_integration/slurm/logs/outlier_bottom_verify_%j.err

set -euo pipefail

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"

mkdir -p code/raw_sensor_integration/slurm/logs
mkdir -p Figures/geographic_outliers/v4

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

python -u code/data_visualization/visualize_outlier_bottom_detection.py

echo "Finished: $(date)"
