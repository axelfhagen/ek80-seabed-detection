#!/bin/bash
# MC-Dropout uncertainty on calibrated zarr (Oct 2024 held-out survey),
# chunk_arch_sweep_2, replacing the earlier raw-input (out-of-distribution) run.
#SBATCH --job-name=mc_dropout_calibrated_oct2024_arch_sweep_2
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

echo "Start: $(date)"
python code/data_visualization/analyze_mc_dropout_uncertainty_zarr.py
echo "End: $(date)"
