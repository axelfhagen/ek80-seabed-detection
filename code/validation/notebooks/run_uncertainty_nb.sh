#!/bin/bash
#SBATCH --job-name=uncertainty-nb
#SBATCH --account=share-ie-imf
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/uncertainty_nb_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/uncertainty_nb_%j.err

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
NB="$REPO_ROOT/code/validation/notebooks/06_validation_uncertainty.ipynb"

jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 \
    --ExecutePreprocessor.kernel_name=master_env \
    "$NB" --output 06_validation_uncertainty.ipynb

echo "Exit code: $?"
