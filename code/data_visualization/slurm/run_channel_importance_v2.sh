#!/bin/bash
#SBATCH --job-name=chan-imp-v2
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/chan_imp_v2_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/chan_imp_v2_%j.err

# Per-channel importance figures (v2) for the U-Net seabed segmenter.
# Needs a GPU for U-Net inference / gradient attribution; reads the Jun 2025
# patch caches that run_gen_jun2025_caches.sh builds.

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"

PYTHONWARNINGS=ignore python code/data_visualization/analyze_channel_importance_v2.py
