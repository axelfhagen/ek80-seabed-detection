#!/bin/bash
#SBATCH --job-name=patch_width_comparison_256_vs_512
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/patch_width_comparison_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/patch_width_comparison_%j.err

# Compare full-echogram inference at patch_width=256 (old, cyan) vs 512 (new,
# red, matches training) with stride=128. Needs a GPU for U-Net inference;
# reads zarr test files (feb_2025 / jun_2025 surveys).
# Output: Figures/patch_width_comparison/

module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"

PYTHONWARNINGS=ignore python code/data_visualization/gen_patch_width_comparison.py --max-files 6
