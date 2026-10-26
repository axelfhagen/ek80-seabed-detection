#!/bin/bash
#SBATCH --job-name=bottom-extract-compare
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/bottom_extract_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/bottom_extract_%j.err

module load Anaconda3/2024.02-1
conda activate master_env

cd /cluster/work/axelfh/Master
mkdir -p code/deep_learning/training/slurm/logs

echo "============================================================"
echo "Bottom extraction method comparison | Job: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  $(date)"
echo "============================================================"

# Runs full-echogram inference over the eval set, derives the baseline + three
# robust bottom lines, and writes 9 grids + metrics.json to
# Figures/bottom_extraction_methods/. Probability maps are cached under
# Figures/bottom_extraction_methods/_cache/ so re-runs (e.g. --replot-only) are fast.
python -u code/data_visualization/compare_bottom_extraction.py --device cuda

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
