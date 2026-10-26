#!/bin/bash
#SBATCH --job-name=label_overview
#SBATCH --account=share-ie-imf
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/label_overview_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/label_overview_%j.err

# ── Environment ───────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
conda activate master_env

cd /cluster/work/axelfh/Master

python code/data_visualization/visualize_label_panels.py --all \
    --zarr-dir Datasets/echopype_zarr \
    --output-dir Figures/label_overview

echo "Job finished at $(date)"
