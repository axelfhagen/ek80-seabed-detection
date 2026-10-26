#!/bin/bash
#SBATCH --job-name=stride_comparison_pw512_128_64_32_16_1
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/stride_comparison_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/data_visualization/slurm/logs/stride_comparison_%j.err

# Sweep sliding-window stride (128, 64, 32, 16, 1) at fixed patch_width=512 and
# compare bottom line, IoU/Dice, convergence to the densest stride, and cost.
# Needs a GPU for U-Net inference; reads zarr test files (feb_2025 / jun_2025).
# Output: Figures/stride_comparison/

module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

PYTHONWARNINGS=ignore python code/data_visualization/gen_stride_comparison.py \
    --max-files 6 --strides 128,64,32,16,1
