#!/bin/bash
# Generate thesis prediction-grid galleries (Results 2x2, Appendix 3x3) for the
# chunk_arch_sweep_2 checkpoint on the Oct 2024 held-out survey.
#SBATCH --job-name=gen_prediction_grids_oct2024_arch_sweep_2
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

echo "Start: $(date)"
python code/data_visualization/gen_prediction_grids_thesis.py \
    --checkpoint-dir "${PROJECT}/code/checkpoints/chunk_arch_sweep_2" \
    --out-dir "${PROJECT}/Figures/prediction_grids"
echo "End: $(date)"
