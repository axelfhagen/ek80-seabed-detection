#!/bin/bash
# Rerun only the chunk_arch_sweep_2 patch conformal failure gallery.

#SBATCH --job-name=patch_conf_arch2
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/thesis_metrics_regen/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/thesis_metrics_regen/slurm/logs/%x_%j.err

set -euo pipefail
module purge
module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"
export PYTHONPATH="$PROJECT/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python -u code/data_visualization/gen_patch_conformal_failures.py \
    --checkpoint "$PROJECT/code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth" \
    --output-dir "$PROJECT/Latex/Figures/patch_conformal_failures/chunk_arch_sweep_2" \
    --calibration-split calibration \
    --test-split test \
    --n-each 2 \
    --calib-max-patches 160 \
    --test-max-patches 240 \
    --candidates-per-file 24 \
    --seed 0
