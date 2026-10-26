#!/bin/bash
# Rerun only the chunk_arch_sweep_2 held-out hard-patch gallery.

#SBATCH --job-name=patch_prob_chunk_arch2
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

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"
export PYTHONPATH="$PROJECT/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python -u code/data_visualization/gen_patch_prob_eval.py \
    --checkpoint "$PROJECT/code/checkpoints/chunk_arch_sweep_2/smp_unet_resnet34_best.pth" \
    --output-dir "$PROJECT/Latex/Figures/patch_prob_eval/chunk_arch_sweep_2_rgb_hard" \
    --split test \
    --display rgb \
    --selection hardest \
    --n 12 \
    --candidates-per-file 8 \
    --max-candidates 128 \
    --batch-size 8 \
    --seed 0
