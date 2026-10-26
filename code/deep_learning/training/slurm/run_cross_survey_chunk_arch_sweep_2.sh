#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Cross-survey generalization figure — Bastø Standard U-Net (chunk_arch_sweep_2).
# Re-runs the cross-survey inference/eval with the chunk_arch_sweep_2 checkpoint
# (smp_unet ResNet-34 from scratch) on the SAME per-survey file sample, split
# roles, sliding-window recipe, reference masks, and metrics as the original
# staged-ResNet-34 figure. Caches per-file metrics, then plots + writes CSV
# and a per-survey text summary.
# Adapted from run_thesis_figure_inference.sh / slurm_template.sh.
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=cross_survey_generalization_chunk_arch_sweep_2
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/deep_learning/training/slurm/logs/%x_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

echo "════════════════════════════════════════════════════════════════════"
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname)"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
fi
echo "Start:  $(date)"
echo "════════════════════════════════════════════════════════════════════"

python code/data_visualization/analyze_cross_survey_generalization_v2.py \
    --compute --max-files 25

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
