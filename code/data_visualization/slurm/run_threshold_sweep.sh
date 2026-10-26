#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Classification-threshold sensitivity sweep for the best U-Net
# (adapted from slurm_template.sh). Runs analyze_threshold_sweep.py over the
# held-out validation survey: sweeps the binarisation threshold 0.05–0.95 and
# reports IoU/Dice/precision/recall and bottom-depth deviation vs threshold.
# Output: Figures/threshold_sweep/
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=eval_threshold_sweep_arch_sweep_2
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

python code/data_visualization/analyze_threshold_sweep.py

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
