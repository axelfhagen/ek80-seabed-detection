#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Inference for two thesis figures (adapted from slurm_template.sh):
#   FIG 1  cross-survey generalization — staged ResNet-34 over all 9 surveys
#   FIG 5  per-column edge degradation — single-patch inference, val + test
# Both analyze scripts cache their results, then plot; re-plotting is login-cheap.
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=eval_thesis_figs_crosssurvey_edges
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

echo "[FIG 1] cross-survey generalization ..."
python code/data_visualization/analyze_cross_survey_generalization.py \
    --compute --max-files 25

echo "[FIG 5] per-column edge degradation ..."
python code/data_visualization/analyze_edge_degradation.py \
    --compute --max-patches 600 --batch 8

echo "════════════════════════════════════════════════════════════════════"
echo "Finished: $(date)"
echo "════════════════════════════════════════════════════════════════════"
