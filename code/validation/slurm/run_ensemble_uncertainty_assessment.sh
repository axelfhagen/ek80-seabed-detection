#!/bin/bash
#SBATCH --job-name=analyze_ensemble_uncertainty_febjun2025_div5
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/validation/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/validation/slurm/logs/%x_%j.err

set -euo pipefail
module purge
module load Anaconda3/2024.02-1
conda activate master_env

PROJECT=/cluster/work/axelfh/Master
cd "$PROJECT"

echo "===================================================================="
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "===================================================================="

# Pooled disagreement-vs-error analysis for the div5 ensemble over the full
# feb+jun 2025 test set. Output -> Figures/ensemble/uncertainty_assessment/
python -u code/data_visualization/analyze_ensemble_uncertainty.py \
    --device cuda "$@"

echo "===================================================================="
echo "Finished: $(date)"
echo "===================================================================="
