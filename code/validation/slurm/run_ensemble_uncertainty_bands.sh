#!/bin/bash
#SBATCH --job-name=plot_ensemble_uncertainty_bands_feb2025_div5
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
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

# Ensemble uncertainty-band visualisation on feb_2025 test files using the
# div5 heterogeneous ensemble: member spread, conformal bands, disagreement
# vs error. Output -> Figures/ensemble/uncertainty_bands/
python -u code/data_visualization/plot_ensemble_uncertainty_bands.py \
    --device cuda "$@"

echo "===================================================================="
echo "Finished: $(date)"
echo "===================================================================="
