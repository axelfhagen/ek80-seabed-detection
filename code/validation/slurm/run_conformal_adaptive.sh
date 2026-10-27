#!/bin/bash
#SBATCH --job-name=conformal_adaptive
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/work/axelfh/Master/code/validation/slurm/logs/%x_%j.out
#SBATCH --error=/cluster/work/axelfh/Master/code/validation/slurm/logs/%x_%j.err

set -euo pipefail
module purge
module load Anaconda3/2024.02-1
conda activate master_env

REPO_ROOT="${REPO_ROOT:-/cluster/work/axelfh/Master}"
PROJECT="$REPO_ROOT"
cd "$PROJECT"

echo "===================================================================="
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "===================================================================="

# Sigma-threshold sweep (locally-weighted) + Mondrian binned by backscatter
# and by bottom-line slope, on the exchangeable jan/feb/jun_2025 split.
python -u code/validation/generate_conformal_adaptive.py \
    --n-calib 60 \
    --n-test 60 \
    --n-bins 5 \
    --max-plot-files 6 \
    --device auto \
    "$@"

echo "===================================================================="
echo "Finished: $(date)"
echo "===================================================================="
